from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, TextIO, cast
from uuid import UUID

from sqlalchemy import Engine, text

from asr_lab.evaluators import AsrEvaluationWorker
from infrastructure.huggingface import resolve_local_snapshot


class AsrTrainingWorker:
    """Prepare immutable Qwen data and supervise one isolated LoRA subprocess."""

    def __init__(
        self,
        engine: Engine,
        *,
        storage_root: Path,
        model_cache_root: Path,
        training_python: Path,
        training_script: Path,
    ) -> None:
        self._engine = engine
        self._storage_root = storage_root.resolve()
        self._model_cache_root = model_cache_root.resolve()
        self._training_python = training_python.resolve()
        self._training_script = training_script.resolve()

    def run_once(self) -> bool:
        run_id = self._claim()
        if run_id is None:
            return False
        try:
            self._execute(run_id)
        except Exception as error:
            self._fail(run_id, str(error))
        return True

    def _claim(self) -> UUID | None:
        with self._engine.begin() as connection:
            value = connection.execute(
                text(
                    """
                    with candidate as (
                        select id from training_runs
                        where status = 'queued'
                        order by created_at
                        for update skip locked
                        limit 1
                    )
                    update training_runs runs
                    set status = 'preparing', started_at = coalesce(started_at, now()),
                        progress_percent = 1, progress_message = '准备冻结训练数据',
                        error_message = null, updated_at = now()
                    from candidate
                    where runs.id = candidate.id
                    returning runs.id
                    """
                )
            ).scalar_one_or_none()
        return None if value is None else UUID(str(value))

    def _execute(self, run_id: UUID) -> None:
        if not self._training_python.is_file():
            raise FileNotFoundError(f"ASR Lab training Python does not exist: {self._training_python}")
        if not self._training_script.is_file():
            raise FileNotFoundError(f"ASR Lab LoRA script does not exist: {self._training_script}")
        with self._engine.connect() as connection:
            run = dict(
                connection.execute(
                    text(
                        """
                        select runs.*, models.base_model_name
                        from training_runs runs
                        join model_versions models on models.id = runs.base_model_version_id
                        where runs.id = :run_id
                        """
                    ),
                    {"run_id": run_id},
                ).mappings().one()
            )
            cases = [
                dict(row)
                for row in connection.execute(
                    text(
                        """
                        select cases.*, assets.artifact_uri, assets.recording_id,
                               recordings.storage_path as recording_storage_path
                        from evaluation_cases cases
                        join evaluation_source_assets assets on assets.id = cases.source_asset_id
                        left join recordings on recordings.id = assets.recording_id
                        where cases.dataset_version_id = :version_id
                          and cases.split in ('train', 'validation')
                        order by cases.split, cases.id
                        """
                    ),
                    {"version_id": run["dataset_version_id"]},
                ).mappings()
            ]
        run_root = self._storage_root / "asr-lab" / "training" / str(run_id)
        audio_root = run_root / "dataset" / "audio"
        output_root = run_root / "output"
        audio_root.mkdir(parents=True, exist_ok=True)
        output_root.mkdir(parents=True, exist_ok=True)
        train_file = run_root / "dataset" / "train.jsonl"
        validation_file = run_root / "dataset" / "validation.jsonl"
        with train_file.open("w", encoding="utf-8") as train_output, validation_file.open("w", encoding="utf-8") as validation_output:
            for index, case in enumerate(cases, start=1):
                if self._cancel_requested(run_id):
                    self._cancel(run_id)
                    return
                self._materialize_case(case, audio_root, train_output if case["split"] == "train" else validation_output)
                self._progress(run_id, min(20, 2 + round(index / max(1, len(cases)) * 18)), f"准备训练样本 {index}/{len(cases)}")

        base_path = resolve_local_snapshot(cast(str, run["base_model_name"]), self._model_cache_root)
        manifest: dict[str, str] = {
            "run_id": str(run_id),
            "dataset_version_id": str(run["dataset_version_id"]),
            "base_model": str(base_path),
            "preset": run["preset_name"],
            "training_method": "lora",
            "train_file": str(train_file),
            "validation_file": str(validation_file),
            "output_dir": str(output_root),
        }
        (run_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        command = [
            str(self._training_python),
            str(self._training_script),
            "--model_path", str(base_path),
            "--train_file", str(train_file),
            "--output_dir", str(output_root),
            "--rank", "16",
            "--alpha", "32",
            "--dropout", "0.05",
            "--batch_size", "1",
            "--grad_acc", "16",
            "--lr", "2e-4",
            "--epochs", "3",
        ]
        if validation_file.stat().st_size > 0:
            command.extend(("--eval_file", str(validation_file)))
        log_path = run_root / "training.log"
        self._progress(run_id, 25, "启动独立 LoRA 训练进程", status="training")
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=self._training_script.parent, stdout=log, stderr=subprocess.STDOUT, text=True)
            while process.poll() is None:
                if self._cancel_requested(run_id):
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    self._cancel(run_id)
                    return
                time.sleep(2)
        if process.returncode != 0:
            tail = self._log_tail(log_path)
            raise RuntimeError(f"LoRA training exited with code {process.returncode}: {tail}")
        adapter_config = output_root / "adapter_config.json"
        if not adapter_config.is_file():
            raise RuntimeError("LoRA training completed without adapter_config.json")
        adapter_uri = output_root.relative_to(self._storage_root).as_posix()
        self._progress(run_id, 95, "注册候选模型", status="validating")
        with self._engine.begin() as connection:
            model_id = UUID(str(connection.execute(
                text(
                    """
                    insert into model_versions (
                        workspace_id, model_family, name, version, base_model_name,
                        adapter_uri, training_run_id, status, runtime_config, metadata,
                        created_by_user_id
                    )
                    values (
                        :workspace_id, 'qwen3_asr', :name, :version, :base_model_name,
                        :adapter_uri, :training_run_id, 'candidate',
                        '{"provider":"qwen_asr"}'::jsonb,
                        cast(:metadata as jsonb), :user_id
                    )
                    returning id
                    """
                ),
                {
                    "workspace_id": run["workspace_id"],
                    "name": run["candidate_model_name"],
                    "version": f"run-{str(run_id)[:8]}",
                    "base_model_name": run["base_model_name"],
                    "adapter_uri": adapter_uri,
                    "training_run_id": run_id,
                    "metadata": json.dumps({"manifest_uri": (run_root / "manifest.json").relative_to(self._storage_root).as_posix()}),
                    "user_id": run["created_by_user_id"],
                },
            ).scalar_one()))
            validation_case_count = cast(
                int,
                connection.execute(
                    text(
                        """
                        select count(*)
                        from evaluation_cases
                        where dataset_version_id = :version_id and split = 'validation'
                        """
                    ),
                    {"version_id": run["dataset_version_id"]},
                ).scalar_one(),
            )
            validation_run_id: UUID | None = None
            if validation_case_count > 0:
                validation_run_id = UUID(
                    str(
                        connection.execute(
                            text(
                                """
                                insert into evaluation_runs (
                                    workspace_id, dataset_version_id, evaluator_type, split,
                                    idempotency_key, config_snapshot, total_case_count,
                                    created_by_user_id
                                )
                                values (
                                    :workspace_id, :dataset_version_id, 'asr', 'validation',
                                    :idempotency_key, cast(:config_snapshot as jsonb),
                                    :total_case_count, :user_id
                                )
                                returning id
                                """
                            ),
                            {
                                "workspace_id": run["workspace_id"],
                                "dataset_version_id": run["dataset_version_id"],
                                "idempotency_key": f"training-validation:{run_id}",
                                "config_snapshot": json.dumps(
                                    {
                                        "normalization_name": "zh_asr",
                                        "normalization_version": "v1",
                                        "source_training_run_id": str(run_id),
                                    },
                                    separators=(",", ":"),
                                ),
                                "total_case_count": validation_case_count * 2,
                                "user_id": run["created_by_user_id"],
                            },
                        ).scalar_one()
                    )
                )
                connection.execute(
                    text(
                        """
                        insert into evaluation_run_models (
                            evaluation_run_id, model_version_id, role, position
                        )
                        values
                            (:run_id, :base_model_id, 'baseline', 0),
                            (:run_id, :candidate_model_id, 'candidate', 1)
                        """
                    ),
                    {
                        "run_id": validation_run_id,
                        "base_model_id": run["base_model_version_id"],
                        "candidate_model_id": model_id,
                    },
                )
            connection.execute(
                text(
                    """
                    update training_runs
                    set status = 'succeeded', progress_percent = 100,
                        progress_message = :progress_message,
                        output_uri = :output_uri, finished_at = now(), updated_at = now()
                    where id = :run_id
                    """
                ),
                {
                    "run_id": run_id,
                    "output_uri": adapter_uri,
                    "progress_message": (
                        "LoRA 训练完成，候选模型已注册并进入验证评测"
                        if validation_run_id is not None
                        else "LoRA 训练完成，候选模型已注册；数据集没有 validation case"
                    ),
                },
            )

    def _materialize_case(self, case: dict[str, Any], audio_root: Path, output: TextIO) -> None:
        source_uri = case["artifact_uri"] or case["recording_storage_path"]
        if not isinstance(source_uri, str):
            raise ValueError("Training case source has no storage path")
        source = (self._storage_root / source_uri).resolve()
        if self._storage_root not in source.parents or not source.is_file():
            raise FileNotFoundError(f"Training source does not exist: {source_uri}")
        target = audio_root / f"{case['id']}.wav"
        with AsrEvaluationWorker.cropped_audio(source, cast(int, case["start_ms"]), cast(int, case["end_ms"])) as temporary:
            target.write_bytes(temporary.read_bytes())
        language = self._qwen_language(cast(str | None, case["language"]))
        record = {"audio": str(target), "text": f"language {language}<asr_text>{case['reference_text_raw']}"}
        output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    @staticmethod
    def _qwen_language(value: str | None) -> str:
        normalized = (value or "").lower()
        if normalized in {"zh", "zh-cn", "chinese"}:
            return "Chinese"
        if normalized in {"en", "en-us", "english"}:
            return "English"
        return "None"

    def _cancel_requested(self, run_id: UUID) -> bool:
        with self._engine.connect() as connection:
            return bool(connection.execute(text("select cancel_requested from training_runs where id = :run_id"), {"run_id": run_id}).scalar_one())

    def _progress(self, run_id: UUID, percent: int, message: str, *, status: str | None = None) -> None:
        status_sql = "status = :status," if status is not None else ""
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    f"""
                    update training_runs
                    set {status_sql} progress_percent = :percent,
                        progress_message = :message, updated_at = now()
                    where id = :run_id
                    """
                ),
                {"run_id": run_id, "status": status, "percent": percent, "message": message},
            )

    def _cancel(self, run_id: UUID) -> None:
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    update training_runs
                    set status = 'cancelled', progress_message = '训练已取消',
                        finished_at = now(), updated_at = now()
                    where id = :run_id
                    """
                ),
                {"run_id": run_id},
            )

    def _fail(self, run_id: UUID, message: str) -> None:
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    update training_runs
                    set status = 'failed', error_message = :message,
                        progress_message = '训练失败', finished_at = now(), updated_at = now()
                    where id = :run_id
                    """
                ),
                {"run_id": run_id, "message": message[-2000:]},
            )

    @staticmethod
    def _log_tail(path: Path) -> str:
        if not path.is_file():
            return "training log is missing"
        return path.read_text(encoding="utf-8", errors="replace")[-2000:]
