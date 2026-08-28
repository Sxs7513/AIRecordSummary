from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from threading import Event, Thread
from typing import Any, TextIO, cast
from uuid import UUID

from sqlalchemy import Engine, text

from l1_foundation.files import FileStore
from l1_foundation.infrastructure.huggingface import resolve_local_snapshot
from l2_core.asr_lab.evaluators import AsrEvaluationWorker

logger = logging.getLogger("train")


class AsrTrainingWorker:
    """Prepare immutable Qwen data and supervise one isolated LoRA subprocess."""

    def __init__(
        self,
        engine: Engine,
        *,
        file_store: FileStore,
        storage_root: Path,
        model_cache_root: Path,
        training_python: Path,
        training_module: str,
        evaluation_context: str = "",
    ) -> None:
        self._engine = engine
        self._file_store = file_store
        self._storage_root = storage_root.resolve()
        self._model_cache_root = model_cache_root.resolve()
        self._training_python = training_python.absolute()
        self._training_module = training_module
        self._evaluation_context = evaluation_context

    def run_once(self, stop_event: Event | None = None) -> bool:
        if stop_event is not None and stop_event.is_set():
            return False
        run_id = self._claim()
        if run_id is None:
            return False
        logger.info("ASR LoRA：领取训练任务 run_id=%s", run_id)
        try:
            self._execute(run_id, stop_event)
        except Exception as error:
            logger.exception("ASR LoRA：训练任务失败 run_id=%s error=%s", run_id, error)
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

    def _execute(self, run_id: UUID, stop_event: Event | None) -> None:
        logger.info(
            "ASR LoRA：开始准备训练 run_id=%s python=%s module=%s",
            run_id,
            self._training_python,
            self._training_module,
        )
        if stop_event is not None and stop_event.is_set():
            self._requeue(run_id)
            return
        if not self._training_python.is_file():
            raise FileNotFoundError(f"ASR Lab training Python does not exist: {self._training_python}")
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
            config_snapshot = run.get("config_snapshot")
            config = cast(dict[str, object], config_snapshot) if isinstance(config_snapshot, dict) else {}
            run_validation_value = config.get("run_validation", False)
            run_validation = run_validation_value if isinstance(run_validation_value, bool) else False
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
                          and (
                              cases.split = 'train'
                              or (:run_validation and cases.split = 'validation')
                          )
                        order by cases.split, cases.id
                        """
                    ),
                    {
                        "version_id": run["dataset_version_id"],
                        "run_validation": run_validation,
                    },
                ).mappings()
            ]
        train_case_count = sum(case["split"] == "train" for case in cases)
        validation_case_count = len(cases) - train_case_count
        logger.info(
            "ASR LoRA：加载训练数据 run_id=%s train_cases=%d validation_cases=%d run_validation=%s",
            run_id,
            train_case_count,
            validation_case_count,
            run_validation,
        )
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
                if stop_event is not None and stop_event.is_set():
                    self._requeue(run_id)
                    return
                self._materialize_case(case, audio_root, train_output if case["split"] == "train" else validation_output)
                self._progress(run_id, min(20, 2 + round(index / max(1, len(cases)) * 18)), f"准备训练样本 {index}/{len(cases)}")
        logger.info(
            "ASR LoRA：训练数据物化完成 run_id=%s train_file=%s validation_enabled=%s",
            run_id,
            train_file,
            validation_file.stat().st_size > 0,
        )

        if stop_event is not None and stop_event.is_set():
            self._requeue(run_id)
            return
        base_path = resolve_local_snapshot(cast(str, run["base_model_name"]), self._model_cache_root)
        logger.info("ASR LoRA：基础模型已解析 run_id=%s base_model=%s", run_id, base_path)
        manifest: dict[str, object] = {
            "run_id": str(run_id),
            "dataset_version_id": str(run["dataset_version_id"]),
            "base_model": str(base_path),
            "training_method": "lora",
            "train_file": str(train_file),
            "validation_file": str(validation_file) if validation_file.stat().st_size > 0 else None,
            "output_dir": str(output_root),
            "preset": {
                "name": run["preset_name"],
                "rank": 16,
                "alpha": 32,
                "dropout": 0.05,
                "batch_size": 1,
                "gradient_accumulation_steps": 16,
                "learning_rate": 2e-4,
                "epochs": 3,
                "num_workers": 0,
            },
        }
        manifest_path = run_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("ASR LoRA：训练 manifest 已生成 run_id=%s manifest=%s", run_id, manifest_path)
        command = [
            str(self._training_python),
            "-m",
            self._training_module,
            "train",
            "--manifest",
            str(manifest_path),
        ]
        log_path = run_root / "training.log"
        self._progress(run_id, 25, "启动独立 LoRA 训练进程", status="training")
        logger.info(
            "ASR LoRA：启动训练子进程 run_id=%s command=%s log=%s",
            run_id,
            " ".join(command),
            log_path,
        )
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=run_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if process.stdout is None:
                raise RuntimeError("LoRA training subprocess stdout is unavailable")
            output_thread = Thread(
                target=self._relay_training_output,
                args=(run_id, process.stdout, log),
                name=f"asr-training-log-{run_id}",
                daemon=True,
            )
            output_thread.start()
            while process.poll() is None:
                if self._cancel_requested(run_id):
                    self._terminate_process(process)
                    output_thread.join(timeout=5)
                    self._cancel(run_id)
                    return
                if stop_event is not None and stop_event.is_set():
                    self._terminate_process(process)
                    output_thread.join(timeout=5)
                    self._requeue(run_id)
                    return
                if stop_event is None:
                    time.sleep(2)
                else:
                    stop_event.wait(2)
            output_thread.join(timeout=5)
        logger.info("ASR LoRA：训练子进程结束 run_id=%s return_code=%s", run_id, process.returncode)
        if process.returncode != 0:
            tail = self._log_tail(log_path)
            raise RuntimeError(f"LoRA training exited with code {process.returncode}: {tail}")
        if stop_event is not None and stop_event.is_set():
            self._requeue(run_id)
            return
        adapter_config = output_root / "adapter_config.json"
        if not adapter_config.is_file():
            raise RuntimeError("LoRA training completed without adapter_config.json")
        adapter_uri = output_root.relative_to(self._storage_root).as_posix()
        logger.info("ASR LoRA：LoRA adapter 已生成 run_id=%s adapter_uri=%s", run_id, adapter_uri)
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
                        '{"provider":"qwen_hf"}'::jsonb,
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
            validation_case_count = (
                cast(
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
                if run_validation
                else 0
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
                                        "asr_context": self._evaluation_context,
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
                        else (
                            "LoRA 训练完成，候选模型已注册；数据集没有 validation case"
                            if run_validation
                            else "LoRA 训练完成，候选模型已注册；已按任务配置跳过验证"
                        )
                    ),
                },
            )
        logger.info(
            "ASR LoRA：训练任务完成 run_id=%s model_id=%s validation_run_id=%s",
            run_id,
            model_id,
            validation_run_id,
        )

    def _materialize_case(self, case: dict[str, Any], audio_root: Path, output: TextIO) -> None:
        source_uri = case["artifact_uri"] or case["recording_storage_path"]
        if not isinstance(source_uri, str):
            raise ValueError("Training case source has no storage path")
        source = self._file_store.get_file_by_key(source_uri)
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
        logger.info(
            "ASR LoRA：训练进度 run_id=%s percent=%d status=%s message=%s",
            run_id,
            percent,
            status or "unchanged",
            message,
        )
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
        logger.info("ASR LoRA：取消训练任务 run_id=%s", run_id)
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

    def _requeue(self, run_id: UUID) -> None:
        logger.info("ASR LoRA：停止信号触发，重新排队训练任务 run_id=%s", run_id)
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    update training_runs
                    set status = 'queued', progress_message = '服务停止，等待重新调度',
                        finished_at = null, updated_at = now()
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

    def _relay_training_output(self, run_id: UUID, output: TextIO, log: TextIO) -> None:
        for raw_line in output:
            log.write(raw_line)
            log.flush()
            line = raw_line.strip()
            if line and "\r" not in raw_line:
                logger.info("ASR LoRA 子进程：run_id=%s %s", run_id, line)
            progress = self._parse_training_progress(line)
            if progress is None:
                continue
            step, max_steps, train_percent, loss, learning_rate = progress
            overall_percent = min(90, 25 + round(train_percent * 0.65))
            details = [f"训练 step {step}/{max_steps}（{train_percent:.1f}%）"]
            if loss is not None:
                details.append(f"loss={loss:g}")
            if learning_rate is not None:
                details.append(f"lr={learning_rate:g}")
            try:
                self._progress(run_id, overall_percent, " · ".join(details))
            except Exception:
                logger.exception("ASR LoRA：同步实时训练进度失败 run_id=%s", run_id)

    @staticmethod
    def _parse_training_progress(line: str) -> tuple[int, int, float, float | None, float | None] | None:
        marker = "TRAIN_PROGRESS "
        if marker not in line:
            return None
        try:
            payload_value: object = json.loads(line.split(marker, maxsplit=1)[1])
            if not isinstance(payload_value, dict):
                return None
            payload = cast(dict[str, object], payload_value)
            step = int(cast(int | float | str, payload["step"]))
            max_steps = int(cast(int | float | str, payload["max_steps"]))
            percent = float(cast(int | float | str, payload["percent"]))
            loss_value = payload.get("loss")
            learning_rate_value = payload.get("learning_rate")
            loss = float(loss_value) if isinstance(loss_value, int | float) else None
            learning_rate = float(learning_rate_value) if isinstance(learning_rate_value, int | float) else None
            return step, max_steps, percent, loss, learning_rate
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
