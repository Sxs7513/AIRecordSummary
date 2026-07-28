from __future__ import annotations

import json
import math
import subprocess
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from sqlalchemy import Connection, Engine, text

from asr_lab.model_runtime import build_asr_runtime
from asr_lab.normalization import normalize_text
from evaluation.metrics import character_error_rate, word_error_rate


class AsrEvaluationWorker:
    """Claim and execute queued ASR evaluation runs in an offline worker."""

    def __init__(self, engine: Engine, storage_root: Path, model_cache_root: Path) -> None:
        self._engine = engine
        self._storage_root = storage_root.resolve()
        self._model_cache_root = model_cache_root.resolve()

    def run_once(self) -> bool:
        run_id = self._claim()
        if run_id is None:
            return False
        try:
            self._execute(run_id)
        except Exception as error:
            with self._engine.begin() as connection:
                connection.execute(
                    text(
                        """
                        update evaluation_runs
                        set status = 'failed', error_message = :error_message,
                            finished_at = now(), updated_at = now()
                        where id = :run_id
                        """
                    ),
                    {"run_id": run_id, "error_message": str(error)[-2000:]},
                )
        return True

    def _claim(self) -> UUID | None:
        with self._engine.begin() as connection:
            value = connection.execute(
                text(
                    """
                    with candidate as (
                        select id
                        from evaluation_runs
                        where status = 'queued'
                        order by created_at
                        for update skip locked
                        limit 1
                    )
                    update evaluation_runs runs
                    set status = 'running', started_at = coalesce(started_at, now()),
                        error_message = null, updated_at = now()
                    from candidate
                    where runs.id = candidate.id
                    returning runs.id
                    """
                )
            ).scalar_one_or_none()
        return None if value is None else UUID(str(value))

    def _execute(self, run_id: UUID) -> None:
        with self._engine.connect() as connection:
            run = dict(connection.execute(text("select * from evaluation_runs where id = :run_id"), {"run_id": run_id}).mappings().one())
            models = [
                dict(row)
                for row in connection.execute(
                    text(
                        """
                        select models.*
                        from evaluation_run_models run_models
                        join model_versions models on models.id = run_models.model_version_id
                        where run_models.evaluation_run_id = :run_id
                        order by run_models.position
                        """
                    ),
                    {"run_id": run_id},
                ).mappings()
            ]
            cases = [
                dict(row)
                for row in connection.execute(
                    text(
                        """
                        select cases.*, assets.artifact_uri, assets.recording_id,
                               assets.checksum as source_checksum,
                               recordings.storage_path as recording_storage_path
                        from evaluation_cases cases
                        join evaluation_source_assets assets on assets.id = cases.source_asset_id
                        left join recordings on recordings.id = assets.recording_id
                        where cases.dataset_version_id = :version_id
                          and cases.split = :split
                        order by cases.id
                        """
                    ),
                    {"version_id": run["dataset_version_id"], "split": run["split"]},
                ).mappings()
            ]
        config = cast(dict[str, Any], run["config_snapshot"])
        normalization_name = str(config.get("normalization_name", "zh_asr"))
        normalization_version = str(config.get("normalization_version", "v1"))

        for model in models:
            if self._cancel_requested(run_id):
                self._finish_cancelled(run_id)
                return
            runtime = build_asr_runtime(model, storage_root=self._storage_root, model_cache_root=self._model_cache_root)
            try:
                for case in cases:
                    if self._result_exists(run_id, UUID(str(model["id"])), UUID(str(case["id"]))):
                        continue
                    if self._cancel_requested(run_id):
                        self._finish_cancelled(run_id)
                        return
                    self._evaluate_case(
                        run_id,
                        model_id=UUID(str(model["id"])),
                        case=case,
                        runtime=runtime,
                        normalization_name=normalization_name,
                        normalization_version=normalization_version,
                    )
            finally:
                runtime.close()
            self._aggregate_model(run_id, UUID(str(model["id"])))

        with self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    update evaluation_runs
                    set status = 'succeeded', finished_at = now(), updated_at = now()
                    where id = :run_id and status = 'running'
                    """
                ),
                {"run_id": run_id},
            )
            source_training_run_id = config.get("source_training_run_id")
            if isinstance(source_training_run_id, str):
                connection.execute(
                    text(
                        """
                        update model_versions models
                        set status = 'validated', updated_at = now()
                        from evaluation_run_models run_models
                        where run_models.evaluation_run_id = :run_id
                          and run_models.model_version_id = models.id
                          and run_models.role = 'candidate'
                          and models.status = 'candidate'
                          and exists (
                              select 1 from evaluation_runs
                              where id = :run_id and failed_case_count = 0
                          )
                        """
                    ),
                    {"run_id": run_id},
                )

    def _evaluate_case(
        self,
        run_id: UUID,
        *,
        model_id: UUID,
        case: dict[str, Any],
        runtime: Any,
        normalization_name: str,
        normalization_version: str,
    ) -> None:
        case_id = UUID(str(case["id"]))
        source_path = self._source_path(case)
        started = time.perf_counter()
        try:
            with self.cropped_audio(source_path, cast(int, case["start_ms"]), cast(int, case["end_ms"])) as sample_path:
                hypothesis_raw = cast(str, runtime.transcribe(sample_path, cast(str | None, case["language"])))
            duration_ms = round((time.perf_counter() - started) * 1000)
            hypothesis_normalized = normalize_text(hypothesis_raw, normalization_name, normalization_version)
            reference_normalized = cast(str, case["reference_text_normalized"])
            cer = character_error_rate(reference_normalized, hypothesis_normalized)
            wer = word_error_rate(reference_normalized, hypothesis_normalized)
            details = {
                "cer": {**asdict(cer), "operations": [asdict(item) for item in cer.operations]},
                "wer": {**asdict(wer), "operations": [asdict(item) for item in wer.operations]},
            }
            with self._engine.begin() as connection:
                connection.execute(
                    text(
                        """
                        insert into evaluation_case_results (
                            evaluation_run_id, model_version_id, evaluation_case_id,
                            hypothesis_text_raw, hypothesis_text_normalized,
                            inference_duration_ms, status, details
                        )
                        values (
                            :run_id, :model_id, :case_id,
                            :raw, :normalized, :duration_ms, 'succeeded', cast(:details as jsonb)
                        )
                        on conflict (evaluation_run_id, model_version_id, evaluation_case_id) do nothing
                        """
                    ),
                    {
                        "run_id": run_id,
                        "model_id": model_id,
                        "case_id": case_id,
                        "raw": hypothesis_raw,
                        "normalized": hypothesis_normalized,
                        "duration_ms": duration_ms,
                        "details": json.dumps(details, ensure_ascii=False, separators=(",", ":")),
                    },
                )
                self._insert_case_metric(connection, run_id, model_id, case_id, "cer", cer.value, cer.reference_units, details["cer"])
                self._insert_case_metric(connection, run_id, model_id, case_id, "wer", wer.value, wer.reference_units, details["wer"])
                self._increment_progress(connection, run_id, failed=False)
        except Exception as error:
            with self._engine.begin() as connection:
                connection.execute(
                    text(
                        """
                        insert into evaluation_case_results (
                            evaluation_run_id, model_version_id, evaluation_case_id,
                            status, error_message
                        )
                        values (:run_id, :model_id, :case_id, 'failed', :error_message)
                        on conflict (evaluation_run_id, model_version_id, evaluation_case_id) do nothing
                        """
                    ),
                    {"run_id": run_id, "model_id": model_id, "case_id": case_id, "error_message": str(error)[-2000:]},
                )
                self._increment_progress(connection, run_id, failed=True)

    @staticmethod
    def _insert_case_metric(
        connection: Connection,
        run_id: UUID,
        model_id: UUID,
        case_id: UUID,
        name: str,
        value: float,
        sample_count: int,
        details: object,
    ) -> None:
        connection.execute(
            text(
                """
                insert into evaluation_metric_values (
                    evaluation_run_id, model_version_id, evaluation_case_id,
                    metric_name, metric_version, value, sample_count, details
                )
                values (
                    :run_id, :model_id, :case_id,
                    :metric_name, 'v1', :value, :sample_count, cast(:details as jsonb)
                )
                on conflict do nothing
                """
            ),
            {
                "run_id": run_id,
                "model_id": model_id,
                "case_id": case_id,
                "metric_name": name,
                "value": value,
                "sample_count": sample_count,
                "details": json.dumps(details, ensure_ascii=False, separators=(",", ":")),
            },
        )

    def _aggregate_model(self, run_id: UUID, model_id: UUID) -> None:
        with self._engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    select details, hypothesis_text_normalized, inference_duration_ms
                    from evaluation_case_results
                    where evaluation_run_id = :run_id
                      and model_version_id = :model_id
                      and status = 'succeeded'
                    """
                ),
                {"run_id": run_id, "model_id": model_id},
            ).mappings().all()
            if not rows:
                return
            durations = sorted(cast(int, row["inference_duration_ms"]) for row in rows if row["inference_duration_ms"] is not None)
            for metric_name in ("cer", "wer"):
                metric_details = [cast(dict[str, Any], row["details"])[metric_name] for row in rows]
                errors = sum(int(item["substitutions"]) + int(item["deletions"]) + int(item["insertions"]) for item in metric_details)
                units = sum(int(item["reference_units"]) for item in metric_details)
                self._upsert_model_metric(connection, run_id, model_id, metric_name, errors / units if units else float(errors > 0), units)
            blank_rate = sum(not cast(str, row["hypothesis_text_normalized"]) for row in rows) / len(rows)
            self._upsert_model_metric(connection, run_id, model_id, "blank_output_rate", blank_rate, len(rows))
            if durations:
                average = sum(durations) / len(durations)
                p95 = durations[min(len(durations) - 1, max(0, math.ceil(len(durations) * 0.95) - 1))]
                self._upsert_model_metric(connection, run_id, model_id, "average_inference_duration_ms", average, len(durations))
                self._upsert_model_metric(connection, run_id, model_id, "p95_inference_duration_ms", p95, len(durations))

    @staticmethod
    def _upsert_model_metric(connection: Connection, run_id: UUID, model_id: UUID, name: str, value: float, count: int) -> None:
        connection.execute(
            text(
                """
                insert into evaluation_metric_values (
                    evaluation_run_id, model_version_id, metric_name,
                    metric_version, value, sample_count
                )
                values (:run_id, :model_id, :name, 'v1', :value, :count)
                on conflict (evaluation_run_id, model_version_id, metric_name, metric_version)
                    where model_version_id is not null and evaluation_case_id is null
                do update set value = excluded.value, sample_count = excluded.sample_count,
                              details = excluded.details, created_at = now()
                """
            ),
            {"run_id": run_id, "model_id": model_id, "name": name, "value": value, "count": count},
        )

    @staticmethod
    def _increment_progress(connection: Connection, run_id: UUID, *, failed: bool) -> None:
        connection.execute(
            text(
                f"""
                update evaluation_runs
                set {"failed_case_count" if failed else "completed_case_count"} =
                        {"failed_case_count" if failed else "completed_case_count"} + 1,
                    updated_at = now()
                where id = :run_id
                """
            ),
            {"run_id": run_id},
        )

    def _result_exists(self, run_id: UUID, model_id: UUID, case_id: UUID) -> bool:
        with self._engine.connect() as connection:
            return (
                connection.execute(
                    text(
                        """
                        select 1 from evaluation_case_results
                        where evaluation_run_id = :run_id
                          and model_version_id = :model_id
                          and evaluation_case_id = :case_id
                        """
                    ),
                    {"run_id": run_id, "model_id": model_id, "case_id": case_id},
                ).scalar_one_or_none()
                is not None
            )

    def _cancel_requested(self, run_id: UUID) -> bool:
        with self._engine.connect() as connection:
            return bool(connection.execute(text("select cancel_requested from evaluation_runs where id = :run_id"), {"run_id": run_id}).scalar_one())

    def _finish_cancelled(self, run_id: UUID) -> None:
        with self._engine.begin() as connection:
            connection.execute(
                text("update evaluation_runs set status = 'cancelled', finished_at = now(), updated_at = now() where id = :run_id"),
                {"run_id": run_id},
            )

    def _source_path(self, case: dict[str, Any]) -> Path:
        uri = case["artifact_uri"] or case["recording_storage_path"]
        if not isinstance(uri, str):
            raise ValueError("Evaluation source asset has no storage path")
        path = (self._storage_root / uri).resolve()
        if self._storage_root not in path.parents or not path.is_file():
            raise FileNotFoundError(f"Evaluation source audio does not exist: {uri}")
        return path

    @staticmethod
    @contextmanager
    def cropped_audio(source: Path, start_ms: int, end_ms: int) -> Generator[Path]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
            target = Path(temporary.name)
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(source), "-ss", str(start_ms / 1000), "-to", str(end_ms / 1000),
                    "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(target),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg case crop failed: {result.stderr.decode(errors='replace')[-1000:]}")
            yield target
        finally:
            target.unlink(missing_ok=True)
