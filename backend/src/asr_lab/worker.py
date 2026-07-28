from __future__ import annotations

import time

from sqlalchemy import Engine

from asr_lab.evaluators import AsrEvaluationWorker
from asr_lab.training import AsrTrainingWorker
from settings import Settings


class AsrLabWorker:
    """Single-GPU ASR Lab worker that serializes evaluation and training work."""

    def __init__(self, engine: Engine, settings: Settings) -> None:
        self._evaluation = AsrEvaluationWorker(
            engine,
            settings.resolved_local_storage_root,
            settings.resolved_huggingface_hub_cache_dir,
        )
        self._training = AsrTrainingWorker(
            engine,
            storage_root=settings.resolved_local_storage_root,
            model_cache_root=settings.resolved_huggingface_hub_cache_dir,
            training_python=settings.resolved_asr_lab_training_python_bin,
            training_script=settings.resolved_asr_lab_training_script,
        )
        self._poll_seconds = settings.asr_lab_worker_poll_seconds

    def run_once(self) -> bool:
        """Run at most one task, preferring short evaluation work over training."""
        if self._evaluation.run_once():
            return True
        return self._training.run_once()

    def run_forever(self) -> None:
        while True:
            if not self.run_once():
                time.sleep(self._poll_seconds)

