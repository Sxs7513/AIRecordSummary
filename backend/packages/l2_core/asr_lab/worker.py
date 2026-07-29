from __future__ import annotations

import logging
from threading import Event

from sqlalchemy import Engine

from l1_foundation.settings import Settings
from l2_core.asr_lab.evaluators import AsrEvaluationWorker
from l2_core.asr_lab.training import AsrTrainingWorker
from l2_core.audio_processing.stages.transcribe_qwen_asr.context import build_qwen_asr_context

logger = logging.getLogger("train")


class AsrLabWorker:
    """Single-GPU ASR Lab worker that serializes evaluation and training work."""

    def __init__(self, engine: Engine, settings: Settings) -> None:
        evaluation_context = build_qwen_asr_context(
            settings.resolved_qwen_asr_context_config,
            settings.qwen_asr_max_context_items,
            settings.qwen_asr_context,
        )
        self._evaluation = AsrEvaluationWorker(
            engine,
            settings.resolved_local_storage_root,
            settings.resolved_huggingface_hub_cache_dir,
            hf_runtime_python=settings.resolved_asr_lab_training_python_bin,
            hf_runtime_module=settings.asr_lab_training_module,
            context=evaluation_context,
        )
        self._training = AsrTrainingWorker(
            engine,
            storage_root=settings.resolved_local_storage_root,
            model_cache_root=settings.resolved_huggingface_hub_cache_dir,
            training_python=settings.resolved_asr_lab_training_python_bin,
            training_module=settings.asr_lab_training_module,
            evaluation_context=evaluation_context,
        )
        self._poll_seconds = settings.asr_lab_worker_poll_seconds
        self._stop_event = Event()

    def run_once(self) -> bool:
        """Run at most one task, preferring short evaluation work over training."""
        if self._stop_event.is_set():
            return False
        if self._evaluation.run_once(self._stop_event):
            return True
        if self._stop_event.is_set():
            return False
        return self._training.run_once(self._stop_event)

    def run_forever(self) -> None:
        logger.info("ASR Lab worker started")
        while not self._stop_event.is_set():
            if not self.run_once():
                self._stop_event.wait(self._poll_seconds)
        logger.info("ASR Lab worker stopped")

    def stop(self) -> None:
        """Request shutdown and wake an idle worker immediately."""
        self._stop_event.set()
