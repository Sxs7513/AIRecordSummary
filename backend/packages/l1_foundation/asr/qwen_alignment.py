from __future__ import annotations

import gc
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

from l1_foundation.asr.qwen import TorchModule
from l1_foundation.infrastructure.huggingface import resolve_local_snapshot

logger = logging.getLogger("asr")


class ForcedAlignToken(Protocol):
    text: str
    start_time: float
    end_time: float


class QwenForcedAligner(Protocol):
    def align(self, *, audio: str, text: str, language: str) -> object: ...


class QwenForcedAlignerFactory(Protocol):
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs: object) -> QwenForcedAligner: ...


class QwenAsrModule(Protocol):
    Qwen3ForcedAligner: QwenForcedAlignerFactory


@dataclass(frozen=True, slots=True)
class QwenForcedAlignmentConfig:
    model_name: str
    model_cache_root: Path


@dataclass(frozen=True, slots=True)
class QwenForcedAlignmentRequest:
    item_id: str
    audio_path: Path
    text: str
    language: str


@dataclass(frozen=True, slots=True)
class QwenForcedAlignmentToken:
    text: str
    start_time: float
    end_time: float


@dataclass(frozen=True, slots=True)
class QwenForcedAlignmentResult:
    item_id: str
    tokens: list[QwenForcedAlignmentToken]


class QwenForcedAlignmentEngine:
    """Owns one lazily-loaded Qwen ForcedAligner in an inference worker."""

    def __init__(self, config: QwenForcedAlignmentConfig) -> None:
        self._config = config
        self._aligner: QwenForcedAligner | None = None

    @property
    def model_name(self) -> str:
        return self._config.model_name

    def release(self) -> None:
        self._aligner = None
        gc.collect()
        try:
            torch = cast(TorchModule, import_module("torch"))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except (ImportError, RuntimeError, AttributeError):
            pass

    def infer_batch(
        self,
        items: Sequence[QwenForcedAlignmentRequest],
        progress: Callable[[int, str], None],
        check_cancelled: Callable[[], None] | None = None,
    ) -> list[QwenForcedAlignmentResult]:
        if not items:
            return []
        aligner = self._load(progress)
        results: list[QwenForcedAlignmentResult] = []
        total = len(items)
        for index, item in enumerate(items):
            if check_cancelled is not None:
                check_cancelled()
            progress(15 + round(80 * index / total), f"ForcedAligner 推理 {index + 1}/{total}")
            aligned = cast(list[list[ForcedAlignToken]], aligner.align(audio=str(item.audio_path), text=item.text, language=item.language))
            if len(aligned) != 1:
                raise RuntimeError(f"ForcedAligner result count mismatch for item {item.item_id}")
            results.append(
                QwenForcedAlignmentResult(
                    item_id=item.item_id,
                    tokens=[QwenForcedAlignmentToken(token.text, token.start_time, token.end_time) for token in aligned[0]],
                )
            )
        progress(100, f"ForcedAligner 推理 {total}/{total}")
        return results

    def _load(self, progress: Callable[[int, str], None]) -> QwenForcedAligner:
        if self._aligner is not None:
            return self._aligner
        progress(5, f"加载 ForcedAligner {self._config.model_name}")
        try:
            torch = cast(TorchModule, import_module("torch"))
            qwen_asr = cast(QwenAsrModule, import_module("qwen_asr"))
        except (ImportError, AttributeError) as error:
            raise RuntimeError("Qwen ForcedAligner dependencies are missing; start the GPU worker with backend/.venv") from error
        device_map, dtype = self._device_options(torch)
        model_path = resolve_local_snapshot(self._config.model_name, self._config.model_cache_root)
        self._aligner = qwen_asr.Qwen3ForcedAligner.from_pretrained(
            str(model_path), dtype=dtype, device_map=device_map, local_files_only=True
        )
        progress(15, "ForcedAligner 加载完成")
        return self._aligner

    @staticmethod
    def _device_options(torch: TorchModule) -> tuple[str, object]:
        if torch.cuda.is_available():
            return "cuda:0", torch.bfloat16
        if torch.backends.mps.is_available():
            return "mps", torch.float16
        return "cpu", torch.float32


__all__ = (
    "QwenForcedAlignmentConfig",
    "QwenForcedAlignmentEngine",
    "QwenForcedAlignmentRequest",
    "QwenForcedAlignmentResult",
    "QwenForcedAlignmentToken",
)
