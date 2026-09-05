from __future__ import annotations

import gc
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

from l1_foundation.infrastructure.huggingface import resolve_local_snapshot

logger = logging.getLogger("asr")


class TorchCuda(Protocol):
    def is_available(self) -> bool: ...

    def empty_cache(self) -> None: ...


class TorchMps(Protocol):
    def is_available(self) -> bool: ...


class TorchMpsMemory(Protocol):
    def empty_cache(self) -> None: ...


class TorchBackends(Protocol):
    mps: TorchMps


class TorchModule(Protocol):
    cuda: TorchCuda
    mps: TorchMpsMemory
    backends: TorchBackends
    bfloat16: object
    float16: object
    float32: object


class QwenAsrModel(Protocol):
    def transcribe(self, *, audio: str | list[str], context: str, language: str | None, return_time_stamps: bool) -> object: ...


class QwenAsrModelFactory(Protocol):
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs: object) -> QwenAsrModel: ...


class QwenAsrModule(Protocol):
    Qwen3ASRModel: QwenAsrModelFactory


@dataclass(frozen=True, slots=True)
class QwenAsrConfig:
    """Configuration required to execute Qwen ASR, independent of a recording pipeline."""

    model_name: str
    language: str
    model_cache_root: Path
    max_new_tokens: int = 4096
    max_inference_batch_size: int = 1
    num_beams: int = 2
    context: str = ""


@dataclass(frozen=True, slots=True)
class QwenAsrInferenceResult:
    language: str | None
    texts: list[str]


class QwenAsrEngine:
    """Owns one lazily-loaded Qwen ASR model in an inference worker."""

    def __init__(self, config: QwenAsrConfig) -> None:
        self._config = config
        self._model: QwenAsrModel | None = None

    @property
    def display_name(self) -> str:
        return "Qwen ASR"

    @property
    def model_name(self) -> str:
        return self._config.model_name

    def release(self) -> None:
        had_model = self._model is not None
        self._model = None
        gc.collect()
        try:
            torch = cast(TorchModule, import_module("torch"))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except (ImportError, RuntimeError, AttributeError) as error:
            logger.warning("%s：模型已释放，但设备缓存清理失败：%s", self.display_name, error)
        else:
            if had_model:
                logger.info("%s：模型和设备缓存已释放", self.display_name)

    def infer_batch(
        self,
        audio_paths: Sequence[Path],
        progress: Callable[[int, str], None],
        check_cancelled: Callable[[], None] | None = None,
        on_item_completed: Callable[[int, str], None] | None = None,
    ) -> QwenAsrInferenceResult:
        """Run a request batch serially; Qwen's actual inference batch size is one."""
        language = self._language_argument()
        if not audio_paths:
            return QwenAsrInferenceResult(language, [])
        model = self._load_model(progress, 5, 15)
        texts: list[str] = []
        total = len(audio_paths)
        for index, path in enumerate(audio_paths):
            if check_cancelled is not None:
                check_cancelled()
            progress(15 + round(80 * index / total), f"Qwen ASR 推理 {index + 1}/{total}")
            result = model.transcribe(
                audio=str(path),
                context=self._config.context,
                language=language,
                return_time_stamps=False,
            )
            text = self._extract_text(result)
            texts.append(text)
            if on_item_completed is not None:
                on_item_completed(index, text)
        progress(100, f"Qwen ASR 推理 {total}/{total}")
        return QwenAsrInferenceResult(language, texts)

    def _load_model(self, progress: Callable[[int, str], None], progress_start: int = 5, progress_end: int = 15) -> QwenAsrModel:
        if self._model is not None:
            return self._model
        logger.info("Qwen ASR：加载模型 %s", self._config.model_name)
        progress(progress_start, f"加载 Qwen ASR 模型 {self._config.model_name}")
        model_path = resolve_local_snapshot(self._config.model_name, self._config.model_cache_root)
        try:
            torch = cast(TorchModule, import_module("torch"))
            qwen_module = cast(QwenAsrModule, import_module("qwen_asr"))
            factory = qwen_module.Qwen3ASRModel
        except (ImportError, AttributeError) as error:
            raise RuntimeError("Qwen ASR dependencies are missing; start the GPU worker with backend/.venv") from error
        device_map, dtype = self._device_options(torch)
        model = factory.from_pretrained(
            str(model_path),
            dtype=dtype,
            device_map=device_map,
            local_files_only=True,
            max_inference_batch_size=1,
            max_new_tokens=self._config.max_new_tokens,
        )
        self._configure_generation(model)
        self._model = model
        logger.info("Qwen ASR：模型加载完成，运行设备 %s", device_map)
        progress(progress_end, f"Qwen ASR 模型加载完成，运行设备 {device_map}")
        return model

    def _configure_generation(self, model: QwenAsrModel) -> None:
        backend_model = getattr(model, "model", None)
        thinker = getattr(backend_model, "thinker", None)
        generation_config = getattr(thinker, "generation_config", None)
        if generation_config is None:
            raise RuntimeError("Qwen ASR backend does not expose thinker.generation_config; num_beams cannot be applied")
        generation_config.num_beams = self._config.num_beams
        generation_config.do_sample = False
        generation_config.num_return_sequences = 1
        logger.info("Qwen ASR：解码配置 num_beams=%d, do_sample=false, num_return_sequences=1", self._config.num_beams)

    def _language_argument(self) -> str | None:
        normalized = self._config.language.strip().lower()
        if normalized in {"", "auto"}:
            return None
        return {
            "zh": "Chinese",
            "zh-cn": "Chinese",
            "chinese": "Chinese",
            "en": "English",
            "en-us": "English",
        }.get(normalized, self._config.language)

    @staticmethod
    def _device_options(torch: TorchModule) -> tuple[str, object]:
        if torch.cuda.is_available():
            return "cuda:0", torch.bfloat16
        if torch.backends.mps.is_available():
            return "mps", torch.float16
        return "cpu", torch.float32

    @classmethod
    def _extract_text(cls, result: object) -> str:
        for item in cls._result_items(result):
            text = cls._get_attr(item, "text")
            if isinstance(text, str):
                return text.strip()
        return ""

    @staticmethod
    def _result_items(result: object) -> Sequence[object]:
        if isinstance(result, Mapping):
            return [result]
        if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
            return result
        return [result]

    @staticmethod
    def _get_attr(value: object, name: str) -> object:
        if isinstance(value, Mapping):
            return value.get(name)
        return getattr(value, name, None)


__all__ = ("QwenAsrConfig", "QwenAsrEngine", "QwenAsrInferenceResult", "QwenAsrModel")
