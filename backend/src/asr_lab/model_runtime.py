from __future__ import annotations

import gc
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

from infrastructure.huggingface import resolve_local_snapshot


class AsrRuntime(Protocol):
    def transcribe(self, audio_path: Path, language: str | None) -> str: ...

    def close(self) -> None: ...


class QwenAsrRuntime:
    """Lazy local Qwen ASR runtime with optional PEFT adapter loading."""

    def __init__(
        self,
        *,
        base_model_name: str,
        model_cache_root: Path,
        adapter_path: Path | None,
        context: str = "",
        max_new_tokens: int = 4096,
    ) -> None:
        self._base_model_name = base_model_name
        self._model_cache_root = model_cache_root
        self._adapter_path = adapter_path
        self._context = context
        self._max_new_tokens = max_new_tokens
        self._wrapper: Any | None = None

    def transcribe(self, audio_path: Path, language: str | None) -> str:
        wrapper = self._load()
        result = wrapper.transcribe(
            audio=str(audio_path),
            context=self._context,
            language=self._language(language),
            return_time_stamps=False,
        )
        return self._extract_text(result)

    def close(self) -> None:
        self._wrapper = None
        gc.collect()
        try:
            torch = cast(Any, import_module("torch"))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except (ImportError, AttributeError, RuntimeError):
            pass

    def _load(self) -> Any:
        if self._wrapper is not None:
            return self._wrapper
        try:
            qwen_asr = cast(Any, import_module("qwen_asr"))
            torch = cast(Any, import_module("torch"))
        except ImportError as error:
            raise RuntimeError("Qwen ASR evaluation requires qwen_asr and torch in the worker environment") from error
        base_path = resolve_local_snapshot(self._base_model_name, self._model_cache_root)
        device_map, dtype = self._device_options(torch)
        wrapper = qwen_asr.Qwen3ASRModel.from_pretrained(
            str(base_path),
            dtype=dtype,
            device_map=device_map,
            local_files_only=True,
            max_inference_batch_size=1,
            max_new_tokens=self._max_new_tokens,
        )
        if self._adapter_path is not None:
            try:
                peft = cast(Any, import_module("peft"))
            except ImportError as error:
                raise RuntimeError("Evaluating a LoRA model requires peft in the worker environment") from error
            wrapper.model = peft.PeftModel.from_pretrained(wrapper.model, str(self._adapter_path), is_trainable=False)
        self._wrapper = wrapper
        return wrapper

    @staticmethod
    def _device_options(torch: Any) -> tuple[str, object]:
        if torch.cuda.is_available():
            return "cuda:0", torch.bfloat16
        if torch.backends.mps.is_available():
            return "mps", torch.float16
        return "cpu", torch.float32

    @staticmethod
    def _language(language: str | None) -> str | None:
        normalized = (language or "").strip().lower()
        if not normalized or normalized == "auto":
            return None
        return {"zh": "Chinese", "zh-cn": "Chinese", "chinese": "Chinese", "en": "English", "en-us": "English"}.get(normalized, language)

    @classmethod
    def _extract_text(cls, result: object) -> str:
        items = cast(Sequence[object], result) if isinstance(result, list | tuple) else () if result is None else (result,)
        for item in items:
            value = cast(Mapping[str, object], item).get("text") if isinstance(item, Mapping) else getattr(item, "text", None)
            if isinstance(value, str):
                return value.strip()
        return ""


def build_asr_runtime(model: Mapping[str, object], *, storage_root: Path, model_cache_root: Path) -> AsrRuntime:
    runtime_config = model.get("runtime_config")
    provider = cast(Mapping[str, object], runtime_config).get("provider") if isinstance(runtime_config, Mapping) else None
    if provider not in {None, "qwen_asr"}:
        raise ValueError(f"Unsupported ASR Lab model provider: {provider}")
    adapter_uri = model.get("adapter_uri")
    adapter_path = None
    if isinstance(adapter_uri, str) and adapter_uri:
        candidate = (storage_root / adapter_uri).resolve()
        if storage_root.resolve() not in candidate.parents or not candidate.exists():
            raise FileNotFoundError(f"LoRA adapter does not exist: {adapter_uri}")
        adapter_path = candidate
    return QwenAsrRuntime(
        base_model_name=str(model["base_model_name"]),
        model_cache_root=model_cache_root,
        adapter_path=adapter_path,
    )
