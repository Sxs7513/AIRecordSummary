from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

from l1_foundation.infrastructure.huggingface import resolve_local_snapshot


class TokenEncoder(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


class EmbeddingTokenCounter:
    """Count text with the tokenizer shipped with the configured embedding model."""

    def __init__(self, model_name: str, model_cache_dir: Path) -> None:
        self._model_name = model_name
        self._model_cache_dir = model_cache_dir
        self._tokenizer: TokenEncoder | None = None

    def __call__(self, text: str) -> int:
        return len(self._load_tokenizer().encode(text, add_special_tokens=True))

    def _load_tokenizer(self) -> TokenEncoder:
        if self._tokenizer is not None:
            return self._tokenizer
        model_path = resolve_local_snapshot(self._model_name, self._model_cache_dir)
        try:
            transformers = import_module("transformers")
            auto_tokenizer = transformers.AutoTokenizer
        except (ImportError, AttributeError) as error:
            raise RuntimeError("transformers is not installed; run scripts/install_audio_dependencies.sh") from error
        self._tokenizer = cast(
            TokenEncoder,
            auto_tokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True),
        )
        return self._tokenizer
