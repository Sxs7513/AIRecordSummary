from __future__ import annotations

import gc
import logging
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

from l1_foundation.infrastructure.huggingface import resolve_local_snapshot
from l1_foundation.pipeline.contracts import ArtifactPayload, ResourceQueue, RetryPolicy, StageContext, StageResult
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore
from l2_core.audio_processing.stages.recording_models import EmbeddedSearchChunk, EmbeddingIndexingInput, EmbeddingIndexingOutput, SearchChunksOutput

logger = logging.getLogger(__name__)


class EmbeddingModel(Protocol):
    def encode(self, texts: Sequence[str], **kwargs: object) -> object: ...


class SentenceTransformersModule(Protocol):
    def SentenceTransformer(self, model_name_or_path: str, **kwargs: object) -> EmbeddingModel: ...


class EmbeddingIndexingStage:
    """Create normalized local Qwen embeddings; persistence is handled by the projection service."""

    name = "embedding_indexing"
    version = "1"
    resource_queue = ResourceQueue.GPU_NORMAL
    retry_policy = RetryPolicy(initial_backoff_seconds=30)
    input_model = EmbeddingIndexingInput

    def __init__(self, artifact_store: ArtifactStore, model_name: str, model_cache_dir: Path, dimensions: int, device: str = "auto") -> None:
        self._artifact_store = artifact_store
        self._model_name = model_name
        self._model_cache_dir = model_cache_dir
        self._dimensions = dimensions
        self._device = device
        self._model: EmbeddingModel | None = None

    async def run(self, context: StageContext, input_payload: EmbeddingIndexingInput) -> StageResult[EmbeddingIndexingOutput]:
        try:
            chunks = SearchChunksOutput.model_validate(self._artifact_store.read_json(input_payload.chunks)).chunks
            embeddings = self._embed([chunk.text for chunk in chunks]) if chunks else []
            if any(len(vector) != self._dimensions for vector in embeddings):
                raise ValueError(f"Embedding dimension does not match configured {self._dimensions}")
            output = EmbeddingIndexingOutput(
                provider="sentence_transformers",
                model_name=self._model_name,
                dimensions=self._dimensions,
                chunks=[EmbeddedSearchChunk(**chunk.model_dump(), embedding=vector) for chunk, vector in zip(chunks, embeddings, strict=True)],
            )
            return StageResult(output=output, artifacts=(ArtifactPayload(artifact_type="search.embedding_index", data=output.model_dump(mode="json")),))
        finally:
            self.release()

    def release(self) -> None:
        had_model = self._model is not None
        self._model = None
        gc.collect()
        self._empty_torch_device_caches()
        if had_model:
            logger.info("录音索引：embedding 模型和设备缓存已释放")

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._load_model()
        encoded = model.encode(texts, batch_size=16, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        to_list = getattr(encoded, "tolist", None)
        if not callable(to_list):
            raise ValueError("Embedding provider returned a value without tolist()")
        values = to_list()
        if not isinstance(values, list):
            raise ValueError("Embedding provider returned an invalid embedding matrix")
        untyped_vectors = cast(list[object], values)
        if any(not isinstance(vector, list) for vector in untyped_vectors):
            raise ValueError("Embedding provider returned an invalid embedding matrix")
        vectors = cast(list[list[object]], untyped_vectors)
        return [[float(cast(float | int | str, value)) for value in vector] for vector in vectors]

    def _load_model(self) -> EmbeddingModel:
        if self._model is not None:
            return self._model
        model_path = resolve_local_snapshot(self._model_name, self._model_cache_dir)
        try:
            module = cast(SentenceTransformersModule, import_module("sentence_transformers"))
        except ImportError as error:
            raise RuntimeError("sentence-transformers is not installed; start the GPU worker with backend/.venv") from error
        options: dict[str, object] = {"local_files_only": True, "trust_remote_code": True}
        if self._device != "auto":
            options["device"] = self._device
        self._model = module.SentenceTransformer(str(model_path), **options)
        return self._model

    @staticmethod
    def _empty_torch_device_caches() -> None:
        try:
            torch = cast(Any, import_module("torch"))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except (ImportError, RuntimeError, AttributeError) as error:
            logger.warning("录音索引：模型已释放，但设备缓存清理失败：%s", error)
