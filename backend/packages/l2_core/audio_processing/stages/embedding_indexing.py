from __future__ import annotations

import gc
import logging
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

from l1_foundation.infrastructure.huggingface import resolve_local_snapshot
from l1_foundation.pipeline.contracts import ArtifactPayload, RetryPolicy, StageContext, StageResult
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore
from l1_foundation.worker import WorkerClient
from l2_core.audio_processing.stages.recording_models import EmbeddedSearchChunk, EmbeddingIndexingInput, EmbeddingIndexingOutput, SearchChunksOutput
from l2_core.audio_processing.worker_tasks import EmbeddingEncodeTaskResult, embedding_encode_command

logger = logging.getLogger("audio_processing")


class EmbeddingModel(Protocol):
    def encode(self, texts: Sequence[str], **kwargs: object) -> object: ...


class SentenceTransformersModule(Protocol):
    def SentenceTransformer(self, model_name_or_path: str, **kwargs: object) -> EmbeddingModel: ...


class EmbeddingIndexingStage:
    """Create normalized local Qwen embeddings; persistence is handled by the projection service."""

    name = "embedding_indexing"
    # Version 8 invalidates embedding artifacts created before raw ASR text
    # and polished text formatting were restored from speaker-level alignment.
    # Embeddings still use only
    # retrieval_text; invalidation is required because this artifact is also
    # the payload used to project chunk metadata into PostgreSQL.
    version = "8"
    retry_policy = RetryPolicy(initial_backoff_seconds=30)
    input_model = EmbeddingIndexingInput

    def __init__(
        self,
        artifact_store: ArtifactStore,
        model_name: str,
        model_cache_dir: Path,
        dimensions: int,
        device: str = "auto",
        worker_client: WorkerClient | None = None,
    ) -> None:
        self._artifact_store = artifact_store
        self._model_name = model_name
        self._model_cache_dir = model_cache_dir
        self._dimensions = dimensions
        self._device = device
        self._model: EmbeddingModel | None = None
        self._worker_client = worker_client

    async def try_restore(self, context: StageContext, _input_payload: EmbeddingIndexingInput) -> StageResult[EmbeddingIndexingOutput] | None:
        return self._artifact_store.try_restore_json(
            context.pipeline_run_id,
            context.stage_run_id,
            self.name,
            self.version,
            "search.embedding_index",
            EmbeddingIndexingOutput,
            input_fingerprint=context.input_fingerprint,
            allow_legacy_restore=context.allow_legacy_restore,
        )

    async def run(self, context: StageContext, input_payload: EmbeddingIndexingInput) -> StageResult[EmbeddingIndexingOutput]:
        chunks = SearchChunksOutput.model_validate(self._artifact_store.read_json(input_payload.chunks)).chunks
        if self._worker_client is None:
            raise RuntimeError("EmbeddingIndexingStage requires WorkerClient")
        if chunks:
            result = await self._worker_client.execute(
                embedding_encode_command([chunk.retrieval_text() for chunk in chunks]),
                result_type=EmbeddingEncodeTaskResult,
                on_progress=lambda progress, message: context.report_progress(round(progress * 100), message or "Embedding 编码"),
            )
            embeddings = result.vectors
        else:
            embeddings = []
        if any(len(vector) != self._dimensions for vector in embeddings):
            raise ValueError(f"Embedding dimension does not match configured {self._dimensions}")
        output = EmbeddingIndexingOutput(
            provider="sentence_transformers",
            model_name=self._model_name,
            dimensions=self._dimensions,
            chunks=[EmbeddedSearchChunk(**chunk.model_dump(), embedding=vector) for chunk, vector in zip(chunks, embeddings, strict=True)],
        )
        return StageResult(output=output, artifacts=(ArtifactPayload(artifact_type="search.embedding_index", data=output.model_dump(mode="json")),))

    def release(self) -> None:
        had_model = self._model is not None
        self._model = None
        gc.collect()
        self._empty_torch_device_caches()
        if had_model:
            logger.info("录音索引：embedding 模型和设备缓存已释放")

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts)

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
        device = self._resolve_device()
        options["device"] = device
        logger.info("录音索引：加载 embedding 模型 %s，推理设备=%s", self._model_name, device)
        self._model = module.SentenceTransformer(str(model_path), **options)
        return self._model

    def _resolve_device(self) -> str:
        """Prefer an available accelerator while keeping an explicit override available."""

        if self._device != "auto":
            return self._device
        try:
            torch = cast(Any, import_module("torch"))
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except (ImportError, RuntimeError, AttributeError) as error:
            logger.warning("录音索引：无法检测 GPU，回退 CPU：%s", error)
        return "cpu"

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
