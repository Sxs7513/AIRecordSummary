from __future__ import annotations

import gc
from typing import Any, cast

from l1_foundation.settings import Settings
from l1_foundation.worker import WorkerExecutionContext
from l2_core.rag.worker_tasks import RerankInput, RerankResult, RerankScore


class RerankHandler:
    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.rag_rerank_model
        self._cache_dir = str(settings.resolved_rag_rerank_model_cache_dir)
        self._batch_size = settings.rag_rerank_inference_batch_size
        self._model: Any | None = None

    def __call__(self, value: RerankInput, context: WorkerExecutionContext) -> RerankResult:
        model = self._load_model()
        selected: list[tuple[str, str]] = []
        candidate_ids: list[str] = []
        input_tokens = 0
        for candidate in value.candidates:
            context.raise_if_cancelled()
            encoded = model.tokenizer(value.query, candidate.text, add_special_tokens=True, truncation=False)
            pair_tokens = len(cast(list[int], encoded["input_ids"]))
            if input_tokens + pair_tokens > value.max_total_tokens:
                break
            input_tokens += pair_tokens
            selected.append((value.query, candidate.text))
            candidate_ids.append(candidate.candidate_id)
        if not selected:
            return RerankResult(
                model_name=self._model_name,
                scores=[],
                input_tokens=0,
                skipped_candidates=len(value.candidates),
            )
        context.report_progress(0.2, "执行 RAG rerank")
        raw_scores: list[float] = []
        for offset in range(0, len(selected), self._batch_size):
            context.raise_if_cancelled()
            batch = selected[offset : offset + self._batch_size]
            predicted = model.predict(batch, batch_size=self._batch_size, show_progress_bar=False, convert_to_numpy=True)
            raw_scores.extend(float(score) for score in predicted)
            context.report_progress(0.2 + 0.75 * (offset + len(batch)) / len(selected), "执行 RAG rerank")
        scores = [RerankScore(candidate_id=candidate_id, score=float(score)) for candidate_id, score in zip(candidate_ids, raw_scores, strict=True)]
        context.report_progress(1, "RAG rerank 完成")
        return RerankResult(
            model_name=self._model_name,
            scores=scores,
            input_tokens=input_tokens,
            skipped_candidates=len(value.candidates) - len(selected),
        )

    def release(self) -> None:
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            return

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self._model_name,
                cache_folder=self._cache_dir,
                trust_remote_code=True,
                model_kwargs={"low_cpu_mem_usage": True},
            )
        return self._model
