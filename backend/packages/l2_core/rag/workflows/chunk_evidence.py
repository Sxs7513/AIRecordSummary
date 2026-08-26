from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any, Literal, cast

from langgraph.graph import END, START, StateGraph

from l1_foundation.observability import finish_invocation, finish_span, start_invocation
from l1_foundation.observability.context import current_span
from l2_core.rag.contracts import Evidence, RagGraphState, RagStateUpdate, ResolvedFilters, RetrievalCandidateRow
from l2_core.rag.execution_middleware import rag_execution_middleware
from l2_core.rag.observability import elapsed_ms, log_event, started_at
from l2_core.rag.retrieval import RagRetriever

logger = logging.getLogger("rag")

NodeStarted = Callable[[RagGraphState, str], float]
NodeCompleted = Callable[..., None]
OperationCompleted = Callable[..., None]
Transition = Callable[[RagGraphState, str, str, str], None]


class ChunkEvidencePipeline:
    """Reusable chunk retrieval, context expansion and rerank workflow."""

    def __init__(
        self,
        retriever: RagRetriever,
        *,
        node_started: NodeStarted,
        node_completed: NodeCompleted,
        operation_completed: OperationCompleted,
        transition: Transition,
    ) -> None:
        self._retriever = retriever
        self._node_started = node_started
        self._node_completed = node_completed
        self._operation_completed = operation_completed
        self._transition = transition
        builder = cast(Any, StateGraph(RagGraphState))
        builder.add_node(
            "retrieve",
            rag_execution_middleware.wrap_node(self.retrieve, graph_name="chunk_evidence", node_name="retrieve"),
        )
        builder.add_node(
            "expand_context",
            rag_execution_middleware.wrap_node(self.expand_context, graph_name="chunk_evidence", node_name="expand_context"),
        )
        builder.add_node(
            "rerank",
            rag_execution_middleware.wrap_node(self.rerank, graph_name="chunk_evidence", node_name="rerank"),
        )
        builder.add_edge(START, "retrieve")
        builder.add_conditional_edges(
            "retrieve",
            self.after_retrieve,
            {"expand_context": "expand_context", "done": END},
        )
        builder.add_conditional_edges(
            "expand_context",
            self.after_expand_context,
            {"rerank": "rerank", "done": END},
        )
        builder.add_edge("rerank", END)
        self._graph: Any = builder.compile()

    async def invoke(self, state: RagGraphState) -> RagGraphState:
        return cast(RagGraphState, await self._graph.ainvoke(state))

    async def retrieve(self, state: RagGraphState) -> RagStateUpdate:
        node_started = self._node_started(state, "retrieve")
        filters = state["filters"]
        if filters is None:
            raise RuntimeError("Chunk evidence workflow requires resolved filters")
        if filters.match_none:
            self._node_completed(
                state,
                "retrieve",
                node_started,
                outcome="empty",
                reason="match_none",
                strategy="fact_lookup",
                evidence_count=0,
            )
            return {
                "retrieval_candidates": [],
                "protected_chunk_ids": [],
                "evidence": [],
                "answer_evidence": [],
                "message": "没有找到符合范围的已完成录音",
            }
        query = state["content_query"]
        candidates = await self.retrieve_candidates(
            query,
            filters,
            state["limit"],
            state.get("run_id", "standalone"),
            expanded_query=state.get("retrieval_expanded_query"),
            lexical_queries=state.get("retrieval_lexical_queries", []),
            protected_lexical_queries=state.get("retrieval_protected_lexical_queries", []),
        )
        self._node_completed(
            state,
            "retrieve",
            node_started,
            outcome="succeeded" if candidates else "empty",
            strategy="fact_lookup",
            evidence_count=0,
            candidate_count=len(candidates),
            recording_count=0,
            requested_limit=state["limit"],
        )
        return {
            "retrieval_candidates": candidates,
            "protected_chunk_ids": [
                str(candidate["chunk_id"])
                for candidate in candidates
                if candidate.get("protected_lexical_terms") or candidate.get("retrieved_via_recording_profile")
            ],
            "evidence": [],
            "answer_evidence": [],
            "rerank_input_tokens": 0,
            "rerank_skipped_candidates": 0,
            "message": None if candidates else "没有找到足够相关的录音片段",
        }

    def after_retrieve(self, state: RagGraphState) -> Literal["expand_context", "done"]:
        target: Literal["expand_context", "done"] = "expand_context" if state["retrieval_candidates"] else "done"
        self._transition(
            state,
            "retrieve",
            target,
            "candidates_ready" if target == "expand_context" else "no_candidates",
        )
        return target

    async def retrieve_candidates(
        self,
        query: str,
        filters: ResolvedFilters,
        limit: int,
        run_id: str,
        *,
        expanded_query: str | None = None,
        lexical_queries: list[str] | None = None,
        protected_lexical_queries: list[str] | None = None,
    ) -> list[RetrievalCandidateRow]:
        if not self._retriever.hybrid_search_enabled:
            vector_started = started_at()
            rows = await asyncio.to_thread(self._retriever.retrieve_candidates, query, filters, limit)
            self._operation_completed("retrieve", "retrieve.vector", rows, vector_started)
            return rows

        started = started_at()
        vector_queries: list[str] = list(dict.fromkeys([query, *([expanded_query] if expanded_query is not None else [])]))
        lexical_searches: list[str] = list(dict.fromkeys(lexical_queries or []))
        protected_searches = set(protected_lexical_queries or [])
        log_event(
            "retrieval_queries_selected",
            run_id,
            vector_queries=vector_queries,
            lexical_queries=lexical_searches,
            protected_lexical_queries=sorted(protected_searches),
        )

        async def vector_search(
            embedding: list[float],
            variant: Literal["original", "expanded"],
            variant_query: str,
        ) -> list[RetrievalCandidateRow]:
            operation_started = started_at()
            operation = f"retrieve.vector.{variant}"
            try:
                rows = await asyncio.to_thread(self._retriever.retrieve_vector_candidates, embedding, filters)
            except Exception as error:
                self._operation_completed(
                    "retrieve",
                    operation,
                    [],
                    operation_started,
                    status="failed",
                    details={"query_variant": variant, "query": variant_query, "error_type": type(error).__name__},
                )
                raise
            self._operation_completed(
                "retrieve",
                operation,
                rows,
                operation_started,
                details={"query_variant": variant, "query": variant_query},
            )
            return rows

        async def lexical_search(variant_query: str) -> list[RetrievalCandidateRow]:
            operation_started = started_at()
            try:
                rows = await asyncio.to_thread(self._retriever.retrieve_lexical_candidates, variant_query, filters)
            except Exception as error:
                self._operation_completed(
                    "retrieve",
                    "retrieve.lexical.term",
                    [],
                    operation_started,
                    status="failed",
                    details={"query_variant": "term", "query": variant_query, "error_type": type(error).__name__},
                )
                raise
            self._operation_completed(
                "retrieve",
                "retrieve.lexical.term",
                rows,
                operation_started,
                details={"query_variant": "term", "query": variant_query},
            )
            return rows

        async def semantic_searches() -> tuple[list[list[RetrievalCandidateRow] | BaseException], list[RetrievalCandidateRow], BaseException | None]:
            try:
                embeddings = await asyncio.to_thread(self._retriever.generate_query_embeddings, vector_queries)
            except Exception as error:
                return [error for _ in vector_queries], [], error

            async def recording_profile_search() -> list[RetrievalCandidateRow]:
                if not bool(getattr(self._retriever, "recording_profile_search_enabled", False)):
                    return []
                profile_candidates = await asyncio.to_thread(
                    self._retriever.retrieve_recording_profile_candidates,
                    embeddings[0],
                    filters,
                )
                log_event(
                    "recording_profile_retrieval_completed",
                    run_id,
                    query=query,
                    match_count=len(profile_candidates),
                    matches=[
                        {
                            "recording_id": str(candidate["recording_id"]),
                            "score": round(candidate["score"], 6),
                        }
                        for candidate in profile_candidates
                    ],
                )
                scoped_rows = await asyncio.to_thread(
                    self._retriever.retrieve_recording_profile_scoped_chunk_candidates,
                    embeddings[0],
                    profile_candidates,
                )
                log_event(
                    "recording_profile_scoped_chunk_retrieval_completed",
                    run_id,
                    recording_match_count=len(profile_candidates),
                    chunk_count=len(scoped_rows),
                    candidates=_candidate_refs(scoped_rows),
                )
                return scoped_rows

            raw_vector_results, raw_profile_results = await asyncio.gather(
                asyncio.gather(
                    *(
                        vector_search(
                            embedding,
                            "original" if index == 0 else "expanded",
                            vector_queries[index],
                        )
                        for index, embedding in enumerate(embeddings)
                    ),
                    return_exceptions=True,
                ),
                recording_profile_search(),
                return_exceptions=True,
            )
            vector_results = cast(list[list[RetrievalCandidateRow] | BaseException], raw_vector_results)
            if isinstance(raw_profile_results, BaseException):
                return vector_results, [], raw_profile_results
            return vector_results, raw_profile_results, None

        semantic_result, lexical_results = await asyncio.gather(
            semantic_searches(),
            asyncio.gather(
                *(lexical_search(item) for item in lexical_searches),
                return_exceptions=True,
            ),
        )
        vector_results, profile_rows, profile_error = semantic_result
        if profile_error is not None:
            log_event(
                "recording_profile_retrieval_failed",
                run_id,
                level=logging.WARNING,
                error_type=type(profile_error).__name__,
            )
        # Preserve the original/expanded vector-list positions so a failed
        # original query cannot accidentally promote the expanded query to its weight.
        vector_lists = [item if isinstance(item, list) else [] for item in vector_results]
        lexical_lists = [item for item in lexical_results if isinstance(item, list)]
        vector_errors = [item for item in vector_results if isinstance(item, BaseException)]
        lexical_errors = [item for item in lexical_results if isinstance(item, BaseException)]
        vector_rows = [row for rows in vector_lists for row in rows]
        lexical_rows = [row for rows in lexical_lists for row in rows]
        log_event(
            "retrieval_candidates_collected",
            run_id,
            vector_results=_candidate_results(vector_queries, vector_results),
            lexical_results=_candidate_results(lexical_searches, lexical_results),
        )
        vector_succeeded = any(isinstance(item, list) for item in vector_results)
        if not vector_succeeded and not lexical_lists:
            raise RuntimeError("All RAG hybrid retrieval queries failed") from (vector_errors[0] if vector_errors else lexical_errors[0])
        if vector_errors or lexical_errors:
            log_event(
                "retrieval_branch_failed",
                run_id,
                level=logging.WARNING,
                branch="vector" if vector_errors else "lexical",
                vector_failed_query_count=len(vector_errors),
                lexical_failed_query_count=len(lexical_errors),
            )
        fused_started = started_at()
        if hasattr(self._retriever, "fuse_candidate_lists"):
            fused = self._retriever.fuse_candidate_lists(vector_lists, lexical_lists, limit)
        else:
            # Compatibility for lightweight retrievers used by integrations while
            # preserving the original single-query fusion contract.
            fused = self._retriever.fuse_candidates(vector_rows, lexical_rows, limit)
        fused = _retain_protected_lexical_candidates(
            fused,
            lexical_searches,
            lexical_results,
            protected_searches,
        )
        fused = _retain_recording_profile_candidates(fused, profile_rows)
        self._operation_completed(
            "retrieve",
            "retrieve.rrf",
            fused,
            fused_started,
            status="degraded" if vector_errors or lexical_errors else "succeeded",
            details={
                "vector_degraded": bool(vector_errors),
                "lexical_degraded": bool(lexical_errors),
                "vector_query_count": len(vector_queries),
                "lexical_query_count": len(lexical_searches),
                "recording_profile_degraded": profile_error is not None,
                "recording_profile_scoped_candidate_count": len(profile_rows),
            },
        )
        overlap = len({row["chunk_id"] for row in vector_rows} & {row["chunk_id"] for row in lexical_rows})
        log_event(
            "hybrid_retrieval_completed",
            run_id,
            query_chars=len(query),
            scope_recording_count=len(filters.recording_ids),
            vector_candidates=len(vector_rows),
            lexical_candidates=len(lexical_rows),
            overlap=overlap,
            fused_candidates=len(fused),
            vector_degraded=bool(vector_errors),
            lexical_degraded=bool(lexical_errors),
            vector_query_count=len(vector_queries),
            lexical_query_count=len(lexical_searches),
            recording_profile_candidates=len(profile_rows),
            recording_profile_degraded=profile_error is not None,
            fused_candidate_refs=_candidate_refs(fused),
            elapsed_ms=elapsed_ms(started),
        )
        return fused

    async def expand_context(self, state: RagGraphState) -> RagStateUpdate:
        node_started = self._node_started(state, "expand_context")
        candidates = state["retrieval_candidates"]
        evidence = await asyncio.to_thread(self._retriever.expand_candidates, candidates)
        self._operation_completed("expand_context", "retrieve.expand", evidence, node_started)
        self._node_completed(
            state,
            "expand_context",
            node_started,
            candidate_count=len(candidates),
            evidence_count=len(evidence),
            recording_count=len({item.recording.id for item in evidence}),
        )
        return {
            "evidence": evidence,
            "answer_evidence": [],
            "message": None if evidence else "没有找到足够相关的录音片段",
        }

    def after_expand_context(self, state: RagGraphState) -> Literal["rerank", "done"]:
        if state["evidence"] and bool(getattr(self._retriever, "rerank_enabled", False)):
            self._transition(state, "expand_context", "rerank", "expanded_evidence_ready")
            return "rerank"
        return "done"

    async def rerank(self, state: RagGraphState) -> RagStateUpdate:
        node_started = self._node_started(state, "rerank")
        evidence = state["evidence"]
        query = state["content_query"]
        invocation = start_invocation("local", usage_kind="rerank")
        try:
            reranked, result = await asyncio.to_thread(self._retriever.rerank_evidence, query, evidence)
        except asyncio.CancelledError as error:
            finish_invocation(invocation, "cancelled", error_type=type(error).__name__)
            raise
        except Exception as error:
            finish_invocation(invocation, "failed", error_type=type(error).__name__)
            finish_span(
                current_span(),
                "failed",
                error_type=type(error).__name__,
                metadata={"evidence_count": len(evidence), "fallback": "expanded_rrf_order", "model_execution": "local"},
            )
            log_event(
                "node_failed",
                state.get("run_id", "standalone"),
                level=logging.WARNING,
                exc_info=True,
                node="rerank",
                error_type=type(error).__name__,
                fallback="expanded_rrf_order",
                model_execution="local",
                reranked_evidence_text=_evidence_log_entries(evidence),
                elapsed_ms=elapsed_ms(node_started),
            )
            self._operation_completed(
                "rerank",
                "retrieve.rerank",
                evidence,
                node_started,
                status="degraded",
                details={"error_type": type(error).__name__, "fallback": "expanded_rrf_order"},
            )
            return {
                "evidence": evidence,
                "rerank_input_tokens": 0,
                "rerank_skipped_candidates": 0,
            }
        if result is None:
            finish_invocation(invocation, "succeeded")
        else:
            finish_invocation(
                invocation,
                "succeeded",
                model=result.model_name,
                prompt_tokens=result.input_tokens,
                completion_tokens=0,
                usage_source="local_tokenizer",
                finish_reason="scored",
            )
        input_tokens = result.input_tokens if result is not None else 0
        skipped = result.skipped_candidates if result is not None else 0
        reranked = _retain_protected_evidence(reranked, evidence, state.get("protected_chunk_ids", []))
        self._operation_completed(
            "rerank",
            "retrieve.rerank",
            reranked,
            node_started,
            details={"input_tokens": input_tokens, "skipped_candidates": skipped},
        )
        self._node_completed(
            state,
            "rerank",
            node_started,
            outcome="succeeded",
            evidence_count=len(evidence),
            output_count=len(reranked),
            input_tokens=input_tokens,
            skipped_candidates=skipped,
            protected_chunk_count=len(state.get("protected_chunk_ids", [])),
            reranked_evidence_text=_evidence_log_entries(reranked),
            model_execution="local",
        )
        return {
            "evidence": reranked,
            "rerank_input_tokens": input_tokens,
            "rerank_skipped_candidates": skipped,
        }


def _evidence_log_entries(evidence: list[Evidence]) -> list[dict[str, object]]:
    """Diagnostic evidence details for application logs; excluded from span metadata."""

    return [
        {
            "index": item.index,
            "recording_id": str(item.recording.id),
            "chunk_id": str(item.chunk.id),
            "start_ms": item.chunk.start_ms,
            "end_ms": item.chunk.end_ms,
        }
        for item in evidence
    ]


def _candidate_results(
    queries: list[str],
    results: list[list[RetrievalCandidateRow] | BaseException],
) -> list[dict[str, object]]:
    return [
        {
            "query": query,
            "status": "failed" if isinstance(result, BaseException) else "succeeded",
            "error_type": type(result).__name__ if isinstance(result, BaseException) else None,
            "candidates": _candidate_refs(result) if isinstance(result, list) else [],
        }
        for query, result in zip(queries, results, strict=True)
    ]


def _candidate_refs(rows: list[RetrievalCandidateRow]) -> list[dict[str, object]]:
    return [
        {
            "chunk_id": str(row["chunk_id"]),
            "recording_id": str(row["recording_id"]),
            "score": row.get("score"),
            "exact_match": row.get("exact_match"),
            "protected_lexical_terms": row.get("protected_lexical_terms", []),
            "retrieved_via_recording_profile": row.get("retrieved_via_recording_profile", False),
            "recording_profile_score": row.get("recording_profile_score"),
            "match_type": row.get("match_type"),
        }
        for row in rows
    ]


def _retain_protected_lexical_candidates(
    fused: list[RetrievalCandidateRow],
    lexical_queries: list[str],
    lexical_results: list[list[RetrievalCandidateRow] | BaseException],
    protected_queries: set[str],
) -> list[RetrievalCandidateRow]:
    """Keep the first exact lexical hit for each meaningful query anchor."""

    fused_by_chunk = {str(row["chunk_id"]): row for row in fused}
    protected_by_chunk: dict[str, set[str]] = {}
    protected_rows: dict[str, RetrievalCandidateRow] = {}
    for lexical_query, result in zip(lexical_queries, lexical_results, strict=True):
        if lexical_query not in protected_queries or not isinstance(result, list):
            continue
        exact_match = next((row for row in result if row.get("exact_match") is True), None)
        if exact_match is None:
            continue
        chunk_id = str(exact_match["chunk_id"])
        protected_rows[chunk_id] = fused_by_chunk.get(chunk_id, exact_match)
        protected_by_chunk.setdefault(chunk_id, set()).add(lexical_query)
    retained: list[RetrievalCandidateRow] = []
    retained_ids: set[str] = set()
    for row in fused:
        chunk_id = str(row["chunk_id"])
        retained_row = row.copy()
        if chunk_id in protected_by_chunk:
            retained_row["protected_lexical_terms"] = sorted(protected_by_chunk[chunk_id])
        retained.append(retained_row)
        retained_ids.add(chunk_id)
    for chunk_id, row in protected_rows.items():
        if chunk_id in retained_ids:
            continue
        retained_row = row.copy()
        retained_row["protected_lexical_terms"] = sorted(protected_by_chunk[chunk_id])
        retained.append(retained_row)
    return retained


def _retain_recording_profile_candidates(
    fused: list[RetrievalCandidateRow],
    profile_rows: list[RetrievalCandidateRow],
) -> list[RetrievalCandidateRow]:
    """Reserve the recording-profile lane without duplicating global candidates."""

    profile_by_chunk = {str(row["chunk_id"]): row for row in profile_rows}
    retained: list[RetrievalCandidateRow] = []
    retained_ids: set[str] = set()
    for row in fused:
        chunk_id = str(row["chunk_id"])
        retained_row = row.copy()
        profile_row = profile_by_chunk.get(chunk_id)
        if profile_row is not None:
            retained_row["retrieved_via_recording_profile"] = True
            profile_score = profile_row.get("recording_profile_score")
            if profile_score is not None:
                retained_row["recording_profile_score"] = profile_score
        retained.append(retained_row)
        retained_ids.add(chunk_id)
    retained.extend(row for row in profile_rows if str(row["chunk_id"]) not in retained_ids)
    return retained


def _retain_protected_evidence(
    reranked: list[Evidence],
    original: list[Evidence],
    protected_chunk_ids: list[str],
) -> list[Evidence]:
    protected_ids = set(protected_chunk_ids)
    protected = [item for item in original if str(item.chunk.id) in protected_ids]
    combined = [*reranked, *protected]
    unique: list[Evidence] = []
    seen: set[str] = set()
    for item in combined:
        chunk_id = str(item.chunk.id)
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        unique.append(item)
    return [item.model_copy(update={"index": index}) for index, item in enumerate(unique, start=1)]
