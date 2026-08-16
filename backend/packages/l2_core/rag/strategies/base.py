from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, Field

from l2_core.rag.contracts import Evidence, RagGraphState, RagHistoryMessage, ResolvedFilters, StrategyId


def _evidence_list() -> list[Evidence]:
    return []


def _fact_list() -> list[StructuredFact]:
    return []


def _source_list() -> list[dict[str, object]]:
    return []


class StrategyInput(BaseModel):
    run_id: str
    execution_mode: Literal["answer", "retrieval"]
    query: str
    content_query: str
    history: list[RagHistoryMessage]
    scope: ResolvedFilters
    limit: int


class StructuredFact(BaseModel):
    key: Literal["file_name", "duration_seconds", "created_at", "location", "speakers"]
    label: str
    value: str | int | float | bool | list[str] | list[dict[str, str | int | float]] | None
    recording_id: UUID


class StrategyResult(BaseModel):
    status: Literal["ready", "not_found", "needs_clarification"]
    answer_context: str = ""
    corrected_answer_context: str | None = None
    evidence: list[Evidence] = Field(default_factory=_evidence_list)
    facts: list[StructuredFact] = Field(default_factory=_fact_list)
    sources: list[dict[str, object]] = Field(default_factory=_source_list)
    message: str | None = None


class RagStrategy(Protocol):
    @property
    def id(self) -> StrategyId: ...

    @property
    def version(self) -> str: ...

    async def invoke(self, state: RagGraphState) -> Mapping[str, object]: ...
