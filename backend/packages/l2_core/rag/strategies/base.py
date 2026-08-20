from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel

from l2_core.rag.contracts import (
    RagGraphState,
    RagHistoryMessage,
    RagStateUpdate,
    ResolvedFilters,
    StrategyId,
    StrategyResult,
    StructuredFact,
    StructuredFactKey,
)

__all__ = ["RagStrategy", "StrategyInput", "StrategyResult", "StructuredFact", "StructuredFactKey"]


class StrategyInput(BaseModel):
    run_id: str
    execution_mode: Literal["answer", "retrieval"]
    query: str
    content_query: str
    history: list[RagHistoryMessage]
    scope: ResolvedFilters
    limit: int


class RagStrategy(Protocol):
    @property
    def id(self) -> StrategyId: ...

    @property
    def version(self) -> str: ...

    async def invoke(self, state: RagGraphState) -> RagStateUpdate: ...
