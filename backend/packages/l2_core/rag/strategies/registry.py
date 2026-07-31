from __future__ import annotations

from collections.abc import Iterable

from l2_core.rag.contracts import StrategyId
from l2_core.rag.strategies.base import RagStrategy


class StrategyRegistry:
    def __init__(self, strategies: Iterable[RagStrategy] = ()) -> None:
        self._strategies: dict[StrategyId, RagStrategy] = {}
        for strategy in strategies:
            self.register(strategy)

    def register(self, strategy: RagStrategy) -> None:
        if strategy.id in self._strategies:
            raise ValueError(f"RAG strategy is already registered: {strategy.id}")
        if not strategy.version.strip():
            raise ValueError(f"RAG strategy has no version: {strategy.id}")
        self._strategies[strategy.id] = strategy

    def get(self, strategy_id: StrategyId) -> RagStrategy:
        try:
            return self._strategies[strategy_id]
        except KeyError as error:
            raise LookupError(f"RAG strategy is not registered: {strategy_id}") from error

    def all(self) -> tuple[RagStrategy, ...]:
        return tuple(self._strategies.values())

    def validate_complete(self, expected: set[StrategyId]) -> None:
        missing = expected - self._strategies.keys()
        extra = self._strategies.keys() - expected
        if missing or extra:
            raise ValueError(
                f"RAG strategy registry mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
            )
