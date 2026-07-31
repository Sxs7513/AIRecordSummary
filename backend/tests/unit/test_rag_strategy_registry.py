from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from l2_core.rag.contracts import RagGraphState
from l2_core.rag.strategies.registry import StrategyRegistry


class _Strategy:
    id: Literal["fact_lookup"] = "fact_lookup"
    version = "1"

    async def invoke(self, state: RagGraphState) -> Mapping[str, object]:
        del state
        return {}


def test_strategy_registry_rejects_duplicates_and_incomplete_registration() -> None:
    registry = StrategyRegistry([_Strategy()])

    try:
        registry.register(_Strategy())
    except ValueError as error:
        assert "already registered" in str(error)
    else:
        raise AssertionError("Expected duplicate strategy registration to fail")

    try:
        registry.validate_complete({"fact_lookup", "metadata_lookup"})
    except ValueError as error:
        assert "metadata_lookup" in str(error)
    else:
        raise AssertionError("Expected incomplete strategy registry to fail")
