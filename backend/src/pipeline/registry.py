from __future__ import annotations

from typing import Any

from pipeline.contracts import Stage


class StageRegistry:
    """Explicit registry for built-in stage plugins."""

    def __init__(self) -> None:
        self._stages: dict[tuple[str, str], Stage[Any, Any]] = {}

    def register(self, stage: Stage[Any, Any]) -> None:
        key = (stage.name, stage.version)
        if key in self._stages:
            raise ValueError(f"Stage is already registered: {stage.name}@{stage.version}")
        self._stages[key] = stage

    def get(self, name: str, version: str) -> Stage[Any, Any]:
        try:
            return self._stages[(name, version)]
        except KeyError as error:
            raise KeyError(f"Stage is not registered: {name}@{version}") from error
