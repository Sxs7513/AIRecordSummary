from __future__ import annotations

from typing import Protocol
from uuid import UUID


class PipelineLifecycleHooks(Protocol):
    """Business callbacks invoked while the generic runtime advances a pipeline."""

    def stage_succeeded(self, subject_id: UUID, stage_name: str, output: object) -> None: ...

    def run_state_changed(self, subject_id: UUID, status: str, error_message: str | None) -> None: ...
