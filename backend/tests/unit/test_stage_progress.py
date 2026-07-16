from __future__ import annotations

from typing import cast
from uuid import uuid4

from pipeline.contracts import StageRunId
from pipeline.runtime.executor import PipelineProgressReporter
from pipeline.runtime.repository import PipelineRepository


class FakeRepository:
    def __init__(self) -> None:
        self.calls: list[tuple[StageRunId, int, str]] = []

    def update_stage_progress(self, stage_run_id: StageRunId, percent: int, message: str) -> None:
        self.calls.append((stage_run_id, percent, message))


def test_persisted_progress_reporter_throttles_small_unchanged_updates() -> None:
    repository = FakeRepository()
    stage_run_id = StageRunId(uuid4())
    reporter = PipelineProgressReporter(cast(PipelineRepository, repository), stage_run_id)

    reporter.report(1, "加载模型")
    reporter.report(2, "加载模型")
    reporter.report(6, "加载模型")
    reporter.report(7, "模型已加载")

    assert [(percent, message) for _, percent, message in repository.calls] == [
        (1, "加载模型"),
        (2, "加载模型"),
        (6, "加载模型"),
        (7, "模型已加载"),
    ]
