from __future__ import annotations

from typing import Any

from l1_foundation.pipeline.contracts import RetryPolicy, StageContext, StageResult


class NoopStage:
    """Typed stage fixture used by unit tests and local runtime smoke tests."""

    name = "noop"
    version = "1"
    retry_policy = RetryPolicy(max_attempts=1)

    async def run(self, context: StageContext, input_payload: dict[str, Any]) -> StageResult[dict[str, Any]]:
        return StageResult(output={"recording_id": str(context.subject_id), **input_payload})
