"""The smallest complete example of a pipeline.

Run it with:

    backend/.venv/bin/python -c \
      'import asyncio; from l1_foundation.pipeline.example import run_example; print(asyncio.run(run_example()))'

The production worker does the same thing, but reads the graph and state from
PostgreSQL instead of using the in-memory loop in ``run_example``.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from l1_foundation.pipeline.contracts import PipelineRunId, PipelineSubjectId, ResourceQueue, RetryPolicy, StageContext, StageResult, StageRunId
from l1_foundation.pipeline.definitions.graph import PipelineDefinition, PipelineNode
from l1_foundation.pipeline.registry import StageRegistry


class PrepareExampleStage:
    """First node: turn the initial payload into an output payload."""

    name = "example_prepare"
    version = "1"
    resource_queue = ResourceQueue.CPU
    retry_policy = RetryPolicy(max_attempts=2, initial_backoff_seconds=1)

    async def run(self, context: StageContext, input_payload: dict[str, Any]) -> StageResult[dict[str, Any]]:
        return StageResult(output={"prepared": True, "subject_id": str(context.subject_id), **input_payload})


class ConsumeExampleStage:
    """Second node: consume the output persisted by its upstream node."""

    name = "example_consume"
    version = "1"
    resource_queue = ResourceQueue.CPU
    retry_policy = RetryPolicy(max_attempts=1)

    async def run(self, context: StageContext, input_payload: dict[str, Any]) -> StageResult[dict[str, Any]]:
        upstream_outputs = input_payload.get("upstream_outputs", {})
        return StageResult(output={"consumed": True, "upstream_outputs": upstream_outputs})


example_pipeline = PipelineDefinition(
    name="example",
    version="1",
    nodes=(
        PipelineNode("prepare", "example_prepare", "1", ResourceQueue.CPU, RetryPolicy(max_attempts=2), input_payload={"source": "example"}),
        PipelineNode("consume", "example_consume", "1", ResourceQueue.CPU, RetryPolicy(max_attempts=1), depends_on=("prepare",)),
    ),
)


def build_example_registry() -> StageRegistry:
    """Return an explicit registry containing the two example plugins."""
    registry = StageRegistry()
    registry.register(PrepareExampleStage())
    registry.register(ConsumeExampleStage())
    return registry


async def run_example() -> dict[str, Any]:
    """Execute ``example_pipeline`` without a database or worker process.

    In production, the owning business domain persists its definition and advances
    ready nodes through its own coordinator. This in-memory runner uses only
    ``example_pipeline`` to determine execution order and dependencies; it does
    not hard-code node execution order.
    """
    registry = build_example_registry()
    subject_id = PipelineSubjectId(UUID("00000000-0000-0000-0000-000000000001"))
    pipeline_run_id = PipelineRunId(UUID("00000000-0000-0000-0000-000000000002"))
    outputs: dict[str, dict[str, Any]] = {}
    for node_index, node in enumerate(example_pipeline.topologically_sorted_nodes(), start=3):
        input_payload = dict(node.input_payload or {})
        input_payload["upstream_outputs"] = {dependency: outputs[dependency] for dependency in node.depends_on}
        context = StageContext(
            subject_id=subject_id,
            pipeline_run_id=pipeline_run_id,
            stage_run_id=StageRunId(UUID(int=node_index)),
            attempt_count=1,
        )
        result = await registry.get(node.stage_name, node.stage_version).run(context, input_payload)
        outputs[node.name] = result.output
    return outputs["consume"]
