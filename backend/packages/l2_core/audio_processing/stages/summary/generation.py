from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from l2_core.generation.contracts import CreateGenerationCommand, GenerationAccessScope, GenerationKind, GenerationPriority, GenerationSnapshot
from l2_core.generation.service import GenerationService


def create_pipeline_summary_generation(
    generation_service: GenerationService,
    stage_run_id: UUID,
    recording_id: UUID,
    attempt_count: int,
    input_payload: dict[str, Any],
) -> GenerationSnapshot:
    """Create the recording-owned stream emitted by the summary pipeline stage."""
    return generation_service.create(
        CreateGenerationCommand(
            kind=GenerationKind.TEXT,
            priority=GenerationPriority.BACKGROUND,
            idempotency_key=f"recording-summary:{stage_run_id}:attempt:{attempt_count}",
            parent_type="stage_run",
            parent_id=str(stage_run_id),
            access_scope=GenerationAccessScope(subject_type="recording", subject_id=recording_id),
            input={"stage_attempt": attempt_count, **input_payload},
        )
    )


def create_manual_summary_generation(generation_service: GenerationService, recording_id: UUID) -> GenerationSnapshot:
    """Create an interactive recording-owned stream for a manual summary regeneration."""
    operation_id = uuid4()
    return generation_service.create(
        CreateGenerationCommand(
            kind=GenerationKind.TEXT,
            priority=GenerationPriority.INTERACTIVE,
            idempotency_key=f"recording-summary-regeneration:{recording_id}:{operation_id}",
            parent_type="recording",
            parent_id=str(recording_id),
            access_scope=GenerationAccessScope(subject_type="recording", subject_id=recording_id),
            input={"operation": "summary_regeneration"},
        )
    )
