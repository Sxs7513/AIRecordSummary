from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

from l2_core.application.recordings import RecordingService


def test_recording_processing_stages_follow_the_declared_graph_order() -> None:
    stages = [
        {"node_name": "summary_embedding_indexing"},
        {"node_name": "generate_summary"},
        {"node_name": "normalize_audio"},
        {"node_name": "correct_asr_windows"},
        {"node_name": "align_transcript"},
        {"node_name": "transcribe_qwen_asr"},
    ]

    ordered = RecordingService.order_stage_rows("recording_processing", stages)

    assert [stage["node_name"] for stage in ordered] == [
        "normalize_audio",
        "transcribe_qwen_asr",
        "correct_asr_windows",
        "align_transcript",
        "generate_summary",
        "summary_embedding_indexing",
    ]


def test_recording_detail_falls_back_to_database_processing_projection_without_redis() -> None:
    service = cast(Any, object.__new__(RecordingService))
    service._processing_state_store = None
    recording_id = uuid4()
    processing_id = uuid4()
    now = datetime.now(UTC)

    runs = service._runtime_pipeline_runs(
        {
            "id": recording_id,
            "processing_id": processing_id,
            "processing_pipeline_name": "recording_processing",
            "processing_pipeline_version": "25",
            "status": "completed",
            "error_message": None,
            "created_at": now,
            "updated_at": now,
        }
    )

    assert len(runs) == 1
    assert cast(UUID, runs[0]["id"]) == processing_id
    assert runs[0]["recording_id"] == recording_id
    assert runs[0]["status"] == "succeeded"
    assert runs[0]["finished_at"] == now
