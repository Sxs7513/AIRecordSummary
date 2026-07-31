from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast
from uuid import uuid4, uuid5

from l1_foundation.infrastructure.storage.local import LocalStorage
from l1_foundation.messaging import EventEnvelope
from l1_foundation.pipeline.contracts import ArtifactRef
from l1_foundation.streaming import SyncRedisStreamStore
from l2_core.access.recordings import RecordingAccessService
from l2_core.application.processing_queue import ProcessingCommandPublisher, queued_processing_state
from l2_core.application.recordings import RecordingService
from l2_core.auth.contracts import CurrentUser


class _Producer:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str, EventEnvelope]] = []

    async def publish(self, topic: str, key: str, event: EventEnvelope) -> None:
        self.messages.append((topic, key, event))


class _Access:
    def require_edit(self, _recording_id: object, _user: object) -> None:
        pass


class _StateStore:
    def __init__(self, values: dict[str, dict[str, object]]) -> None:
        self.values = values

    def get_state(self, key: str) -> dict[str, object] | None:
        return self.values.get(key)


def test_queued_processing_state_is_immediately_renderable_by_recording_detail() -> None:
    processing_id = uuid4()
    recording_id = uuid4()

    state = queued_processing_state(processing_id, recording_id, "recording_processing", "4")

    assert state == {
        "processing_id": str(processing_id),
        "subject_type": "recording",
        "subject_id": str(recording_id),
        "pipeline_name": "recording_processing",
        "pipeline_version": "4",
        "status": "queued",
        "stages": {},
        "created_at": state["created_at"],
        "updated_at": state["updated_at"],
    }
    assert state["created_at"] == state["updated_at"]


def test_recording_deletion_requests_processing_cancellation() -> None:
    recording_id = uuid4()
    producer = _Producer()
    service = object.__new__(RecordingService)
    service._processing_publisher = ProcessingCommandPublisher(producer)  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage]

    asyncio.run(service._request_processing_cancel(recording_id))  # pyright: ignore[reportPrivateUsage]

    [(topic, key, event)] = producer.messages
    assert topic == "processing.cancel"
    assert key == str(recording_id)
    assert event.event_type == "processing.cancel.requested"
    assert event.payload == {"subject_type": "recording", "subject_id": str(recording_id)}


def test_embedding_retry_publishes_a_dedicated_command_with_existing_chunks() -> None:
    processing_id = uuid4()
    recording_id = uuid4()
    chunks = ArtifactRef(
        artifact_type="search.chunks",
        artifact_version="1",
        producer_stage="build_search_chunks",
        uri="artifacts/search-chunks.json",
        checksum="checksum",
    )
    producer = _Producer()
    publisher = ProcessingCommandPublisher(producer)  # type: ignore[arg-type]

    asyncio.run(publisher.retry_embedding_index(processing_id, recording_id, chunks))

    [(topic, key, event)] = producer.messages
    assert topic == "processing.commands"
    assert key == str(processing_id)
    assert event.event_type == "processing.embedding-index.requested"
    assert event.processing_id == processing_id
    assert event.payload["subject_id"] == str(recording_id)
    assert event.payload["chunks"] == chunks.model_dump(mode="json")


def test_embedding_retry_reuses_the_current_processing_chunk_artifact(tmp_path: Path) -> None:
    processing_id = uuid4()
    recording_id = uuid4()
    chunks = ArtifactRef(
        artifact_type="search.chunks",
        artifact_version="1",
        producer_stage="build_search_chunks",
        uri="artifacts/search-chunks.json",
    )
    storage = LocalStorage(tmp_path)
    artifact_path = storage.resolve(chunks.uri)
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("{}", encoding="utf-8")
    state_store = _StateStore(
        {
            f"recording:{recording_id}:processing": {"processing_id": str(processing_id)},
            f"processing:{processing_id}:state": {
                "stages": {
                    "build_search_chunks": {
                        "status": "succeeded",
                        "artifacts": [chunks.model_dump(mode="json")],
                    }
                }
            },
        }
    )
    producer = _Producer()
    service = object.__new__(RecordingService)
    service._access = cast(RecordingAccessService, _Access())  # pyright: ignore[reportPrivateUsage]
    service._storage = storage  # pyright: ignore[reportPrivateUsage]
    service._processing_publisher = ProcessingCommandPublisher(producer)  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage]
    service._processing_state_store = cast(SyncRedisStreamStore, state_store)  # pyright: ignore[reportPrivateUsage]

    stage_run_id = asyncio.run(service.retry_embedding_indexing(cast(CurrentUser, object()), recording_id))

    assert stage_run_id == uuid5(processing_id, "embedding_indexing")
    [(_, _, event)] = producer.messages
    assert event.event_type == "processing.embedding-index.requested"
