from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4, uuid5

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Connection

from l1_foundation.messaging import KafkaEventProducer, OutboxRepository, Topics, new_event
from l1_foundation.pipeline.contracts import ArtifactRef


class ProcessingQueueUnavailableError(RuntimeError):
    """Raised when Kafka does not acknowledge a Processing command."""


PROCESSING_ID_NAMESPACE = UUID("97f66f2a-2034-4e14-93b7-b3c5ff78b289")


def stable_recording_processing_id(
    workspace_id: UUID,
    owner_user_id: UUID,
    pipeline_name: str,
    pipeline_version: str,
    file_md5: str,
) -> UUID:
    """Identify one user's processing of identical bytes with one immutable pipeline."""
    identity = json.dumps(
        {
            "file_md5": file_md5.lower(),
            "owner_user_id": str(owner_user_id),
            "pipeline_name": pipeline_name,
            "pipeline_version": pipeline_version,
            "workspace_id": str(workspace_id),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return uuid5(PROCESSING_ID_NAMESPACE, identity)


class ProcessingWorkItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    processing_id: UUID
    subject_type: str
    subject_id: UUID
    pipeline_name: str
    pipeline_version: str
    initial_artifacts: list[ArtifactRef]


class ProcessingCancelWorkItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    subject_type: str
    subject_id: UUID


class EmbeddingReindexWorkItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    processing_id: UUID
    subject_id: UUID
    chunks: ArtifactRef


def queued_processing_state(
    processing_id: UUID,
    subject_id: UUID,
    pipeline_name: str,
    pipeline_version: str,
) -> dict[str, Any]:
    """Build the API-side live projection visible before a worker starts the DAG."""
    now = datetime.now(UTC).isoformat()
    return {
        "processing_id": str(processing_id),
        "subject_type": "recording",
        "subject_id": str(subject_id),
        "pipeline_name": pipeline_name,
        "pipeline_version": pipeline_version,
        "status": "queued",
        "stages": {},
        "created_at": now,
        "updated_at": now,
    }


class ProcessingCommandPublisher:
    def __init__(self, producer: KafkaEventProducer, outbox: OutboxRepository | None = None) -> None:
        self._kafka_producer = producer
        self._outbox = outbox or OutboxRepository()

    def enqueue_recording(
        self,
        connection: Connection,
        subject_id: UUID,
        pipeline_name: str,
        pipeline_version: str,
        source: ArtifactRef,
        *,
        processing_id: UUID,
        workspace_id: UUID | None = None,
    ) -> UUID:
        item = ProcessingWorkItem(
            processing_id=processing_id,
            subject_type="recording",
            subject_id=subject_id,
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
            initial_artifacts=[source],
        )
        event = new_event(
            "processing.requested",
            "production-api",
            correlation_id=processing_id,
            workspace_id=workspace_id,
            processing_id=processing_id,
            payload=item.model_dump(mode="json"),
        )
        self._outbox.enqueue(
            connection,
            channel="processing-command",
            topic=Topics.PROCESSING_COMMANDS,
            partition_key=str(processing_id),
            aggregate_type="processing",
            aggregate_id=processing_id,
            event=event,
        )
        return processing_id

    def enqueue_recording_retry(
        self,
        connection: Connection,
        subject_id: UUID,
        pipeline_name: str,
        pipeline_version: str,
        source: ArtifactRef,
        *,
        processing_id: UUID,
        workspace_id: UUID | None = None,
    ) -> UUID:
        """Reopen one terminal processing run while retaining its artifact identity."""
        item = ProcessingWorkItem(
            processing_id=processing_id,
            subject_type="recording",
            subject_id=subject_id,
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
            initial_artifacts=[source],
        )
        event = new_event(
            "processing.retry.requested",
            "production-api",
            correlation_id=processing_id,
            workspace_id=workspace_id,
            processing_id=processing_id,
            payload=item.model_dump(mode="json"),
        )
        self._outbox.enqueue(
            connection,
            channel="processing-command",
            topic=Topics.PROCESSING_COMMANDS,
            partition_key=str(processing_id),
            aggregate_type="processing",
            aggregate_id=processing_id,
            event=event,
        )
        return processing_id

    def enqueue_cancel(self, connection: Connection, subject_id: UUID) -> None:
        item = ProcessingCancelWorkItem(subject_type="recording", subject_id=subject_id)
        event = new_event(
            "processing.cancel.requested",
            "production-api",
            correlation_id=subject_id,
            payload=item.model_dump(mode="json"),
        )
        self._outbox.enqueue(
            connection,
            channel="processing-command",
            topic=Topics.PROCESSING_CANCEL,
            partition_key=str(subject_id),
            aggregate_type="recording",
            aggregate_id=subject_id,
            event=event,
        )

    def enqueue_embedding_retry(self, connection: Connection, processing_id: UUID, subject_id: UUID, chunks: ArtifactRef) -> None:
        item = EmbeddingReindexWorkItem(processing_id=processing_id, subject_id=subject_id, chunks=chunks)
        event = new_event(
            "processing.embedding-index.requested",
            "production-api",
            correlation_id=processing_id,
            processing_id=processing_id,
            payload=item.model_dump(mode="json"),
        )
        self._outbox.enqueue(
            connection,
            channel="processing-command",
            topic=Topics.PROCESSING_COMMANDS,
            partition_key=str(processing_id),
            aggregate_type="processing",
            aggregate_id=processing_id,
            event=event,
        )

    async def submit_recording(
        self,
        subject_id: UUID,
        pipeline_name: str,
        pipeline_version: str,
        source: ArtifactRef,
        *,
        processing_id: UUID | None = None,
        workspace_id: UUID | None = None,
    ) -> UUID:
        processing_id = processing_id or uuid4()
        item = ProcessingWorkItem(
            processing_id=processing_id,
            subject_type="recording",
            subject_id=subject_id,
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
            initial_artifacts=[source],
        )
        try:
            await self._kafka_producer.publish(
                Topics.PROCESSING_COMMANDS,
                str(processing_id),
                new_event(
                    "processing.requested",
                    "production-api",
                    correlation_id=processing_id,
                    workspace_id=workspace_id,
                    processing_id=processing_id,
                    payload=item.model_dump(mode="json"),
                ),
            )
        except Exception as error:
            raise ProcessingQueueUnavailableError("Processing queue unavailable") from error
        return processing_id

    async def cancel_recording(self, subject_id: UUID) -> None:
        item = ProcessingCancelWorkItem(subject_type="recording", subject_id=subject_id)
        try:
            await self._kafka_producer.publish(
                Topics.PROCESSING_CANCEL,
                str(subject_id),
                new_event(
                    "processing.cancel.requested",
                    "production-api",
                    correlation_id=subject_id,
                    payload=item.model_dump(mode="json"),
                ),
            )
        except Exception as error:
            raise ProcessingQueueUnavailableError("Processing cancellation queue unavailable") from error

    async def retry_embedding_index(self, processing_id: UUID, subject_id: UUID, chunks: ArtifactRef) -> None:
        item = EmbeddingReindexWorkItem(processing_id=processing_id, subject_id=subject_id, chunks=chunks)
        try:
            await self._kafka_producer.publish(
                Topics.PROCESSING_COMMANDS,
                str(processing_id),
                new_event(
                    "processing.embedding-index.requested",
                    "production-api",
                    correlation_id=processing_id,
                    processing_id=processing_id,
                    payload=item.model_dump(mode="json"),
                ),
            )
        except Exception as error:
            raise ProcessingQueueUnavailableError("Embedding reindex queue unavailable") from error
