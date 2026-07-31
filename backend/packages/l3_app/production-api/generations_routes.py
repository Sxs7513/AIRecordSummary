from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from l1_foundation.streaming import RedisStreamStore
from l2_core.generation.contracts import GenerationEvent, GenerationNotFoundError, GenerationSnapshot
from l2_core.generation.redis_runtime import generation_stream_key, redis_stream_sequence
from l2_core.generation.service import GenerationService
from l2_core.rag.queue import GenerationCancelWorkItem
from production_dependencies import (
    CurrentUserDependency,
    GenerationAccessServiceDependency,
    GenerationCommandPublisherDependency,
    GenerationRedisEventStoreDependency,
    GenerationServiceDependency,
)

router = APIRouter()


@router.get("/{run_id}", response_model=GenerationSnapshot)
def get_generation(
    run_id: UUID, service: GenerationServiceDependency, access: GenerationAccessServiceDependency, user: CurrentUserDependency
) -> GenerationSnapshot:
    """Return a durable view of the generation's current state."""
    try:
        access.require_view(run_id, user)
        return service.get(run_id)
    except GenerationNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generation run not found") from error
    except PermissionError as error:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Generation access denied") from error


@router.get("/{run_id}/events")
async def stream_generation_events(
    run_id: UUID,
    request: Request,
    service: GenerationServiceDependency,
    access: GenerationAccessServiceDependency,
    redis_event_store: GenerationRedisEventStoreDependency,
    user: CurrentUserDependency,
    after: Annotated[str | None, Query(pattern=r"^\d+-\d+$")] = None,
    last_event_id: Annotated[str | None, Header()] = None,
) -> StreamingResponse:
    """Replay persisted events, then keep one HTTP SSE subscriber attached to live events."""
    resume_after = _resume_after(after, last_event_id)
    try:
        access.require_view(run_id, user)
        snapshot = service.get(run_id)
    except GenerationNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generation run not found") from error
    except PermissionError as error:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Generation access denied") from error
    stream = _stream_events(request, service, redis_event_store, run_id, snapshot, resume_after)
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.delete("/{run_id}", response_model=GenerationSnapshot, status_code=status.HTTP_202_ACCEPTED)
async def cancel_generation(
    run_id: UUID,
    service: GenerationServiceDependency,
    access: GenerationAccessServiceDependency,
    publisher: GenerationCommandPublisherDependency,
    user: CurrentUserDependency,
) -> GenerationSnapshot:
    """Reliably request cooperative cancellation through Kafka."""
    try:
        access.require_view(run_id, user)
        snapshot = service.get(run_id)
        if snapshot.status.is_terminal:
            return snapshot
        try:
            await publisher.cancel(GenerationCancelWorkItem(generation_id=run_id))
        except Exception as error:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Generation cancellation queue unavailable") from error
        return snapshot.model_copy(update={"cancel_requested": True})
    except GenerationNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generation run not found") from error
    except PermissionError as error:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Generation access denied") from error


async def _stream_events(
    request: Request,
    service: GenerationService,
    redis_event_store: RedisStreamStore,
    run_id: UUID,
    snapshot: GenerationSnapshot,
    resume_after: str | None,
) -> AsyncGenerator[str]:
    cursor = resume_after if resume_after is not None else service.cursor(run_id)
    if resume_after is None:
        yield encode_sse(_snapshot_event(snapshot), cursor)
        if snapshot.status.value in {"succeeded", "failed", "cancelled"}:
            return
    while not await request.is_disconnected():
        events = await redis_event_store.read(
            generation_stream_key(run_id),
            cursor,
            block_ms=request.app.state.settings.redis_stream_block_ms,
        )
        if not events:
            yield ": heartbeat\n\n"
            continue
        for streamed in events:
            cursor = streamed.id
            event = GenerationEvent(
                run_id=run_id,
                seq=redis_stream_sequence(streamed.id),
                type=streamed.type,
                at=datetime.now(UTC),
                data=streamed.data,
            )
            yield encode_sse(event, streamed.id)
            if event.type in {"output.final", "run.error", "run.cancelled"}:
                return


def _resume_after(after: str | None, last_event_id: str | None) -> str | None:
    if after is not None:
        return after
    if last_event_id is None or not last_event_id.strip():
        return None
    parts = last_event_id.split("-", maxsplit=1)
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Last-Event-ID must be a Redis Stream ID")
    return last_event_id


def _snapshot_event(snapshot: GenerationSnapshot) -> GenerationEvent:
    return GenerationEvent(
        run_id=snapshot.id,
        seq=snapshot.last_sequence,
        type="snapshot",
        at=datetime.now(UTC),
        data={
            "kind": snapshot.kind.value,
            "priority": snapshot.priority.value,
            "status": snapshot.status.value,
            "phase": snapshot.phase.model_dump(mode="json") if snapshot.phase is not None else None,
            "blocks": [block.model_dump(mode="json") for block in snapshot.blocks],
            "sources": snapshot.sources,
            "output": snapshot.output,
            "error": _error_payload(snapshot),
        },
    )


def _error_payload(snapshot: GenerationSnapshot) -> dict[str, str] | None:
    if snapshot.error_code is None or snapshot.error_message is None:
        return None
    return {"code": snapshot.error_code, "message": snapshot.error_message}


def encode_sse(event: GenerationEvent, event_id: str | None = None) -> str:
    """Encode one protocol envelope as a complete HTTP SSE frame."""
    data = json.dumps(event.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
    return f"id: {event_id or event.seq}\nevent: {event.type}\ndata: {data}\n\n"
