from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from api.dependencies import CurrentUserDependency, GenerationAccessServiceDependency
from generation.contracts import GenerationEvent, GenerationNotFoundError, GenerationSnapshot
from generation.hub import GenerationStreamHub, GenerationSubscription
from generation.service import GenerationService

router = APIRouter()


def get_generation_service(request: Request) -> GenerationService:
    """Resolve the generation application service owned by the FastAPI app."""
    return request.app.state.generation_service


def get_generation_hub(request: Request) -> GenerationStreamHub:
    """Resolve the current process's real-time fan-out hub."""
    return request.app.state.generation_hub


GenerationServiceDependency = Annotated[GenerationService, Depends(get_generation_service)]
GenerationHubDependency = Annotated[GenerationStreamHub, Depends(get_generation_hub)]


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
    hub: GenerationHubDependency,
    user: CurrentUserDependency,
    after: Annotated[int | None, Query(ge=0)] = None,
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
    subscription = hub.subscribe(run_id)
    stream = _stream_events(request, service, hub, run_id, snapshot, resume_after, subscription)
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.delete("/{run_id}", response_model=GenerationSnapshot, status_code=status.HTTP_202_ACCEPTED)
def cancel_generation(
    run_id: UUID, service: GenerationServiceDependency, access: GenerationAccessServiceDependency, user: CurrentUserDependency
) -> GenerationSnapshot:
    """Request cooperative cancellation at the executor's next token or chunk boundary."""
    try:
        access.require_view(run_id, user)
        return service.cancel(run_id)
    except GenerationNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generation run not found") from error
    except PermissionError as error:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Generation access denied") from error


async def _stream_events(
    request: Request,
    service: GenerationService,
    hub: GenerationStreamHub,
    run_id: UUID,
    snapshot: GenerationSnapshot,
    resume_after: int | None,
    subscription: GenerationSubscription,
) -> AsyncGenerator[str]:
    last_sent = resume_after if resume_after is not None else snapshot.last_sequence
    try:
        if resume_after is None:
            yield encode_sse(_snapshot_event(snapshot))
            if snapshot.status.value in {"succeeded", "failed", "cancelled"}:
                return
        for event in service.store.events_after(run_id, last_sent):
            if event.seq > last_sent:
                last_sent = event.seq
                yield encode_sse(event)
                if event.type in {"output.final", "run.error", "run.cancelled"}:
                    return
        while not await request.is_disconnected():
            event = await asyncio.to_thread(subscription.get, 15.0)
            if event is None:
                yield ": heartbeat\n\n"
                continue
            if event.seq <= last_sent:
                continue
            last_sent = event.seq
            yield encode_sse(event)
            if event.type in {"output.final", "run.error", "run.cancelled"}:
                return
    finally:
        hub.unsubscribe(run_id, subscription)


def _resume_after(after: int | None, last_event_id: str | None) -> int | None:
    if after is not None:
        return after
    if last_event_id is None or not last_event_id.strip():
        return None
    try:
        value = int(last_event_id)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Last-Event-ID must be a non-negative integer") from error
    if value < 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Last-Event-ID must be a non-negative integer")
    return value


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


def encode_sse(event: GenerationEvent) -> str:
    """Encode one protocol envelope as a complete HTTP SSE frame."""
    data = json.dumps(event.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
    return f"id: {event.seq}\nevent: {event.type}\ndata: {data}\n\n"
