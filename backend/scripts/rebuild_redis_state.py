from __future__ import annotations

# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
import asyncio
from typing import cast
from uuid import uuid4

from aiokafka import AIOKafkaConsumer

from l1_foundation.messaging import EventEnvelope, Topics
from l1_foundation.settings import get_settings
from l1_foundation.streaming import RedisStreamStore


async def rebuild() -> None:
    """Replay compacted Kafka state topics into an empty Redis instance, then exit."""
    settings = get_settings()
    redis = RedisStreamStore.from_url(
        settings.redis_url,
        maxlen=settings.redis_stream_maxlen,
        terminal_ttl_seconds=settings.redis_terminal_ttl_seconds,
    )
    consumer = AIOKafkaConsumer(
        *Topics.COMPACTED,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=f"redis-state-rebuild-{uuid4()}",
        client_id=f"{settings.kafka_client_id}-redis-rebuild",
        enable_auto_commit=False,
        auto_offset_reset="earliest",
    )
    await consumer.start()
    try:
        while True:
            batches = await consumer.getmany(timeout_ms=1000, max_records=500)
            for records in batches.values():
                for record in records:
                    if record.value is None:
                        continue
                    event = EventEnvelope.model_validate_json(cast(bytes, record.value))
                    await _restore(redis, record.topic, event)
            assignments = consumer.assignment()
            if assignments:
                ends = await consumer.end_offsets(assignments)
                caught_up = True
                for partition in assignments:
                    if await consumer.position(partition) < ends[partition]:
                        caught_up = False
                        break
                if caught_up:
                    return
    finally:
        await consumer.stop()
        await redis.close()


async def _restore(redis: RedisStreamStore, topic: str, event: EventEnvelope) -> None:
    if topic == Topics.PROCESSING_STATE and event.processing_id is not None:
        state_key = f"processing:{event.processing_id}:state"
        stream_key = f"processing:{event.processing_id}:events"
        await redis.set_state(state_key, event.payload)
        subject_type = event.payload.get("subject_type")
        subject_id = event.payload.get("subject_id")
        if subject_type == "recording" and subject_id is not None:
            await redis.set_state(
                f"recording:{subject_id}:processing",
                {"processing_id": str(event.processing_id)},
            )
        if event.payload.get("status") in {"succeeded", "partial_failed", "failed", "cancelled"}:
            await redis.finish(state_key, stream_key)
    elif topic == Topics.COMPUTE_STATE and event.task_id is not None:
        state_key = f"compute:{event.task_id}:state"
        stream_key = f"compute:{event.task_id}:events"
        await redis.set_state(state_key, event.payload)
        if event.payload.get("status") in {"succeeded", "failed", "cancelled"}:
            await redis.finish(state_key, stream_key)
    elif topic == Topics.GENERATION_STATE and event.generation_id is not None:
        snapshot = event.payload.get("snapshot")
        command = event.payload.get("command")
        if isinstance(snapshot, dict):
            state = cast(dict[str, object], dict(snapshot))
            state["cursor"] = "0-0"
            if isinstance(command, dict):
                state["command"] = command
                idempotency_key = command.get("idempotency_key")
                if idempotency_key is not None:
                    await redis.set_state(
                        f"generation:idempotency:{idempotency_key}",
                        {"run_id": str(event.generation_id)},
                    )
            state_key = f"generation:{event.generation_id}"
            stream_key = f"generation:{event.generation_id}:events"
            await redis.set_state(state_key, state)
            if state.get("status") in {"succeeded", "failed", "cancelled"}:
                await redis.finish(state_key, stream_key)


if __name__ == "__main__":
    asyncio.run(rebuild())
