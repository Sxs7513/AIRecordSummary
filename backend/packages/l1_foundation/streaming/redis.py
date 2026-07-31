from __future__ import annotations

# pyright: reportUnknownMemberType=false
import json
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, cast

from redis import Redis as SyncRedis
from redis.asyncio import Redis


@dataclass(frozen=True, slots=True)
class RedisStreamEvent:
    id: str
    type: str
    data: dict[str, Any]


class RedisStreamStore:
    """Redis-backed active state and bounded resumable event streams."""

    def __init__(self, client: Redis, *, maxlen: int = 10_000, terminal_ttl_seconds: int = 86_400) -> None:
        self._client = client
        self._maxlen = maxlen
        self._terminal_ttl_seconds = terminal_ttl_seconds

    @classmethod
    def from_url(cls, url: str, *, maxlen: int = 10_000, terminal_ttl_seconds: int = 86_400) -> RedisStreamStore:
        return cls(Redis.from_url(url, decode_responses=True), maxlen=maxlen, terminal_ttl_seconds=terminal_ttl_seconds)

    async def ping(self) -> None:
        await cast(Awaitable[bool], self._client.ping())

    async def close(self) -> None:
        await self._client.aclose()

    async def append(self, stream: str, event_type: str, data: dict[str, Any]) -> str:
        event_id = await self._client.xadd(
            stream,
            {"type": event_type, "data": json.dumps(data, ensure_ascii=False, separators=(",", ":"))},
            maxlen=self._maxlen,
            approximate=True,
        )
        return str(event_id)

    async def read(self, stream: str, after: str, *, block_ms: int = 15_000, count: int = 100) -> list[RedisStreamEvent]:
        response = await self._client.xread({stream: after}, count=count, block=block_ms)
        events: list[RedisStreamEvent] = []
        for _, entries in response:
            for event_id, fields in entries:
                events.append(RedisStreamEvent(str(event_id), str(fields["type"]), json.loads(str(fields["data"]))))
        return events

    async def set_state(self, key: str, state: dict[str, Any], *, ttl_seconds: int | None = None) -> None:
        payload = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        if ttl_seconds is None:
            await self._client.set(key, payload)
        else:
            await self._client.set(key, payload, ex=ttl_seconds)

    async def set_state_if_absent(self, key: str, state: dict[str, Any]) -> bool:
        """Create an initial projection without overwriting a faster worker update."""
        return bool(
            await self._client.set(
                key,
                json.dumps(state, ensure_ascii=False, separators=(",", ":")),
                nx=True,
            )
        )

    async def get_state(self, key: str) -> dict[str, Any] | None:
        value = await self._client.get(key)
        if value is None:
            return None
        parsed = json.loads(str(value))
        return cast(dict[str, Any], parsed) if isinstance(parsed, dict) else None

    async def finish(self, state_key: str, stream_key: str) -> None:
        pipeline = self._client.pipeline(transaction=True)
        pipeline.expire(state_key, self._terminal_ttl_seconds)
        pipeline.expire(stream_key, self._terminal_ttl_seconds)
        await pipeline.execute()

    async def request_cancel(self, task_id: str) -> None:
        await self._client.set(f"task:{task_id}:cancel", "1", ex=self._terminal_ttl_seconds)

    async def is_cancel_requested(self, task_id: str) -> bool:
        return bool(await self._client.exists(f"task:{task_id}:cancel"))


class SyncRedisStreamStore:
    """Synchronous writer used by model callbacks and thread-based handlers."""

    def __init__(self, client: SyncRedis, *, maxlen: int = 10_000, terminal_ttl_seconds: int = 86_400) -> None:
        self._client = client
        self._maxlen = maxlen
        self._terminal_ttl_seconds = terminal_ttl_seconds

    @classmethod
    def from_url(cls, url: str, *, maxlen: int = 10_000, terminal_ttl_seconds: int = 86_400) -> SyncRedisStreamStore:
        client = SyncRedis.from_url(url, decode_responses=True)
        return cls(client, maxlen=maxlen, terminal_ttl_seconds=terminal_ttl_seconds)

    def ping(self) -> None:
        self._client.ping()

    def close(self) -> None:
        self._client.close()

    def append(self, stream: str, event_type: str, data: dict[str, Any]) -> str:
        event_id = self._client.xadd(
            stream,
            {"type": event_type, "data": json.dumps(data, ensure_ascii=False, separators=(",", ":"))},
            maxlen=self._maxlen,
            approximate=True,
        )
        return str(event_id)

    def set_state(self, key: str, state: dict[str, Any], *, ttl_seconds: int | None = None) -> None:
        payload = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        if ttl_seconds is None:
            self._client.set(key, payload)
        else:
            self._client.set(key, payload, ex=ttl_seconds)

    def set_state_if_absent(self, key: str, state: dict[str, Any]) -> bool:
        """Create an initial projection without overwriting a faster worker update."""
        return bool(
            self._client.set(
                key,
                json.dumps(state, ensure_ascii=False, separators=(",", ":")),
                nx=True,
            )
        )

    def get_state(self, key: str) -> dict[str, Any] | None:
        value = self._client.get(key)
        if value is None:
            return None
        parsed = json.loads(str(value))
        return cast(dict[str, Any], parsed) if isinstance(parsed, dict) else None

    def get_states_by_pattern(self, pattern: str) -> dict[str, dict[str, Any]]:
        keys = cast(tuple[str, ...], tuple(self._client.scan_iter(match=pattern)))
        if not keys:
            return {}
        values = cast(list[str | None], self._client.mget(keys))
        states: dict[str, dict[str, Any]] = {}
        for key, value in zip(keys, values, strict=True):
            if value is None:
                continue
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                states[key] = cast(dict[str, Any], parsed)
        return states

    def read(self, stream: str, after: str, *, block_ms: int = 15_000, count: int = 100) -> list[RedisStreamEvent]:
        response = cast(
            list[tuple[str, list[tuple[str, dict[str, str]]]]],
            self._client.xread({stream: after}, count=count, block=block_ms),
        )
        events: list[RedisStreamEvent] = []
        for _, entries in response:
            for event_id, fields in entries:
                events.append(RedisStreamEvent(str(event_id), str(fields["type"]), json.loads(str(fields["data"]))))
        return events

    def finish(self, state_key: str, stream_key: str, *, ttl_seconds: int | None = None) -> None:
        ttl = ttl_seconds or self._terminal_ttl_seconds
        pipeline = self._client.pipeline(transaction=True)
        pipeline.expire(state_key, ttl)
        pipeline.expire(stream_key, ttl)
        pipeline.execute()

    def expire(self, *keys: str, ttl_seconds: int | None = None) -> None:
        if not keys:
            return
        ttl = ttl_seconds or self._terminal_ttl_seconds
        pipeline = self._client.pipeline(transaction=True)
        for key in keys:
            pipeline.expire(key, ttl)
        pipeline.execute()

    def expire_pattern(self, pattern: str, *, ttl_seconds: int | None = None) -> None:
        keys = cast(tuple[str, ...], tuple(self._client.scan_iter(match=pattern)))
        self.expire(*keys, ttl_seconds=ttl_seconds)

    def request_cancel(self, task_id: str) -> None:
        self._client.set(f"task:{task_id}:cancel", "1", ex=self._terminal_ttl_seconds)

    def is_cancel_requested(self, task_id: str) -> bool:
        return bool(self._client.exists(f"task:{task_id}:cancel"))

    def delete(self, *keys: str) -> int:
        if not keys:
            return 0
        return cast(int, self._client.delete(*keys))

    def delete_pattern(self, pattern: str) -> int:
        keys = cast(tuple[str, ...], tuple(self._client.scan_iter(match=pattern)))
        return self.delete(*keys)
