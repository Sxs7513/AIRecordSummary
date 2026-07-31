from __future__ import annotations

import asyncio
import json
from typing import Any

from l1_foundation.streaming.redis import RedisStreamStore, SyncRedisStreamStore


class _Pipeline:
    def __init__(self, client: _FakeRedis | _FakeSyncRedis) -> None:
        self._client = client
        self._expirations: list[tuple[str, int]] = []

    def expire(self, key: str, ttl: int) -> None:
        self._expirations.append((key, ttl))

    def _commit(self) -> None:
        self._client.expirations.extend(self._expirations)

    async def execute(self) -> None:
        self._commit()


class _SyncPipeline(_Pipeline):
    def execute(self) -> None:
        self._commit()


class _FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.expirations: list[tuple[str, int]] = []
        self.streams: dict[str, list[tuple[str, dict[str, str]]]] = {}

    async def xadd(self, stream: str, fields: dict[str, str], **_: Any) -> str:
        event_id = f"{len(self.streams.get(stream, [])) + 1}-0"
        self.streams.setdefault(stream, []).append((event_id, fields))
        return event_id

    async def xread(self, streams: dict[str, str], **_: Any) -> list[tuple[str, list[tuple[str, dict[str, str]]]]]:
        stream, after = next(iter(streams.items()))
        entries = [entry for entry in self.streams.get(stream, []) if entry[0] > after]
        return [(stream, entries)] if entries else []

    async def set(self, key: str, value: str, **_: Any) -> None:
        self.values[key] = value

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def exists(self, key: str) -> int:
        return int(key in self.values)

    def pipeline(self, **_: Any) -> _Pipeline:
        return _Pipeline(self)


class _FakeSyncRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.expirations: list[tuple[str, int]] = []

    def set(self, key: str, value: str, *, nx: bool = False, **_: Any) -> bool | None:
        if nx and key in self.values:
            return None
        self.values[key] = value
        return True

    def delete(self, *keys: str) -> int:
        deleted = sum(key in self.values for key in keys)
        for key in keys:
            self.values.pop(key, None)
        return deleted

    def mget(self, keys: tuple[str, ...]) -> list[str | None]:
        return [self.values.get(key) for key in keys]

    def scan_iter(self, *, match: str) -> list[str]:
        prefix = match.removesuffix("*")
        return [key for key in self.values if key.startswith(prefix)]

    def pipeline(self, **_: Any) -> _SyncPipeline:
        return _SyncPipeline(self)


def test_stream_append_read_state_and_ttl() -> None:
    asyncio.run(_assert_stream_append_read_state_and_ttl())


async def _assert_stream_append_read_state_and_ttl() -> None:
    redis = _FakeRedis()
    store = RedisStreamStore(redis, terminal_ttl_seconds=60)  # type: ignore[arg-type]

    event_id = await store.append("generation:1:events", "content.delta", {"text": "你好"})
    await store.set_state("generation:1", {"status": "running", "text": "你好"})
    events = await store.read("generation:1:events", "0-0", block_ms=1)
    await store.finish("generation:1", "generation:1:events")

    assert event_id == "1-0"
    assert events[0].type == "content.delta"
    assert json.dumps(events[0].data, ensure_ascii=False) == '{"text": "你好"}'
    assert await store.get_state("generation:1") == {"status": "running", "text": "你好"}
    assert redis.expirations == [("generation:1", 60), ("generation:1:events", 60)]


def test_sync_initial_state_does_not_overwrite_a_worker_projection() -> None:
    redis = _FakeSyncRedis()
    store = SyncRedisStreamStore(redis)  # type: ignore[arg-type]

    assert store.set_state_if_absent("processing:1:state", {"status": "queued"})
    store.set_state("processing:1:state", {"status": "running"})
    assert not store.set_state_if_absent("processing:1:state", {"status": "queued"})

    assert json.loads(redis.values["processing:1:state"]) == {"status": "running"}


def test_sync_store_deletes_exact_keys_and_checkpoint_pattern() -> None:
    redis = _FakeSyncRedis()
    store = SyncRedisStreamStore(redis)  # type: ignore[arg-type]
    redis.values = {
        "generation:1": "{}",
        "generation:1:rag-checkpoint:a": "{}",
        "generation:1:rag-checkpoint:b": "{}",
        "generation:2:rag-checkpoint:a": "{}",
    }

    assert store.delete("generation:1") == 1
    assert store.delete_pattern("generation:1:rag-checkpoint:*") == 2
    assert set(redis.values) == {"generation:2:rag-checkpoint:a"}


def test_sync_store_loads_checkpoint_states_by_pattern_in_one_batch() -> None:
    redis = _FakeSyncRedis()
    store = SyncRedisStreamStore(redis)  # type: ignore[arg-type]
    redis.values = {
        "generation:1:rag-checkpoint:a": '{"node":"route"}',
        "generation:1:rag-checkpoint:b": '{"node":"retrieve"}',
        "generation:2:rag-checkpoint:a": '{"node":"route"}',
    }

    states = store.get_states_by_pattern("generation:1:rag-checkpoint:*")

    assert {value["node"] for value in states.values()} == {"route", "retrieve"}


def test_sync_store_expires_exact_keys_and_patterns() -> None:
    redis = _FakeSyncRedis()
    store = SyncRedisStreamStore(redis)  # type: ignore[arg-type]
    redis.values = {
        "generation:1": "{}",
        "generation:1:rag-checkpoint:a": "{}",
        "generation:1:rag-checkpoint:b": "{}",
    }

    store.finish("generation:1", "generation:1:events", ttl_seconds=300)
    store.expire_pattern("generation:1:rag-checkpoint:*", ttl_seconds=300)

    assert redis.expirations == [
        ("generation:1", 300),
        ("generation:1:events", 300),
        ("generation:1:rag-checkpoint:a", 300),
        ("generation:1:rag-checkpoint:b", 300),
    ]
