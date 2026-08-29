from __future__ import annotations

import asyncio

import pytest

import l1_foundation.messaging.kafka as kafka_module
from l1_foundation.messaging.kafka import SyncKafkaEventConsumer


class _StartupFailure(RuntimeError):
    pass


class _FailingConsumer:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    async def start(self) -> None:
        raise _StartupFailure("original startup failure")

    async def stop(self) -> None:
        return None


class _LoopBoundConsumer:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.loop = asyncio.get_running_loop()

    async def start(self) -> None:
        return None

    async def run(self, _handler: object) -> None:
        await asyncio.Event().wait()

    async def wait_for_assignment(self) -> None:
        return None

    async def stop(self) -> None:
        return None


def test_sync_consumer_preserves_original_startup_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kafka_module, "KafkaEventConsumer", _FailingConsumer)
    consumer = SyncKafkaEventConsumer(
        ["compute.results"],
        bootstrap_servers="unused",
        group_id="test-group",
        client_id="test-client",
    )

    with pytest.raises(_StartupFailure, match="original startup failure"):
        consumer.start(lambda _event: None)

    assert consumer._thread is None
    assert consumer._loop is None


def test_sync_consumer_constructs_async_consumer_inside_running_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kafka_module, "KafkaEventConsumer", _LoopBoundConsumer)
    consumer = SyncKafkaEventConsumer(
        ["compute.results"],
        bootstrap_servers="unused",
        group_id="test-group",
        client_id="test-client",
    )

    consumer.start(lambda _event: None)
    consumer.stop()
