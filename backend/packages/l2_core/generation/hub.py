from __future__ import annotations

from queue import Empty, Queue
from threading import Lock
from uuid import UUID

from l2_core.generation.contracts import GenerationEvent


class GenerationSubscription:
    """A synchronous queue that can safely receive events from worker threads."""

    def __init__(self, queue: Queue[GenerationEvent | object]) -> None:
        self._queue = queue

    def get(self, timeout_seconds: float) -> GenerationEvent | None:
        try:
            item = self._queue.get(timeout=timeout_seconds)
        except Empty:
            return None
        return item if isinstance(item, GenerationEvent) else None

    def publish(self, event: GenerationEvent) -> None:
        self._queue.put(event)


class GenerationStreamHub:
    """In-process live fan-out; PostgreSQL remains the durable stream source."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._subscribers: dict[UUID, set[GenerationSubscription]] = {}

    def subscribe(self, run_id: UUID) -> GenerationSubscription:
        subscription = GenerationSubscription(Queue())
        with self._lock:
            self._subscribers.setdefault(run_id, set()).add(subscription)
        return subscription

    def unsubscribe(self, run_id: UUID, subscription: GenerationSubscription) -> None:
        with self._lock:
            queues = self._subscribers.get(run_id)
            if queues is None:
                return
            queues.discard(subscription)
            if not queues:
                self._subscribers.pop(run_id, None)

    def publish(self, event: GenerationEvent) -> None:
        with self._lock:
            queues = tuple(self._subscribers.get(event.run_id, ()))
        for subscription in queues:
            subscription.publish(event)
