from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Thread
from typing import TypeVar, cast

from task_runtime.resources import ResourceQueue

logger = logging.getLogger(__name__)
ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class _ScheduledWork[WorkResultT]:
    resource_queue: ResourceQueue
    work: Callable[[], WorkResultT | Awaitable[WorkResultT]]
    future: Future[WorkResultT]


class ResourceScheduler:
    """One-process CPU/GPU admission controller with no business operation registry."""

    def __init__(self) -> None:
        self._cpu_queue: Queue[_ScheduledWork[object]] = Queue()
        self._gpu_high_queue: Queue[_ScheduledWork[object]] = Queue()
        self._gpu_normal_queue: Queue[_ScheduledWork[object]] = Queue()
        self._stop_event = Event()
        self._threads: tuple[Thread, ...] = ()

    def start(self) -> None:
        if self._threads:
            return
        self._threads = (
            Thread(target=self._run_cpu, name="resource-runner-cpu", daemon=True),
            Thread(target=self._run_gpu, name="resource-runner-gpu", daemon=True),
        )
        for thread in self._threads:
            thread.start()

    async def submit(self, resource_queue: ResourceQueue, work: Callable[[], ResultT | Awaitable[ResultT]]) -> ResultT:
        return await asyncio.wrap_future(self.schedule(resource_queue, work))

    def schedule(self, resource_queue: ResourceQueue, work: Callable[[], ResultT | Awaitable[ResultT]]) -> Future[ResultT]:
        if not self._threads or self._stop_event.is_set():
            raise RuntimeError("Resource scheduler is not running")
        future: Future[ResultT] = Future()
        item = cast(_ScheduledWork[object], _ScheduledWork(resource_queue, work, future))
        if resource_queue == ResourceQueue.CPU:
            self._cpu_queue.put(item)
        elif resource_queue == ResourceQueue.GPU_HIGH:
            self._gpu_high_queue.put(item)
        elif resource_queue == ResourceQueue.GPU_NORMAL:
            self._gpu_normal_queue.put(item)
        else:
            raise ValueError(f"Unsupported resource queue: {resource_queue}")
        return future

    def stop(self, timeout_seconds: float = 5.0) -> None:
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout_seconds)
        active = [thread.name for thread in self._threads if thread.is_alive()]
        if active:
            logger.warning("resource runners exceeded shutdown grace period: %s", ", ".join(active))
        self._threads = ()

    def _run_cpu(self) -> None:
        self._run_queue(self._cpu_queue)

    def _run_gpu(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._gpu_high_queue.get_nowait()
            except Empty:
                try:
                    item = self._gpu_normal_queue.get(timeout=0.2)
                except Empty:
                    continue
            self._execute(item)

    def _run_queue(self, queue: Queue[_ScheduledWork[object]]) -> None:
        while not self._stop_event.is_set():
            try:
                item = queue.get(timeout=0.2)
            except Empty:
                continue
            self._execute(item)

    @staticmethod
    def _execute(item: _ScheduledWork[object]) -> None:
        if not item.future.set_running_or_notify_cancel():
            return
        try:
            value = item.work()
            result = asyncio.run(ResourceScheduler._resolve(value)) if inspect.isawaitable(value) else value
        except BaseException as error:
            item.future.set_exception(error)
        else:
            item.future.set_result(result)

    @staticmethod
    async def _resolve(value: Awaitable[object]) -> object:
        """Normalize a generic Awaitable into the coroutine required by asyncio.run."""
        return await value
