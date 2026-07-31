from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Thread
from typing import TypeVar, cast

from l1_foundation.task_runtime.resources import ResourceQueue

logger = logging.getLogger("worker")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class _Execution[ExecutionResultT]:
    work: Callable[[], ExecutionResultT]
    future: Future[ExecutionResultT]


class ComputeExecutionPool:
    """Worker-owned CPU/GPU execution queues with GPU priority admission."""

    def __init__(self) -> None:
        self._cpu_queue: Queue[_Execution[object]] = Queue()
        self._io_queue: Queue[_Execution[object]] = Queue()
        self._gpu_high_queue: Queue[_Execution[object]] = Queue()
        self._gpu_normal_queue: Queue[_Execution[object]] = Queue()
        self._stop_event = Event()
        self._threads: tuple[Thread, ...] = ()

    def start(self) -> None:
        if self._threads:
            return
        self._threads = (
            Thread(target=self._run_cpu, name="compute-worker-cpu", daemon=True),
            *(Thread(target=self._run_io, name=f"compute-worker-io-{index}", daemon=True) for index in range(4)),
            Thread(target=self._run_gpu, name="compute-worker-gpu", daemon=True),
        )
        for thread in self._threads:
            thread.start()

    async def submit(self, resource_queue: ResourceQueue, work: Callable[[], ResultT]) -> ResultT:
        if not self._threads or self._stop_event.is_set():
            raise RuntimeError("Compute execution pool is not running")
        future: Future[ResultT] = Future()
        execution = cast(_Execution[object], _Execution(work, future))
        if resource_queue == ResourceQueue.IO:
            self._io_queue.put(execution)
        elif resource_queue == ResourceQueue.CPU:
            self._cpu_queue.put(execution)
        elif resource_queue == ResourceQueue.GPU_HIGH:
            self._gpu_high_queue.put(execution)
        elif resource_queue == ResourceQueue.GPU_NORMAL:
            self._gpu_normal_queue.put(execution)
        else:
            raise ValueError(f"Unsupported compute queue: {resource_queue}")
        return await asyncio.wrap_future(future)

    def stop(self, timeout_seconds: float = 5.0) -> None:
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout_seconds)
        active = [thread.name for thread in self._threads if thread.is_alive()]
        if active:
            logger.warning("Compute execution threads exceeded shutdown grace period: %s", ", ".join(active))
        self._threads = ()

    def _run_cpu(self) -> None:
        self._run_queue(self._cpu_queue)

    def _run_io(self) -> None:
        self._run_queue(self._io_queue)

    def _run_gpu(self) -> None:
        while not self._stop_event.is_set():
            try:
                execution = self._gpu_high_queue.get_nowait()
            except Empty:
                try:
                    execution = self._gpu_normal_queue.get(timeout=0.2)
                except Empty:
                    continue
            self._execute(execution)

    def _run_queue(self, queue: Queue[_Execution[object]]) -> None:
        while not self._stop_event.is_set():
            try:
                execution = queue.get(timeout=0.2)
            except Empty:
                continue
            self._execute(execution)

    @staticmethod
    def _execute(execution: _Execution[object]) -> None:
        if not execution.future.set_running_or_notify_cancel():
            return
        try:
            result = execution.work()
        except BaseException as error:
            execution.future.set_exception(error)
        else:
            execution.future.set_result(result)
