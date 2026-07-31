from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from typing import Protocol
from uuid import UUID

from pydantic import BaseModel

from l1_foundation.worker.contracts import ComputeCommand, ComputeEvent, ComputeTaskSnapshot


class WorkerClient(Protocol):
    """Async Kafka/Redis compute client contract."""

    async def __aenter__(self) -> WorkerClient: ...
    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None: ...
    async def close(self) -> None: ...
    async def ready(self) -> None: ...
    async def submit[InputT: BaseModel](self, command: ComputeCommand[InputT]) -> ComputeTaskSnapshot: ...
    async def status(self, task_id: UUID) -> ComputeTaskSnapshot: ...
    async def cancel(self, task_id: UUID) -> ComputeTaskSnapshot: ...
    def stream[InputT: BaseModel](self, command: ComputeCommand[InputT]) -> AsyncIterator[ComputeEvent]: ...
    async def execute[InputT: BaseModel, ResultT: BaseModel](
        self,
        command: ComputeCommand[InputT],
        *,
        result_type: type[ResultT],
        on_progress: Callable[[float, str | None], None] | None = None,
    ) -> ResultT: ...
    async def execute_streaming[InputT: BaseModel, ResultT: BaseModel](
        self,
        command: ComputeCommand[InputT],
        *,
        result_type: type[ResultT],
        on_progress: Callable[[float, str | None], None] | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> ResultT: ...


class SyncWorkerClient(Protocol):
    """Blocking Kafka/Redis compute client contract used by thread-owned stages."""

    def close(self) -> None: ...
    def ready(self) -> None: ...
    def submit[InputT: BaseModel](self, command: ComputeCommand[InputT]) -> ComputeTaskSnapshot: ...
    def status(self, task_id: UUID) -> ComputeTaskSnapshot: ...
    def cancel(self, task_id: UUID) -> ComputeTaskSnapshot: ...
    def stream[InputT: BaseModel](self, command: ComputeCommand[InputT]) -> Iterator[ComputeEvent]: ...
    def execute[InputT: BaseModel, ResultT: BaseModel](
        self,
        command: ComputeCommand[InputT],
        *,
        result_type: type[ResultT],
        on_progress: Callable[[float, str | None], None] | None = None,
    ) -> ResultT: ...
    def execute_streaming[InputT: BaseModel, ResultT: BaseModel](
        self,
        command: ComputeCommand[InputT],
        *,
        result_type: type[ResultT],
        on_progress: Callable[[float, str | None], None] | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> ResultT: ...
