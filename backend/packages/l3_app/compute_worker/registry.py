from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from pydantic import BaseModel

from l1_foundation.task_runtime.resources import ResourceQueue
from l1_foundation.worker.contracts import JsonObject, WorkerExecutionContext

type OperationKey = tuple[str, str]


@dataclass(frozen=True, slots=True)
class ComputeOperationSpec[InputT: BaseModel, ResultT: BaseModel]:
    name: str
    version: str
    resource_queue: ResourceQueue
    input_type: type[InputT]
    result_type: type[ResultT]
    handler: Callable[[InputT, WorkerExecutionContext], ResultT]
    release: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Compute operation name must not be empty")
        if not self.version.strip():
            raise ValueError("Compute operation version must not be empty")


@dataclass(frozen=True, slots=True)
class RegisteredComputeOperation:
    name: str
    version: str
    resource_queue: ResourceQueue
    execute: Callable[[JsonObject, WorkerExecutionContext], JsonObject]
    release: Callable[[], None] | None


class ComputeOperationNotFoundError(LookupError):
    pass


class ComputeOperationConflictError(ValueError):
    pass


class ComputeOperationRegistry:
    """Static mapping from a versioned operation name to one typed handler."""

    def __init__(self) -> None:
        self._operations: dict[OperationKey, RegisteredComputeOperation] = {}

    def register[InputT: BaseModel, ResultT: BaseModel](self, spec: ComputeOperationSpec[InputT, ResultT]) -> None:
        key = (spec.name, spec.version)
        if key in self._operations:
            raise ComputeOperationConflictError(f"Compute operation is already registered: {spec.name}@{spec.version}")

        def execute(payload: JsonObject, context: WorkerExecutionContext) -> JsonObject:
            operation_input = spec.input_type.model_validate(payload)
            result = spec.handler(operation_input, context)
            return cast(JsonObject, result.model_dump(mode="json"))

        self._operations[key] = RegisteredComputeOperation(
            name=spec.name,
            version=spec.version,
            resource_queue=spec.resource_queue,
            execute=execute,
            release=spec.release,
        )

    def resolve(self, name: str, version: str) -> RegisteredComputeOperation:
        try:
            return self._operations[(name, version)]
        except KeyError as error:
            raise ComputeOperationNotFoundError(f"Unsupported compute operation: {name}@{version}") from error

    @property
    def operation_count(self) -> int:
        return len(self._operations)

    def release_all(self) -> None:
        for operation in self._operations.values():
            if operation.release is not None:
                operation.release()
