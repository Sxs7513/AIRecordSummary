from __future__ import annotations

import asyncio
from collections.abc import Callable

from l1_foundation.llm.worker_handler import LlmGenerateInput, LlmGenerateResult
from l1_foundation.observability.context import InvocationHandle, finish_invocation, start_invocation
from l1_foundation.worker.client import WorkerClient
from l1_foundation.worker.contracts import ComputeCommand


class InstrumentedModelClient:
    """LLM-specific facade over WorkerClient with transparent invocation telemetry."""

    def __init__(self, worker_client: WorkerClient) -> None:
        self._worker_client = worker_client

    async def execute(
        self,
        command: ComputeCommand[LlmGenerateInput],
        *,
        result_type: type[LlmGenerateResult] = LlmGenerateResult,
    ) -> LlmGenerateResult:
        started = self._started_record(command)
        try:
            result = await self._worker_client.execute(command, result_type=result_type)
        except BaseException as error:
            self._finish_error(started, error)
            raise
        self._finish_success(started, result)
        return result

    async def execute_streaming(
        self,
        command: ComputeCommand[LlmGenerateInput],
        *,
        result_type: type[LlmGenerateResult] = LlmGenerateResult,
        on_delta: Callable[[str], None] | None = None,
    ) -> LlmGenerateResult:
        started = self._started_record(command)
        try:
            result = await self._worker_client.execute_streaming(command, result_type=result_type, on_delta=on_delta)
        except BaseException as error:
            self._finish_error(started, error)
            raise
        self._finish_success(started, result)
        return result

    @staticmethod
    def _started_record(command: ComputeCommand[LlmGenerateInput]) -> InvocationHandle | None:
        return start_invocation(
            provider=command.input.provider.value,
            invocation_id=command.task_id,
            stream=command.input.stream,
        )

    @staticmethod
    def _finish_success(started: InvocationHandle | None, result: LlmGenerateResult) -> None:
        if started is None:
            return
        usage_source = (
            "local_tokenizer"
            if result.provider.value == "local" and (result.prompt_tokens is not None or result.completion_tokens is not None)
            else "provider"
            if result.prompt_tokens is not None or result.completion_tokens is not None
            else "unavailable"
        )
        finish_invocation(
            started,
            "succeeded",
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            usage_source=usage_source,
            finish_reason=result.finish_reason,
            provider_request_id=result.request_id,
        )

    @staticmethod
    def _finish_error(started: InvocationHandle | None, error: BaseException) -> None:
        if started is None:
            return
        status = "cancelled" if isinstance(error, asyncio.CancelledError) else "failed"
        finish_invocation(started, status, error_type=type(error).__name__)
