from __future__ import annotations

import asyncio
from typing import cast
from uuid import uuid4

from l1_foundation.llm import ChatMessage, ChatRole, CompletionOptions, LlmGenerateResult, LlmProvider, build_llm_generate_command
from l1_foundation.llm.worker_handler import LlmGenerateInput
from l1_foundation.messaging import EventEnvelope, Topics
from l1_foundation.observability import (
    InstrumentedModelClient,
    ObservabilityClient,
    ObservabilityScope,
    finish_span,
    observation_scope,
    start_span,
)
from l1_foundation.worker import WorkerClient
from l1_foundation.worker.contracts import ComputeCommand


class _CapturingProducer:
    def __init__(self) -> None:
        self.delivered: list[tuple[str, str, EventEnvelope]] = []
        self.started = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.started = False

    async def publish(self, topic: str, key: str, event: EventEnvelope) -> None:
        self.delivered.append((topic, key, event))


class _FakeWorker:
    async def execute(
        self,
        _command: ComputeCommand[LlmGenerateInput],
        *,
        result_type: type[LlmGenerateResult],
    ) -> LlmGenerateResult:
        return result_type(
            text="answer",
            provider=LlmProvider.GEMINI,
            model="gemini-test",
            prompt_tokens=12,
            completion_tokens=3,
        )


def test_instrumented_model_client_publishes_running_and_terminal_records() -> None:
    producer = _CapturingProducer()

    async def scenario() -> LlmGenerateResult:
        worker = cast(WorkerClient, _FakeWorker())
        telemetry = ObservabilityClient(producer=producer)
        await telemetry.start()
        try:
            with observation_scope(telemetry, ObservabilityScope(workspace_id=uuid4(), generation_run_id=uuid4())):
                span = start_span("answer")
                result = await InstrumentedModelClient(worker).execute(
                    build_llm_generate_command(
                        LlmProvider.GEMINI,
                        [ChatMessage(ChatRole.USER, "question")],
                        CompletionOptions(max_tokens=20),
                        context_size=1024,
                        stream=False,
                    )
                )
                finish_span(span, "succeeded", metadata={"evidence_count": 2})
            return result
        finally:
            await telemetry.close()

    result = asyncio.run(scenario())

    assert result.text == "answer"
    assert [topic for topic, _, _ in producer.delivered] == [
        Topics.RAG_EXECUTION_EVENTS,
        Topics.MODEL_INVOCATION_EVENTS,
        Topics.MODEL_INVOCATION_EVENTS,
        Topics.RAG_EXECUTION_EVENTS,
    ]
    assert [event.payload["status"] for _, _, event in producer.delivered] == ["running", "running", "succeeded", "succeeded"]
    terminal_invocation = producer.delivered[2][2].payload
    assert terminal_invocation["operation"] == "answer"
    assert terminal_invocation["prompt_tokens"] == 12
    assert terminal_invocation["completion_tokens"] == 3
    assert terminal_invocation["usage_source"] == "provider"


def test_disabled_observability_client_does_not_start_producer() -> None:
    producer = _CapturingProducer()
    client = ObservabilityClient(enabled=False, producer=producer)

    async def scenario() -> None:
        await client.start()
        await client.close()

    asyncio.run(scenario())
    assert not producer.started
    assert producer.delivered == []
