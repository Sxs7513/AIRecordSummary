from __future__ import annotations

import re
from collections.abc import Sequence

from l1_foundation.llm import (
    ChatMessage,
    ChatRole,
    CompletionOptions,
    LlmGenerateResult,
    LlmProvider,
    ResponseFormat,
    ResponseFormatType,
    build_llm_generate_command,
)
from l1_foundation.worker import SyncWorkerClient
from l2_core.audio_processing.stages.build_search_chunks.contracts import TopicSection, TopicSectionsOutput
from l2_core.audio_processing.stages.build_search_chunks.prompt import build_topic_boundary_prompt
from l2_core.audio_processing.stages.recording_models import Utterance


class TopicBoundaryDetector:
    """Detect continuous topic sections; invalid model output triggers deterministic fallback."""

    def __init__(
        self,
        worker_client: SyncWorkerClient,
        provider: LlmProvider,
        context_size: int,
        max_batch_chars: int = 3500,
    ) -> None:
        self._worker_client = worker_client
        self._provider = provider
        self._context_size = context_size
        self._max_batch_chars = max_batch_chars

    def detect(self, utterances: Sequence[Utterance]) -> list[TopicSection]:
        if not utterances:
            return []
        sections: list[TopicSection] = []
        for batch in self._batches(utterances):
            sections.extend(self._detect_batch(batch))
        self._validate(sections, utterances)
        return sections

    def _detect_batch(self, utterances: Sequence[Utterance]) -> list[TopicSection]:
        response = self._worker_client.execute(
            build_llm_generate_command(
                self._provider,
                [ChatMessage(ChatRole.USER, build_topic_boundary_prompt(utterances))],
                CompletionOptions(
                    max_tokens=max(384, min(3072, len(utterances) * 80)),
                temperature=0,
                response_format=ResponseFormat(
                    type=ResponseFormatType.JSON_SCHEMA,
                    json_schema=TopicSectionsOutput.model_json_schema(),
                    strict=False,
                ),
            ),
                context_size=self._context_size,
                stream=False,
            ),
            result_type=LlmGenerateResult,
        )
        parsed = TopicSectionsOutput.model_validate_json(self._json_object(response.text))
        self._validate(parsed.sections, utterances)
        return parsed.sections

    def _batches(self, utterances: Sequence[Utterance]) -> list[list[Utterance]]:
        batches: list[list[Utterance]] = []
        pending: list[Utterance] = []
        chars = 0
        for utterance in utterances:
            if pending and chars + len(utterance.text) > self._max_batch_chars:
                batches.append(pending)
                pending = []
                chars = 0
            pending.append(utterance)
            chars += len(utterance.text)
        if pending:
            batches.append(pending)
        return batches

    @staticmethod
    def _json_object(raw: str) -> str:
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
        if fenced is not None:
            return fenced.group(1)
        matched = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if matched is None:
            raise ValueError("Topic detector did not return a JSON object")
        return matched.group(0)

    @staticmethod
    def _validate(sections: Sequence[TopicSection], utterances: Sequence[Utterance]) -> None:
        expected = [item.utterance_index for item in utterances]
        covered: list[int] = []
        for section in sections:
            if not section.topic.strip() or section.end_utterance_index < section.start_utterance_index:
                raise ValueError("Invalid topic section")
            covered.extend(range(section.start_utterance_index, section.end_utterance_index + 1))
        if covered != expected:
            raise ValueError("Topic sections must cover every utterance exactly once")
