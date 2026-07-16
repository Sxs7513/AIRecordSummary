from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from audio_processing.stages.build_search_chunks.contracts import TopicSection
from audio_processing.stages.recording_models import SearchChunk, Utterance


class SearchChunkBuilder:
    """Apply topic boundaries and deterministic hard limits without duplicating overlap text."""

    def __init__(self, max_chars: int, max_duration_ms: int, max_utterances: int) -> None:
        self._max_chars = max_chars
        self._max_duration_ms = max_duration_ms
        self._max_utterances = max_utterances

    def build(self, utterances: list[Utterance], sections: list[TopicSection] | None) -> list[SearchChunk]:
        chunks: list[SearchChunk] = []
        if sections is None:
            self._append_bounded(chunks, utterances, None, None, "deterministic_fallback")
            return chunks
        by_index = {item.utterance_index: item for item in utterances}
        for section_index, section in enumerate(sections):
            members = [by_index[index] for index in range(section.start_utterance_index, section.end_utterance_index + 1)]
            self._append_bounded(chunks, members, section.topic.strip(), section_index, "topic_boundary")
        return chunks

    def _append_bounded(
        self,
        chunks: list[SearchChunk],
        utterances: Sequence[Utterance],
        topic: str | None,
        section_index: int | None,
        method: Literal["topic_boundary", "deterministic_fallback"],
    ) -> None:
        pending: list[Utterance] = []
        for utterance in utterances:
            if pending and self._should_flush(pending, utterance):
                chunks.append(self._to_chunk(len(chunks), pending, topic, section_index, method))
                pending = []
            pending.append(utterance)
        if pending:
            chunks.append(self._to_chunk(len(chunks), pending, topic, section_index, method))

    def _should_flush(self, current: list[Utterance], next_utterance: Utterance) -> bool:
        text_length = sum(len(item.text) for item in current) + len(next_utterance.text) + len(current)
        duration_ms = next_utterance.end_ms - current[0].start_ms
        return text_length > self._max_chars or duration_ms > self._max_duration_ms or len(current) >= self._max_utterances

    @staticmethod
    def _to_chunk(
        chunk_index: int,
        utterances: list[Utterance],
        topic: str | None,
        topic_section_index: int | None,
        build_method: Literal["topic_boundary", "deterministic_fallback"],
    ) -> SearchChunk:
        return SearchChunk(
            chunk_index=chunk_index,
            text="\n".join(f"{item.speaker_label}: {item.text}" for item in utterances),
            start_ms=utterances[0].start_ms,
            end_ms=utterances[-1].end_ms,
            speaker_labels=list(dict.fromkeys(item.speaker_label for item in utterances)),
            speaker_cluster_ids=list(dict.fromkeys(item.speaker_cluster_id for item in utterances)),
            source_utterance_indexes=[item.utterance_index for item in utterances],
            source_diarization_segment_ids=[source_id for item in utterances for source_id in item.source_diarization_segment_ids],
            topic=topic,
            topic_section_index=topic_section_index,
            build_method=build_method,
        )
