from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from l2_core.audio_processing.stages.build_search_chunks.contracts import TopicSection
from l2_core.audio_processing.stages.recording_models import SearchChunk, Utterance
from l2_core.rag.search_document import build_retrieval_text


@dataclass(frozen=True)
class _ChunkPiece:
    utterance: Utterance
    text: str


class SearchChunkBuilder:
    """Apply topic boundaries and deterministic hard limits without duplicating overlap text."""

    def __init__(self, token_counter: Callable[[str], int], max_tokens: int, max_duration_ms: int, max_utterances: int) -> None:
        self._token_counter = token_counter
        self._max_tokens = max_tokens
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
            self._append_bounded(
                chunks,
                members,
                section.topic.strip(),
                section_index,
                "topic_boundary",
                terms=list(dict.fromkeys(term.strip() for term in section.terms if term.strip())),
                search_context=section.search_context.strip() if section.search_context and section.search_context.strip() else None,
            )
        return chunks

    def _append_bounded(
        self,
        chunks: list[SearchChunk],
        utterances: Sequence[Utterance],
        topic: str | None,
        section_index: int | None,
        method: Literal["topic_boundary", "deterministic_fallback"],
        *,
        terms: list[str] | None = None,
        search_context: str | None = None,
    ) -> None:
        pending: list[_ChunkPiece] = []
        for utterance in utterances:
            for piece in self._sentence_pieces(utterance):
                if pending and self._should_flush(pending, piece, topic, terms or [], search_context):
                    chunks.append(
                        self._to_chunk(
                            len(chunks), pending, topic, section_index, method, terms or [], search_context
                        )
                    )
                    pending = []
                pending.append(piece)
        if pending:
            chunks.append(
                self._to_chunk(
                    len(chunks), pending, topic, section_index, method, terms or [], search_context
                )
            )

    def _should_flush(
        self,
        current: list[_ChunkPiece],
        next_piece: _ChunkPiece,
        topic: str | None,
        terms: list[str],
        search_context: str | None,
    ) -> bool:
        candidate = [*current, next_piece]
        retrieval_text = build_retrieval_text(self._render_text(candidate), topic, terms, search_context)
        duration_ms = next_piece.utterance.end_ms - current[0].utterance.start_ms
        utterance_count = len({piece.utterance.utterance_index for piece in candidate})
        return (
            self._token_counter(retrieval_text) > self._max_tokens
            or duration_ms > self._max_duration_ms
            or utterance_count > self._max_utterances
        )

    @staticmethod
    def _sentence_pieces(utterance: Utterance) -> list[_ChunkPiece]:
        boundaries: list[int] = []
        text = utterance.text
        for index, character in enumerate(text):
            is_decimal_point = (
                character == "."
                and index > 0
                and index + 1 < len(text)
                and text[index - 1].isdigit()
                and text[index + 1].isdigit()
            )
            if character == "。" or (character == "." and not is_decimal_point):
                boundaries.append(index + 1)
        if not boundaries or boundaries[-1] != len(text):
            boundaries.append(len(text))
        pieces: list[_ChunkPiece] = []
        start = 0
        for end in boundaries:
            value = text[start:end]
            if value.strip():
                pieces.append(_ChunkPiece(utterance=utterance, text=value))
            start = end
        return pieces

    @staticmethod
    def _render_text(pieces: Sequence[_ChunkPiece]) -> str:
        lines: list[str] = []
        current_utterance_index: int | None = None
        current_speaker = ""
        current_text = ""
        for piece in pieces:
            utterance = piece.utterance
            if current_utterance_index is not None and utterance.utterance_index != current_utterance_index:
                lines.append(f"{current_speaker}: {current_text.strip()}")
                current_text = ""
            if utterance.utterance_index != current_utterance_index:
                current_utterance_index = utterance.utterance_index
                current_speaker = utterance.speaker_label
            current_text += piece.text
        if current_utterance_index is not None:
            lines.append(f"{current_speaker}: {current_text.strip()}")
        return "\n".join(lines)

    @staticmethod
    def _to_chunk(
        chunk_index: int,
        pieces: list[_ChunkPiece],
        topic: str | None,
        topic_section_index: int | None,
        build_method: Literal["topic_boundary", "deterministic_fallback"],
        terms: list[str],
        search_context: str | None,
    ) -> SearchChunk:
        utterances = list(dict.fromkeys(piece.utterance.utterance_index for piece in pieces))
        source_utterances = {piece.utterance.utterance_index: piece.utterance for piece in pieces}
        ordered_utterances = [source_utterances[index] for index in utterances]
        return SearchChunk(
            chunk_index=chunk_index,
            text=SearchChunkBuilder._render_text(pieces),
            start_ms=ordered_utterances[0].start_ms,
            end_ms=ordered_utterances[-1].end_ms,
            speaker_labels=list(dict.fromkeys(item.speaker_label for item in ordered_utterances)),
            speaker_cluster_ids=list(dict.fromkeys(item.speaker_cluster_id for item in ordered_utterances)),
            source_utterance_indexes=utterances,
            source_diarization_segment_ids=list(
                dict.fromkeys(source_id for item in ordered_utterances for source_id in item.source_diarization_segment_ids)
            ),
            topic=topic,
            terms=terms,
            search_context=search_context,
            topic_section_index=topic_section_index,
            build_method=build_method,
        )
