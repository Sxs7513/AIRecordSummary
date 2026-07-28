from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any, cast
from uuid import UUID

from sqlalchemy import Engine, text

from l2_core.audio_processing.contracts import RecordingId
from l2_core.audio_processing.stages.recording_models import (
    DiarizationOutput,
    EmbeddingIndexingOutput,
    RecordingSummaryOutput,
    TranscriptOutput,
    UtterancesOutput,
)
from l2_core.rag.normalization import normalize_search_text


class RecordingProjectionService:
    """Idempotently materialize pipeline artifacts into the existing recording read models."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def project(self, recording_id: RecordingId, stage_name: str, output: object) -> None:
        if stage_name == "diarize_pyannote":
            self._project_diarization(recording_id, DiarizationOutput.model_validate(output))
        elif stage_name == "align_transcript":
            self._project_transcript(recording_id, TranscriptOutput.model_validate(output))
        elif stage_name == "build_utterances":
            self._project_utterances(recording_id, UtterancesOutput.model_validate(output))
        elif stage_name == "embedding_indexing":
            self._project_embedding_index(recording_id, EmbeddingIndexingOutput.model_validate(output))
        elif stage_name == "generate_summary":
            self._project_summary(recording_id, RecordingSummaryOutput.model_validate(output))

    def _project_diarization(self, recording_id: RecordingId, output: DiarizationOutput) -> None:
        with self._engine.begin() as connection:
            connection.execute(text("delete from speaker_diarization_segments where recording_id = :recording_id"), {"recording_id": recording_id})
            for segment in output.segments:
                connection.execute(
                    text(
                        """
                        insert into speaker_diarization_segments (
                            recording_id, speaker_cluster_id, speaker_label, start_ms, end_ms, confidence
                        ) values (:recording_id, :speaker_cluster_id, :speaker_label, :start_ms, :end_ms, :confidence)
                        """
                    ),
                    {
                        "recording_id": recording_id,
                        "speaker_cluster_id": segment.speaker_cluster_id,
                        "speaker_label": segment.speaker_label,
                        "start_ms": segment.start_ms,
                        "end_ms": segment.end_ms,
                        "confidence": segment.confidence,
                    },
                )

    def _project_transcript(self, recording_id: RecordingId, output: TranscriptOutput) -> None:
        with self._engine.begin() as connection:
            transcription_id = cast(
                UUID,
                connection.execute(
                    text(
                        """
                        insert into transcriptions (recording_id, language, model_name, full_text, segment_count)
                        values (:recording_id, :language, :model_name, :full_text, :segment_count)
                        on conflict (recording_id) do update set
                            language = excluded.language, model_name = excluded.model_name, full_text = excluded.full_text,
                            segment_count = excluded.segment_count, updated_at = now()
                        returning id
                        """
                    ),
                    {
                        "recording_id": recording_id,
                        "language": output.language,
                        "model_name": output.model_name,
                        "full_text": "".join(segment.text for segment in output.segments),
                        "segment_count": len(output.segments),
                    },
                ).scalar_one(),
            )
            connection.execute(text("delete from transcription_segments where transcription_id = :transcription_id"), {"transcription_id": transcription_id})
            connection.execute(text("delete from transcription_tokens where transcription_id = :transcription_id"), {"transcription_id": transcription_id})
            for index, segment in enumerate(output.segments):
                source_cluster_id, source_start_ms, source_end_ms = self._parse_diarization_source_id(segment.source_diarization_segment_id)
                diarization_id = connection.execute(
                    text(
                        """
                        select id from speaker_diarization_segments
                        where recording_id = :recording_id and speaker_cluster_id = :speaker_cluster_id
                          and start_ms = :start_ms and end_ms = :end_ms
                        order by created_at desc limit 1
                        """
                    ),
                    {
                        "recording_id": recording_id,
                        "speaker_cluster_id": source_cluster_id,
                        "start_ms": source_start_ms,
                        "end_ms": source_end_ms,
                    },
                ).scalar_one_or_none()
                connection.execute(
                    text(
                        """
                        insert into transcription_segments (
                            recording_id, transcription_id, segment_index, start_ms, end_ms, text,
                            speaker_label, speaker_cluster_id, diarization_segment_id
                        ) values (
                            :recording_id, :transcription_id, :segment_index, :start_ms, :end_ms, :text,
                            :speaker_label, :speaker_cluster_id, :diarization_segment_id
                        )
                        """
                    ),
                    {
                        "recording_id": recording_id,
                        "transcription_id": transcription_id,
                        "segment_index": index,
                        "start_ms": segment.start_ms,
                        "end_ms": segment.end_ms,
                        "text": segment.text,
                        "speaker_label": segment.speaker_label,
                        "speaker_cluster_id": segment.speaker_cluster_id,
                        "diarization_segment_id": diarization_id,
                    },
                )
            for token in output.alignment_tokens or []:
                connection.execute(
                    text(
                        """
                        insert into transcription_tokens (
                            recording_id, transcription_id, token_index, source_window_index, text, start_ms, end_ms,
                            speaker_cluster_id, speaker_label, attribution_status
                        ) values (
                            :recording_id, :transcription_id, :token_index, :source_window_index, :text, :start_ms, :end_ms,
                            :speaker_cluster_id, :speaker_label, :attribution_status
                        )
                        """
                    ),
                    {"recording_id": recording_id, "transcription_id": transcription_id, **token.model_dump()},
                )

    def _project_utterances(self, recording_id: RecordingId, output: UtterancesOutput) -> None:
        with self._engine.begin() as connection:
            connection.execute(text("delete from utterance_segments where recording_id = :recording_id"), {"recording_id": recording_id})
            for utterance in output.segments:
                source_ids = self._transcription_segment_ids(connection, recording_id, utterance.source_diarization_segment_ids)
                connection.execute(
                    text(
                        """
                        insert into utterance_segments (
                            recording_id, utterance_index, start_ms, end_ms, text, speaker_label, speaker_cluster_id,
                            source_transcription_segment_ids
                        ) values (
                            :recording_id, :utterance_index, :start_ms, :end_ms, :text, :speaker_label, :speaker_cluster_id,
                            :source_transcription_segment_ids
                        )
                        """
                    ),
                    {
                        "recording_id": recording_id,
                        "utterance_index": utterance.utterance_index,
                        "start_ms": utterance.start_ms,
                        "end_ms": utterance.end_ms,
                        "text": utterance.text,
                        "speaker_label": utterance.speaker_label,
                        "speaker_cluster_id": utterance.speaker_cluster_id,
                        "source_transcription_segment_ids": source_ids,
                    },
                )

    def _project_summary(self, recording_id: RecordingId, output: RecordingSummaryOutput) -> None:
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    insert into recording_summaries (recording_id, provider, model_name, summary_text)
                    values (:recording_id, :provider, :model_name, :summary_text)
                    on conflict (recording_id) do update set
                        provider = excluded.provider, model_name = excluded.model_name,
                        summary_text = excluded.summary_text, updated_at = now()
                    """
                ),
                {
                    "recording_id": recording_id,
                    "provider": output.provider,
                    "model_name": output.model_name,
                    "summary_text": output.summary_text,
                },
            )

    def _project_embedding_index(self, recording_id: RecordingId, output: EmbeddingIndexingOutput) -> None:
        if any(len(chunk.embedding) != output.dimensions for chunk in output.chunks):
            raise ValueError(f"Embedding dimensions do not match model metadata ({output.dimensions})")

        with self._engine.begin() as connection:
            embedding_model_id = cast(
                UUID,
                connection.execute(
                    text(
                        """
                        insert into embedding_models (
                            provider, model_name, dimensions, distance_metric, is_active
                        ) values (
                            :provider, :model_name, :dimensions, 'cosine', true
                        )
                        on conflict (provider, model_name, dimensions) do update set
                            distance_metric = excluded.distance_metric,
                            is_active = true
                        returning id
                        """
                    ),
                    {
                        "provider": output.provider,
                        "model_name": output.model_name,
                        "dimensions": output.dimensions,
                    },
                ).scalar_one(),
            )
            connection.execute(
                text("delete from recording_search_chunks where recording_id = :recording_id"),
                {"recording_id": recording_id},
            )
            for chunk in output.chunks:
                utterance_ids = self._utterance_segment_ids(connection, recording_id, chunk.source_utterance_indexes)
                transcription_ids = self._transcription_segment_ids(
                    connection,
                    recording_id,
                    chunk.source_diarization_segment_ids,
                )
                connection.execute(
                    text(
                        """
                        insert into recording_search_chunks (
                            recording_id, embedding_model_id, chunk_index, text, normalized_text,
                            start_ms, end_ms, speaker_labels, speaker_cluster_ids,
                            source_utterance_segment_ids, source_transcription_segment_ids,
                            is_target_person, matched_speaker_profile_ids, metadata, embedding
                        ) values (
                            :recording_id, :embedding_model_id, :chunk_index, :text, :normalized_text,
                            :start_ms, :end_ms, :speaker_labels, :speaker_cluster_ids,
                            :source_utterance_segment_ids, :source_transcription_segment_ids,
                            false, cast(:matched_speaker_profile_ids as uuid[]), cast(:metadata as jsonb),
                            cast(:embedding as halfvec)
                        )
                        """
                    ),
                    {
                        "recording_id": recording_id,
                        "embedding_model_id": embedding_model_id,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "normalized_text": normalize_search_text(chunk.text),
                        "start_ms": chunk.start_ms,
                        "end_ms": chunk.end_ms,
                        "speaker_labels": chunk.speaker_labels,
                        "speaker_cluster_ids": chunk.speaker_cluster_ids,
                        "source_utterance_segment_ids": utterance_ids,
                        "source_transcription_segment_ids": transcription_ids,
                        "matched_speaker_profile_ids": [],
                        "metadata": json.dumps(
                            {
                                "topic": chunk.topic,
                                "topic_section_index": chunk.topic_section_index,
                                "build_method": chunk.build_method,
                            },
                            ensure_ascii=False,
                        ),
                        "embedding": self._vector_literal(chunk.embedding),
                    },
                )

    @staticmethod
    def _utterance_segment_ids(connection: Any, recording_id: RecordingId, indexes: Sequence[int]) -> list[UUID]:
        if not indexes:
            return []
        rows = connection.execute(
            text(
                """
                select id
                from utterance_segments
                where recording_id = :recording_id
                  and utterance_index = any(cast(:utterance_indexes as integer[]))
                order by utterance_index
                """
            ),
            {"recording_id": recording_id, "utterance_indexes": list(indexes)},
        ).scalars()
        return [cast(UUID, row) for row in rows]

    @staticmethod
    def _transcription_segment_ids(connection: Any, recording_id: RecordingId, source_ids: Sequence[str]) -> list[UUID]:
        resolved: list[UUID] = []
        for source_id in source_ids:
            cluster_id, start_ms, end_ms = RecordingProjectionService._parse_diarization_source_id(source_id)
            row = connection.execute(
                text(
                    """
                    select transcription_segments.id
                    from transcription_segments
                    join speaker_diarization_segments on speaker_diarization_segments.id = transcription_segments.diarization_segment_id
                    where transcription_segments.recording_id = :recording_id
                      and speaker_diarization_segments.speaker_cluster_id = :speaker_cluster_id
                      and speaker_diarization_segments.start_ms = :start_ms
                      and speaker_diarization_segments.end_ms = :end_ms
                    order by transcription_segments.segment_index limit 1
                    """
                ),
                {"recording_id": recording_id, "speaker_cluster_id": cluster_id, "start_ms": start_ms, "end_ms": end_ms},
            ).scalar_one_or_none()
            if row is not None and row not in resolved:
                resolved.append(cast(UUID, row))
        return resolved

    @staticmethod
    def _vector_literal(values: Sequence[float]) -> str:
        return f"[{','.join(str(value) if math.isfinite(value) else '0' for value in values)}]"

    @staticmethod
    def _parse_diarization_source_id(source_id: str) -> tuple[str, int, int]:
        cluster_id, start, end = source_id.rsplit(":", maxsplit=2)
        return cluster_id, int(start), int(end)
