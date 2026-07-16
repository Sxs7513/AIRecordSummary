from __future__ import annotations

from typing import cast
from uuid import uuid4

from sqlalchemy import create_engine, text

from audio_processing.stages.summary.regeneration import RecordingSummaryRegenerationService
from audio_processing.stages.summary.stage import GenerateSummaryStage
from generation.service import GenerationService
from task_runtime.scheduler import ResourceScheduler


def test_summary_regeneration_loads_only_materialized_utterance_columns() -> None:
    engine = create_engine("sqlite://")
    recording_id = uuid4()
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                create table utterance_segments (
                    recording_id text,
                    utterance_index integer,
                    start_ms integer,
                    end_ms integer,
                    text text,
                    speaker_cluster_id text,
                    speaker_label text,
                    source_transcription_segment_ids text
                )
                """
            )
        )
        connection.execute(
            text(
                """
                insert into utterance_segments (
                    recording_id, utterance_index, start_ms, end_ms, text, speaker_cluster_id, speaker_label, source_transcription_segment_ids
                ) values (:recording_id, 0, 100, 200, '测试文本', null, null, 'legacy value')
                """
            ),
            {"recording_id": str(recording_id)},
        )
    service = RecordingSummaryRegenerationService(
        engine,
        cast(ResourceScheduler, object()),
        cast(GenerationService, object()),
        cast(GenerateSummaryStage, object()),
    )

    utterances = service.load_utterances(recording_id)

    assert len(utterances) == 1
    assert utterances[0].text == "测试文本"
    assert utterances[0].speaker_cluster_id == "unknown"
    assert utterances[0].speaker_label == "Unknown Speaker"
    assert utterances[0].source_diarization_segment_ids == []
