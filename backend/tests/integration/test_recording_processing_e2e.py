from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Literal
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from sqlalchemy import Connection, Engine, create_engine, text

from application.recording_processing import StartRecordingProcessing
from audio_processing.contracts import RecordingId
from audio_processing.hooks import RecordingProcessingHooks
from audio_processing.registry import build_recording_stage_registry
from audio_processing.stages.align_transcript import AlignTranscriptStage
from audio_processing.stages.build_utterances import BuildUtterancesStage
from audio_processing.stages.correct_text import CorrectAsrWindowsStage, LocalTextCorrector
from audio_processing.stages.diarize_pyannote import PyannoteDiarizeStage
from audio_processing.stages.normalize_audio import NormalizeAudioStage
from audio_processing.stages.preprocess_asr_audio import PreprocessAsrAudioStage
from audio_processing.stages.recording_models import (
    AlignTranscriptInput,
    BuildUtterancesInput,
    CorrectAsrWindowsInput,
    DiarizeInput,
    GenerateSummaryInput,
    NormalizeAudioInput,
    PreprocessAsrAudioInput,
    TranscribeQwenAsrInput,
)
from audio_processing.stages.summary.stage import GenerateSummaryStage
from audio_processing.stages.transcribe_qwen_asr import QwenAsrTranscribeStage
from pipeline.contracts import ArtifactPayload, ArtifactRef, PipelineRunId, PipelineSubjectId, StageContext, StageRunId
from pipeline.runtime.artifact_store import ArtifactStore
from pipeline.runtime.coordinator import PipelineCoordinator
from pipeline.runtime.executor import PipelineExecutor
from pipeline.runtime.repository import PipelineRepository
from scripts.initialize_database import initialize_database
from settings import REPOSITORY_ROOT, Settings, get_settings
from task_runtime.scheduler import ResourceScheduler

pytestmark = pytest.mark.integration

logger = logging.getLogger(__name__)


@pytest.mark.skipif(os.environ.get("RUN_AUDIO_E2E") != "1", reason="set RUN_AUDIO_E2E=1 to run local model integration")
def test_real_audio_stage_chain(tmp_path: Path) -> None:
    """Run normalize → diarize → Qwen ASR → correction → utterances → summary on a supplied short recording."""
    source = Path(os.environ["AUDIO_E2E_FILE"]).resolve()
    assert source.is_file(), f"AUDIO_E2E_FILE does not exist: {source}"
    settings = get_settings()
    storage_root = tmp_path / "storage"
    source_path = storage_root / "source" / source.name
    source_path.parent.mkdir(parents=True)
    shutil.copy2(source, source_path)
    artifact_store = ArtifactStore(storage_root)
    context = StageContext(PipelineSubjectId(uuid4()), PipelineRunId(uuid4()), StageRunId(uuid4()), attempt_count=1)

    source_ref = ArtifactRef(artifact_type="audio.source", artifact_version="1", uri=source_path.relative_to(storage_root).as_posix())
    normalized = asyncio.run(NormalizeAudioStage(storage_root).run(context, NormalizeAudioInput(source_audio=source_ref)))
    normalized_ref = _persist(artifact_store, context, "normalize_audio", normalized.artifacts[0])

    diarized = asyncio.run(
        PyannoteDiarizeStage(storage_root, artifact_store, settings.pyannote_model, settings.pyannote_auth_token).run(
            context, DiarizeInput(audio=normalized_ref)
        )
    )
    assert diarized.output.segments
    diarization_ref = _persist(artifact_store, context, "diarize_pyannote", diarized.artifacts[0])

    preprocessed = asyncio.run(
        PreprocessAsrAudioStage(storage_root, artifact_store, settings.asr_preprocess_recording_enabled).run(
            context, PreprocessAsrAudioInput(audio=normalized_ref)
        )
    )
    preprocessed_ref = _persist(artifact_store, context, "preprocess_asr_audio", preprocessed.artifacts[0])

    transcript = asyncio.run(
        QwenAsrTranscribeStage(
            storage_root,
            artifact_store,
            settings.qwen_asr_model,
            settings.qwen_asr_language,
            settings.resolved_huggingface_hub_cache_dir,
        ).run(context, TranscribeQwenAsrInput(audio=preprocessed_ref, diarization=diarization_ref))
    )
    assert transcript.output.windows
    transcript_ref = _persist(artifact_store, context, "transcribe_qwen_asr", transcript.artifacts[0])

    corrected = asyncio.run(
        CorrectAsrWindowsStage(
            artifact_store,
            LocalTextCorrector(
                REPOSITORY_ROOT,
                settings.transcription_correction_enabled,
                settings.llm_correction_enabled,
                settings.llm_correction_model_repo,
                settings.llm_correction_model_file,
                settings.llm_correction_context_size,
                settings.resolved_whisper_initial_prompt_config,
            ),
            "pycorrector_llm" if settings.llm_correction_enabled else "pycorrector" if settings.transcription_correction_enabled else "rules",
            settings.llm_correction_model_repo if settings.llm_correction_enabled else None,
        ).run(context, CorrectAsrWindowsInput(transcript=transcript_ref))
    )
    assert corrected.output.windows
    corrected_ref = _persist(artifact_store, context, "correct_asr_windows", corrected.artifacts[0])
    aligned = asyncio.run(
        AlignTranscriptStage(
            storage_root,
            artifact_store,
            settings.transcript_alignment_model,
            settings.resolved_huggingface_hub_cache_dir,
        ).run(
            context,
            AlignTranscriptInput(audio=preprocessed_ref, diarization=diarization_ref, transcript=corrected_ref),
        )
    )
    aligned_ref = _persist(artifact_store, context, "align_transcript", aligned.artifacts[0])
    utterances = asyncio.run(BuildUtterancesStage(artifact_store).run(context, BuildUtterancesInput(transcript=aligned_ref)))
    assert utterances.output.segments
    final_ref = _persist(artifact_store, context, "build_utterances", utterances.artifacts[0])
    summary = asyncio.run(
        GenerateSummaryStage(
            artifact_store,
            settings.resolved_local_llm_model_path,
            settings.recording_summary_context_size,
            settings.resolved_recording_summary_prompt_config,
        ).run(context, GenerateSummaryInput(utterances=final_ref))
    )
    assert summary.output.summary_text.strip()


@pytest.mark.skipif(os.environ.get("RUN_PIPELINE_E2E") != "1", reason="set RUN_PIPELINE_E2E=1 to run the database-backed pipeline E2E")
def test_real_audio_recording_processing_pipeline_e2e(tmp_path: Path) -> None:
    """Run the declared graph through create_run, queue workers, artifact bindings, and projections."""
    source = Path(os.environ["AUDIO_E2E_FILE"]).resolve()
    assert source.is_file(), f"AUDIO_E2E_FILE does not exist: {source}"
    database_name = _test_database_name()
    settings = get_settings().model_copy(update={"database_url": None, "db_name": database_name, "local_storage_root": tmp_path / "storage"})
    engine: Engine | None = None
    try:
        logger.info("initializing temporary E2E database: %s", database_name)
        initialize_database(settings)
        storage_root = settings.resolved_local_storage_root
        source_path = storage_root / "source" / source.name
        source_path.parent.mkdir(parents=True)
        shutil.copy2(source, source_path)
        engine = create_engine(settings.sqlalchemy_database_url, pool_pre_ping=True)
        repository = PipelineRepository(engine)
        recording_id = RecordingId(uuid4())
        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    insert into recordings (id, title, file_name, storage_path, mime_type, file_size_bytes, status)
                    values (:id, :title, :file_name, :storage_path, 'audio/wav', :file_size_bytes, 'uploaded')
                    """
                ),
                {
                    "id": recording_id,
                    "title": "pipeline e2e test",
                    "file_name": source.name,
                    "storage_path": source_path.relative_to(storage_root).as_posix(),
                    "file_size_bytes": source_path.stat().st_size,
                },
            )
        source_audio = ArtifactRef(
            artifact_type="audio.source",
            artifact_version="1",
            uri=source_path.relative_to(storage_root).as_posix(),
            metadata={"file_name": source.name, "mime_type": "audio/wav"},
        )
        run_id = StartRecordingProcessing(repository).execute(recording_id, source_audio)
        logger.info("created recording_processing run: run_id=%s recording_id=%s", run_id, recording_id)
        artifact_store = ArtifactStore(storage_root)
        scheduler = ResourceScheduler()
        scheduler.start()
        executor = PipelineExecutor(
            repository,
            build_recording_stage_registry(settings, artifact_store),
            artifact_store,
        )
        coordinator = PipelineCoordinator(repository, scheduler, executor, RecordingProcessingHooks(engine))
        asyncio.run(_drain_recording_pipeline(coordinator))
        scheduler.stop()
        with engine.connect() as connection:
            run_status = connection.execute(text("select status from pipeline_runs where id = :run_id"), {"run_id": run_id}).scalar_one()
            stage_states = _stage_states(connection, run_id)
            logger.info("pipeline run finished: run_id=%s status=%s stages=%s", run_id, run_status, stage_states)
            assert run_status == "succeeded", f"pipeline stages: {stage_states}"
            assert _count_rows(connection, "transcriptions", recording_id) == 1
            assert _count_rows(connection, "transcription_segments", recording_id) > 0
            assert _count_rows(connection, "speaker_diarization_segments", recording_id) > 0
            assert _count_rows(connection, "utterance_segments", recording_id) > 0
            assert _count_rows(connection, "recording_search_chunks", recording_id) > 0
            assert _count_rows(connection, "recording_summaries", recording_id) == 1
    finally:
        if engine is not None:
            engine.dispose()
        _drop_test_database(settings)


def _count_rows(
    connection: Connection,
    table_name: Literal[
        "transcriptions",
        "transcription_segments",
        "speaker_diarization_segments",
        "utterance_segments",
        "recording_search_chunks",
        "recording_summaries",
    ],
    recording_id: RecordingId,
) -> int:
    result = connection.execute(text(f"select count(*) from {table_name} where recording_id = :recording_id"), {"recording_id": recording_id})
    return int(result.scalar_one())


async def _drain_recording_pipeline(coordinator: PipelineCoordinator) -> None:
    """E2E-only deterministic loop: recording workflow submits work and scheduler executes it."""
    for _ in range(10_000):
        changed = await coordinator.run_once()
        if not changed and coordinator.is_idle:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("recording pipeline did not become idle")


def _stage_states(connection: Connection, run_id: PipelineRunId) -> list[dict[str, str | int | None]]:
    rows = connection.execute(
        text(
            """
            select node_name, resource_queue, status, attempt_count, error_message
            from stage_runs
            where pipeline_run_id = :run_id
            order by created_at, id
            """
        ),
        {"run_id": run_id},
    ).mappings()
    return [
        {
            "node": str(row["node_name"]),
            "queue": str(row["resource_queue"]),
            "status": str(row["status"]),
            "attempt": int(row["attempt_count"]),
            "error": str(row["error_message"]) if row["error_message"] is not None else None,
        }
        for row in rows
    ]


def _test_database_name() -> str:
    prefix = os.environ.get("PIPELINE_E2E_DB_PREFIX", "ai_record_summary_e2e")
    if not prefix.replace("_", "").isalnum():
        raise ValueError("PIPELINE_E2E_DB_PREFIX may contain only letters, digits, and underscores")
    return f"{prefix}_{uuid4().hex[:12]}"


def _drop_test_database(settings: Settings) -> None:
    admin_url = settings.sqlalchemy_admin_database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    with psycopg.connect(admin_url, autocommit=True) as connection, connection.cursor() as cursor:
        cursor.execute(
            "select pg_terminate_backend(pid) from pg_stat_activity where datname = %s and pid <> pg_backend_pid()",
            (settings.db_name,),
        )
        cursor.execute(sql.SQL("drop database if exists {}").format(sql.Identifier(settings.db_name)))


def _persist(artifact_store: ArtifactStore, context: StageContext, stage_name: str, artifact: ArtifactPayload) -> ArtifactRef:
    return artifact_store.write_json(context.subject_id, context.pipeline_run_id, context.stage_run_id, stage_name, artifact)
