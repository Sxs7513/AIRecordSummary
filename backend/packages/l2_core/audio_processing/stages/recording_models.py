from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from l1_foundation.pipeline.contracts import ArtifactRef
from l2_core.rag.search_document import build_retrieval_text

type AsrProvider = Literal["qwen_asr", "funasr_nano"]


class NormalizeAudioInput(BaseModel):
    source_audio: ArtifactRef


class NormalizedAudioOutput(BaseModel):
    storage_path: str
    sample_rate_hz: int = Field(gt=0)
    channels: Literal[1]
    format: Literal["wav"]


class PreprocessAsrAudioInput(BaseModel):
    audio: ArtifactRef


class DiarizeInput(BaseModel):
    audio: ArtifactRef


class DiarizationSegment(BaseModel):
    id: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    speaker_cluster_id: str
    speaker_label: str
    confidence: float | None = Field(default=None, ge=0, le=1)


class DiarizationOutput(BaseModel):
    provider: Literal["pyannote"]
    model_name: str
    segments: list[DiarizationSegment]


class TranscribeAsrInput(BaseModel):
    audio: ArtifactRef
    diarization: ArtifactRef


TranscribeQwenAsrInput = TranscribeAsrInput


class AsrWindowTranscript(BaseModel):
    """Raw ASR text for one continuous-speech window, before speaker attribution."""

    window_index: int = Field(ge=0)
    input_start_ms: int = Field(ge=0)
    input_end_ms: int = Field(ge=0)
    core_start_ms: int = Field(ge=0)
    core_end_ms: int = Field(ge=0)
    language: str | None
    text: str
    core_diarization_segment_ids: list[str] = Field(default_factory=list)


class AsrWindowTranscriptOutput(BaseModel):
    provider: AsrProvider
    model_name: str
    language: str | None
    alignment_supported: bool = True
    windows: list[AsrWindowTranscript]


class CorrectAsrWindowsInput(BaseModel):
    transcript: ArtifactRef


class CorrectedAsrWindowTranscript(AsrWindowTranscript):
    original_text: str


class CorrectedAsrWindowTranscriptOutput(BaseModel):
    asr_provider: AsrProvider
    asr_model_name: str
    correction_provider: Literal["pycorrector_llm", "pycorrector", "llm", "rules"]
    correction_model_name: str | None
    language: str | None
    windows: list[CorrectedAsrWindowTranscript]


class AlignTranscriptInput(BaseModel):
    audio: ArtifactRef
    diarization: ArtifactRef
    transcript: ArtifactRef


class TranscriptSegment(BaseModel):
    source_diarization_segment_id: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    text: str
    original_text: str | None = None
    speaker_cluster_id: str
    speaker_label: str


class AlignedTranscriptToken(BaseModel):
    token_index: int = Field(ge=0)
    text: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    speaker_cluster_id: str | None = None
    speaker_label: str | None = None
    attribution_status: Literal["matched", "ambiguous", "unmatched"]
    source_window_index: int = Field(ge=0)
    source_diarization_segment_id: str | None = None


class TranscriptOutput(BaseModel):
    provider: AsrProvider
    model_name: str
    language: str | None
    segments: list[TranscriptSegment]
    original_full_text: str | None = None
    alignment_tokens: list[AlignedTranscriptToken] | None = None
    alignment_model_name: str | None = None


class BuildUtterancesInput(BaseModel):
    transcript: ArtifactRef


class Utterance(BaseModel):
    utterance_index: int = Field(ge=0)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    text: str
    original_text: str | None = None
    speaker_cluster_id: str
    speaker_label: str
    source_segment_indexes: list[int] = Field(default_factory=lambda: [])
    source_diarization_segment_ids: list[str]


class UtterancesOutput(BaseModel):
    segments: list[Utterance]


class BuildSearchChunksInput(BaseModel):
    utterances: ArtifactRef


class SearchChunk(BaseModel):
    chunk_index: int = Field(ge=0)
    text: str
    original_text: str = ""
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    speaker_labels: list[str]
    speaker_cluster_ids: list[str]
    source_utterance_indexes: list[int]
    source_diarization_segment_ids: list[str]
    topic: str | None = None
    terms: list[str] = Field(default_factory=list, max_length=8)
    search_context: str | None = Field(default=None, max_length=300)
    topic_section_index: int | None = Field(default=None, ge=0)
    build_method: Literal["topic_boundary", "deterministic_fallback"] = "deterministic_fallback"

    def retrieval_text(self) -> str:
        return build_retrieval_text(self.text, self.topic, self.terms, self.search_context)

    def lexical_text(self) -> str:
        """Raw ASR text used exclusively by keyword retrieval.

        Unlike ``retrieval_text``, this deliberately has no fallback to the
        polished text and carries no generated topic/context metadata.
        """
        return self.original_text


class SearchChunksOutput(BaseModel):
    build_method: Literal["topic_boundary", "deterministic_fallback"] = "deterministic_fallback"
    chunks: list[SearchChunk]


class EmbeddingIndexingInput(BaseModel):
    chunks: ArtifactRef


class EmbeddedSearchChunk(SearchChunk):
    embedding: list[float]


class EmbeddingIndexingOutput(BaseModel):
    provider: Literal["sentence_transformers"]
    model_name: str
    dimensions: int = Field(gt=0)
    chunks: list[EmbeddedSearchChunk]


class GenerateSummaryInput(BaseModel):
    utterances: ArtifactRef


class RecordingSummaryOutput(BaseModel):
    provider: Literal["local", "zhipu", "gemini", "qwen"]
    model_name: str
    summary_text: str


class SummaryEmbeddingIndexingInput(BaseModel):
    summary: ArtifactRef


class SummaryEmbeddingIndexingOutput(BaseModel):
    provider: Literal["sentence_transformers"]
    model_name: str
    dimensions: int = Field(gt=0)
    document_index: int = Field(default=0, ge=0)
    document_type: Literal["profile"] = "profile"
    retrieval_text: str
    content_hash: str
    embedding: list[float]
