from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from operator import add
from typing import Annotated, Literal, NotRequired, Required, Self, TypedDict, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from l2_core.rag.adjudication.contracts import AdjudicationAgentState, ClaimConfirmationDecision
from l2_core.rag.search_document import build_retrieval_text

RouteStatus = Literal["resolved", "ambiguous", "unresolved"]
StrategyId = Literal["fact_lookup", "metadata_lookup", "scope_summary"]
RetrievalStrategy = Literal["scope_summary", "chunk_search"]
RouteErrorCode = Literal["ambiguous_recording_scope", "unresolved_query", "unsupported_time_expression"]
EvidenceMatchType = Literal["vector", "lexical", "hybrid", "scope"]
type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


class RetrievalCandidateRow(TypedDict, total=False):
    """A trusted search result after conversion from a SQLAlchemy row mapping."""

    chunk_id: Required[UUID]
    recording_id: Required[UUID]
    text: Required[str]
    start_ms: Required[int]
    end_ms: Required[int]
    speaker_labels: Required[list[str]]
    is_target_person: Required[bool]
    source_utterance_segment_ids: Required[list[UUID]]
    metadata: Required[Mapping[str, object] | None]
    title: Required[str]
    file_name: Required[str]
    location: Required[str | None]
    duration_seconds: Required[int | None]
    created_at: Required[datetime]
    score: Required[float]
    exact_match: bool
    match_type: EvidenceMatchType
    matched_speaker_profile_ids: list[UUID]
    protected_lexical_terms: list[str]
    retrieved_via_recording_profile: bool
    recording_profile_score: float


class RecordingProfileCandidate(TypedDict):
    recording_id: UUID
    score: float


class MetadataSpeaker(TypedDict):
    name: str
    speaking_duration_seconds: float


class RecordingMetadataRow(TypedDict):
    id: UUID
    file_name: str
    location: str | None
    duration_seconds: int | None
    created_at: datetime
    speakers: list[MetadataSpeaker]


class ScopeRecordingRow(TypedDict):
    id: UUID
    title: str
    file_name: str
    location: str | None
    duration_seconds: int | None
    created_at: datetime


class ScopeUtteranceRow(TypedDict):
    recording_id: UUID
    utterance_index: int
    speaker_label: str | None
    text: str
    start_ms: int
    end_ms: int
    is_target_person: bool
    speaker_profile_id: UUID | None
    utterance_count: int
    speaker_labels: list[str]


class EvidenceSourceRecording(TypedDict):
    id: str
    title: NotRequired[str]
    fileName: str
    location: str | None
    durationSeconds: int | None


class EvidenceSourceChunk(TypedDict):
    id: str
    startMs: int
    endMs: int
    speakerLabels: list[str]
    isTargetPerson: bool
    matchedSpeakerProfiles: list[str]


class EvidenceSourceFacts(TypedDict):
    scope_verified: bool
    speaker_count: NotRequired[int | None]
    utterance_count: NotRequired[int | None]
    transcript_truncated: NotRequired[bool]


class EvidenceSource(TypedDict):
    index: int
    recording: EvidenceSourceRecording
    chunk: EvidenceSourceChunk
    score: float
    matchType: EvidenceMatchType
    facts: EvidenceSourceFacts
    url: str
    adjudication: NotRequired[list[dict[str, object]]]


def _uuid_list() -> list[UUID]:
    return []


class TimeRange(BaseModel):
    text: str = Field(min_length=1)
    start: datetime | None = None
    end: datetime | None = None

    @model_validator(mode="after")
    def validate_boundaries(self) -> Self:
        if self.start is None and self.end is None:
            raise ValueError("Time range requires at least one boundary")
        for boundary in (self.start, self.end):
            if boundary is not None and (boundary.tzinfo is None or boundary.utcoffset() is None):
                raise ValueError("Time range boundaries must include a timezone offset")
        if self.start is not None and self.end is not None and self.start >= self.end:
            raise ValueError("Time range start must be before end")
        return self


class InferredFilters(BaseModel):
    """Constraints inferred from the question; never copied from a public API filter payload."""

    person_names: list[str] = Field(default_factory=list)
    file_names: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    target_person_only: bool = False
    speaker_profile_ids: list[UUID] = Field(default_factory=_uuid_list)


class RagRoute(BaseModel):
    status: RouteStatus
    strategy_id: StrategyId | None = None
    content_query: str | None = None
    recording_limit: int | None = Field(default=None, ge=1, le=10)
    recording_rank: int | None = Field(default=None, ge=1, le=10)
    time_range: TimeRange | None = None
    inferred_filters: InferredFilters = Field(default_factory=InferredFilters)
    history_recording_ids: list[UUID] = Field(default_factory=_uuid_list)
    error_code: RouteErrorCode | None = None
    reason: str = ""

    @model_validator(mode="before")
    @classmethod
    def normalize_strategy(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        payload = {
            key: item
            for key, item in cast(Mapping[object, object], value).items()
            if isinstance(key, str)
        }
        strategy = payload.get("strategy_id", payload.get("strategy"))
        normalized = {
            "chunk_search": "fact_lookup",
            "fact_lookup": "fact_lookup",
            "metadata_lookup": "metadata_lookup",
            "scope_summary": "scope_summary",
        }.get(strategy if isinstance(strategy, str) else "", strategy)
        payload["strategy_id"] = normalized
        payload.pop("strategy", None)
        return payload

    @property
    def strategy(self) -> RetrievalStrategy | None:
        """Compatibility view for callers migrating from retrieval modes to strategies."""

        if self.strategy_id == "fact_lookup":
            return "chunk_search"
        if self.strategy_id == "scope_summary":
            return "scope_summary"
        return None

class RagHistorySource(BaseModel):
    """A trusted recording-reference fact from a prior assistant answer."""

    recording_id: UUID


def _history_source_list() -> list[RagHistorySource]:
    return []


class RagHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    sources: list[RagHistorySource] = Field(default_factory=_history_source_list)


class ResolvedFilters(BaseModel):
    match_none: bool = False
    recording_scope_resolved: bool = False
    recording_ids: list[UUID] = Field(default_factory=_uuid_list)
    speaker_profile_ids: list[UUID] = Field(default_factory=_uuid_list)
    person_names: list[str] = Field(default_factory=list)
    file_names: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    target_person_only: bool = False
    created_from: datetime | None = None
    created_to: datetime | None = None


class EvidenceRecording(BaseModel):
    id: UUID
    title: str
    file_name: str
    location: str | None = None
    duration_seconds: int | None = None
    created_at: datetime | None = None


class EvidenceChunk(BaseModel):
    id: UUID
    text: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    speaker_labels: list[str] = Field(default_factory=list)
    is_target_person: bool = False
    matched_speaker_profiles: list[UUID] = Field(default_factory=_uuid_list)
    topic: str | None = None
    terms: list[str] = Field(default_factory=list)
    search_context: str | None = None

    def retrieval_text(self) -> str:
        return build_retrieval_text(self.text, self.topic, self.terms, self.search_context)


class EvidenceFacts(BaseModel):
    scope_verified: bool = False
    speaker_count: int | None = Field(default=None, ge=0)
    utterance_count: int | None = Field(default=None, ge=0)
    transcript_truncated: bool = False


class Evidence(BaseModel):
    index: int = Field(ge=1)
    recording: EvidenceRecording
    chunk: EvidenceChunk
    score: float
    match_type: EvidenceMatchType
    facts: EvidenceFacts = Field(default_factory=EvidenceFacts)
    url: str

    def source_payload(self) -> EvidenceSource:
        return {
            "index": self.index,
            "recording": {
                "id": str(self.recording.id),
                "title": self.recording.title,
                "fileName": self.recording.file_name,
                "location": self.recording.location,
                "durationSeconds": self.recording.duration_seconds,
            },
            "chunk": {
                "id": str(self.chunk.id),
                "startMs": self.chunk.start_ms,
                "endMs": self.chunk.end_ms,
                "speakerLabels": self.chunk.speaker_labels,
                "isTargetPerson": self.chunk.is_target_person,
                "matchedSpeakerProfiles": [str(item) for item in self.chunk.matched_speaker_profiles],
            },
            "score": self.score,
            "matchType": self.match_type,
            "facts": EvidenceSourceFacts(
                scope_verified=self.facts.scope_verified,
                speaker_count=self.facts.speaker_count,
                utterance_count=self.facts.utterance_count,
                transcript_truncated=self.facts.transcript_truncated,
            ),
            "url": self.url,
        }


class EvidenceGrade(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["direct_answer", "qualified_answer", "abstain"]
    reason: str = ""


class AnswerPlanItem(BaseModel):
    statement: str = Field(min_length=1)
    evidence_indexes: list[int] = Field(min_length=1)


class AnswerPlan(BaseModel):
    items: list[AnswerPlanItem] = Field(min_length=1)


class RetrievalTerms(BaseModel):
    """A scope-free content query, faithful anchors, and answer-evidence cues."""

    model_config = ConfigDict(extra="forbid")

    content_query: str = Field(min_length=1)
    terms: list[str] = Field(default_factory=list, max_length=6)
    phrases: list[str] = Field(default_factory=list, max_length=4)
    evidence_queries: list[str] = Field(default_factory=list, max_length=3)


StructuredFactKey = Literal["file_name", "duration_seconds", "created_at", "location", "speakers"]
StructuredFactValue = str | int | float | bool | list[str] | list[dict[str, str | int | float]] | None


def _evidence_list() -> list[Evidence]:
    return []


def _structured_fact_list() -> list[StructuredFact]:
    return []


def _evidence_source_list() -> list[EvidenceSource]:
    return []


class StructuredFact(BaseModel):
    key: StructuredFactKey
    label: str
    value: StructuredFactValue
    recording_id: UUID


class StrategyResult(BaseModel):
    status: Literal["ready", "not_found", "needs_clarification"]
    answer_context: str = ""
    corrected_answer_context: str | None = None
    evidence: list[Evidence] = Field(default_factory=_evidence_list)
    facts: list[StructuredFact] = Field(default_factory=_structured_fact_list)
    sources: list[EvidenceSource] = Field(default_factory=_evidence_source_list)
    original_sources: list[EvidenceSource] = Field(default_factory=_evidence_source_list)
    corrected_sources: list[EvidenceSource] = Field(default_factory=_evidence_source_list)
    message: str | None = None


class RagGraphState(TypedDict):
    run_id: str
    execution_mode: Literal["answer", "retrieval"]
    query: str
    history: list[RagHistoryMessage]
    limit: int
    scope_recording_ids: list[str]
    route: RagRoute | None
    route_error: str | None
    filters: ResolvedFilters | None
    history_scope_active: bool
    content_query: str
    retrieval_expanded_query: str | None
    retrieval_lexical_queries: list[str]
    retrieval_protected_lexical_queries: list[str]
    retrieval_attempt: int
    retrieval_candidates: list[RetrievalCandidateRow]
    protected_chunk_ids: list[str]
    rerank_input_tokens: int
    rerank_skipped_candidates: int
    evidence: list[Evidence]
    answer_evidence: list[Evidence]
    message: str | None
    grade: EvidenceGrade | None
    original_grade: EvidenceGrade | None
    corrected_grade: EvidenceGrade | None
    planning_required: bool
    answer_plan: AnswerPlan | None
    query_correction_risk: bool
    adjudication_agent_state: AdjudicationAgentState | None
    adjudication_user_decision: ClaimConfirmationDecision | None
    token_usage: Annotated[int, add]
    strategy_result: StrategyResult | None


class RagStateUpdate(TypedDict, total=False):
    """Fields a RAG graph node may update; keys outside graph state are rejected."""

    route: RagRoute | None
    route_error: str | None
    filters: ResolvedFilters | None
    history_scope_active: bool
    content_query: str
    retrieval_expanded_query: str | None
    retrieval_lexical_queries: list[str]
    retrieval_protected_lexical_queries: list[str]
    retrieval_attempt: int
    retrieval_candidates: list[RetrievalCandidateRow]
    protected_chunk_ids: list[str]
    rerank_input_tokens: int
    rerank_skipped_candidates: int
    evidence: list[Evidence]
    answer_evidence: list[Evidence]
    message: str | None
    grade: EvidenceGrade | None
    original_grade: EvidenceGrade | None
    corrected_grade: EvidenceGrade | None
    planning_required: bool
    answer_plan: AnswerPlan | None
    query_correction_risk: bool
    adjudication_agent_state: AdjudicationAgentState | None
    adjudication_user_decision: ClaimConfirmationDecision | None
    token_usage: int
    strategy_result: StrategyResult | None
