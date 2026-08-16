from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from operator import add
from typing import Annotated, Literal, Self, TypedDict, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from l2_core.rag.adjudication.contracts import AdjudicationAgentState, ClaimConfirmationDecision
from l2_core.rag.search_document import build_retrieval_text

RouteStatus = Literal["resolved", "ambiguous", "unresolved"]
StrategyId = Literal["fact_lookup", "metadata_lookup", "scope_summary"]
RetrievalStrategy = Literal["scope_summary", "chunk_search"]
RouteErrorCode = Literal["ambiguous_recording_scope", "unresolved_query", "unsupported_time_expression"]


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
    recording_ids: list[UUID] = Field(default_factory=_uuid_list)
    speaker_profile_ids: list[UUID] = Field(default_factory=_uuid_list)


class RagRoute(BaseModel):
    status: RouteStatus
    strategy_id: StrategyId | None = None
    content_query: str | None = None
    recording_limit: int | None = Field(default=None, ge=1, le=10)
    recording_rank: int | None = Field(default=None, ge=1, le=10)
    time_range: TimeRange | None = None
    inferred_filters: InferredFilters = Field(default_factory=InferredFilters)
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
    match_type: Literal["vector", "lexical", "hybrid", "scope"]
    facts: EvidenceFacts = Field(default_factory=EvidenceFacts)
    url: str

    def source_payload(self) -> dict[str, object]:
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
            "facts": self.facts.model_dump(mode="json"),
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
    """A scope-free content query and faithful anchors prepared for retrieval."""

    model_config = ConfigDict(extra="forbid")

    content_query: str = Field(min_length=1)
    terms: list[str] = Field(default_factory=list, max_length=6)
    phrases: list[str] = Field(default_factory=list, max_length=4)


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
    content_query: str
    retrieval_expanded_query: str | None
    retrieval_lexical_queries: list[str]
    retrieval_protected_lexical_queries: list[str]
    retrieval_attempt: int
    retrieval_candidates: list[dict[str, object]]
    protected_chunk_ids: list[str]
    rerank_input_tokens: int
    rerank_skipped_candidates: int
    evidence: list[Evidence]
    answer_evidence: list[Evidence]
    message: str | None
    grade: EvidenceGrade | None
    planning_required: bool
    answer_plan: AnswerPlan | None
    query_correction_risk: bool
    adjudication_agent_state: AdjudicationAgentState | None
    adjudication_user_decision: ClaimConfirmationDecision | None
    token_usage: Annotated[int, add]
    strategy_result: object | None
