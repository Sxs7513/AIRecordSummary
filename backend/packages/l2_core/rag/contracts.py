from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, TypedDict
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

RouteStatus = Literal["resolved", "ambiguous", "unresolved"]
RetrievalStrategy = Literal["scope_summary", "chunk_search"]
RouteErrorCode = Literal["ambiguous_recording_scope", "unresolved_query", "unsupported_time_expression"]


def _uuid_list() -> list[UUID]:
    return []


class TimeRange(BaseModel):
    text: str = Field(min_length=1)
    kind: Literal["relative_duration", "calendar_period", "absolute_range"]
    unit: Literal["day", "week", "month", "quarter", "year"] | None = None
    value: int | None = Field(default=None, ge=1, le=100)
    offset: int | None = Field(default=None, ge=-100, le=100)

    @model_validator(mode="after")
    def validate_kind_fields(self) -> Self:
        if self.kind == "relative_duration":
            if self.unit is None or self.value is None or self.offset is not None:
                raise ValueError("relative_duration requires unit and value, and does not accept offset")
        elif self.kind == "calendar_period":
            if self.unit is None or self.value is not None:
                raise ValueError("calendar_period requires unit, and does not accept value")
        elif self.unit is not None or self.value is not None or self.offset is not None:
            raise ValueError("absolute_range only accepts text")
        return self


class InferredFilters(BaseModel):
    """Constraints inferred from the question; never copied from a public API filter payload."""

    person_names: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    target_person_only: bool = False
    recording_ids: list[UUID] = Field(default_factory=_uuid_list)
    speaker_profile_ids: list[UUID] = Field(default_factory=_uuid_list)


class RagRoute(BaseModel):
    status: RouteStatus
    strategy: RetrievalStrategy | None = None
    topic: str | None = None
    recording_limit: int | None = Field(default=None, ge=1, le=10)
    recording_rank: int | None = Field(default=None, ge=1, le=10)
    time_range: TimeRange | None = None
    inferred_filters: InferredFilters = Field(default_factory=InferredFilters)
    error_code: RouteErrorCode | None = None
    reason: str = ""


class RagHistorySource(BaseModel):
    """A trusted recording-reference fact from a prior assistant answer."""

    recording_id: UUID
    title: str
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)


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


class EvidenceChunk(BaseModel):
    id: UUID
    text: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    speaker_labels: list[str] = Field(default_factory=list)
    is_target_person: bool = False
    matched_speaker_profiles: list[UUID] = Field(default_factory=_uuid_list)


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
    sufficient: bool
    rewrite_query: str | None = None
    planning_required: bool = False
    planning_reason: str = ""
    reason: str = ""


class AnswerPlanItem(BaseModel):
    statement: str = Field(min_length=1)
    evidence_indexes: list[int] = Field(min_length=1)


class AnswerPlan(BaseModel):
    items: list[AnswerPlanItem] = Field(min_length=1)


class RagGraphState(TypedDict):
    run_id: str
    query: str
    history: list[RagHistoryMessage]
    limit: int
    scope_recording_ids: list[str]
    route: RagRoute | None
    route_error: str | None
    filters: ResolvedFilters | None
    retrieval_query: str
    retrieval_attempt: int
    evidence: list[Evidence]
    answer_evidence: list[Evidence]
    message: str | None
    grade: EvidenceGrade | None
    planning_required: bool
    answer_plan: AnswerPlan | None
