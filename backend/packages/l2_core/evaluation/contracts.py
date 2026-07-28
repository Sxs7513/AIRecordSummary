from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID


class AnnotationStatus(StrEnum):
    DRAFT = "draft"
    REVIEWED = "reviewed"
    APPROVED = "approved"


class DatasetVersionStatus(StrEnum):
    BUILDING = "building"
    FROZEN = "frozen"


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ApprovedAnnotation:
    id: UUID
    source_asset_id: UUID
    source_checksum: str
    start_ms: int
    end_ms: int
    reference_text: str
    language: str | None
    group_key: str
    train_allowed: bool
    evaluation_allowed: bool


@dataclass(frozen=True, slots=True)
class FrozenCase:
    annotation: ApprovedAnnotation
    split: DatasetSplit
    normalized_reference_text: str


@dataclass(frozen=True, slots=True)
class SplitSummary:
    group_count: int
    case_count: int
    duration_ms: int


@dataclass(frozen=True, slots=True)
class DatasetVersionPreview:
    cases: tuple[FrozenCase, ...]
    train: SplitSummary
    validation: SplitSummary
    test: SplitSummary
    excluded_count: int
    checksum: str

