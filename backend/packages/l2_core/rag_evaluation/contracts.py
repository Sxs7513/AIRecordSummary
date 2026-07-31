from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from l2_core.rag.contracts import EvidenceGrade


@dataclass(frozen=True, slots=True)
class EvidenceAnchor:
    id: UUID
    recording_id: UUID
    source_chunk_id: UUID | None
    quote: str
    start_ms: int
    end_ms: int
    relevance: int
    content_checksum: str


@dataclass(frozen=True, slots=True)
class RankedItem:
    recording_id: UUID
    source_chunk_id: UUID | None
    text: str
    start_ms: int
    end_ms: int
    score: float
    vector_score: float | None = None
    lexical_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


@dataclass(frozen=True, slots=True)
class EvidenceMatch:
    evidence_id: UUID | None
    relevance: int
    kind: str


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    hit_at_1: float
    hit_at_5: float
    hit_at_10: float
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    reciprocal_rank: float
    ndcg_at_10: float

    def as_dict(self) -> dict[str, float]:
        return {
            "hit_at_1": self.hit_at_1,
            "hit_at_5": self.hit_at_5,
            "hit_at_10": self.hit_at_10,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "recall_at_20": self.recall_at_20,
            "reciprocal_rank": self.reciprocal_rank,
            "ndcg_at_10": self.ndcg_at_10,
        }


@dataclass(frozen=True, slots=True)
class GradeMetrics:
    verdict_accuracy: float
    answer_false_negative_rate: float
    unsafe_answer_rate: float

    def as_dict(self) -> dict[str, float]:
        return {
            "verdict_accuracy": self.verdict_accuracy,
            "answer_false_negative_rate": self.answer_false_negative_rate,
            "unsafe_answer_rate": self.unsafe_answer_rate,
        }


GradeAssessmentPair = tuple[EvidenceGrade, EvidenceGrade]
