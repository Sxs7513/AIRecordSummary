from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

type AnswerVerdict = Literal["direct_answer", "qualified_answer", "abstain"]


def answerability_matches(expected: AnswerVerdict, actual: AnswerVerdict) -> bool:
    return (expected != "abstain") == (actual != "abstain")


class AnswerKeyPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=1_000)
    evidence_ids: list[UUID] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_evidence_ids(self) -> Self:
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("key point evidence ids must be unique")
        return self


class AnswerAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_verdict: AnswerVerdict
    reference_answer: str | None = Field(default=None, max_length=10_000)
    key_points: list[AnswerKeyPoint] = Field(default_factory=lambda: [], max_length=30)

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        key_point_ids = [item.id for item in self.key_points]
        if len(key_point_ids) != len(set(key_point_ids)):
            raise ValueError("key point ids must be unique")
        if self.expected_verdict == "abstain":
            if self.key_points:
                raise ValueError("abstain annotation cannot contain key points")
            if self.reference_answer is not None:
                raise ValueError("abstain annotation cannot contain a reference answer")
            return self
        if not self.key_points:
            raise ValueError("answerable annotation requires at least one key point")
        if self.reference_answer is None or not self.reference_answer.strip():
            raise ValueError("answerable annotation requires a reference answer")
        self.reference_answer = self.reference_answer.strip()
        return self

    def validate_case_evidence(self, evidence_ids: set[UUID]) -> None:
        referenced = {evidence_id for item in self.key_points for evidence_id in item.evidence_ids}
        missing = referenced - evidence_ids
        if missing:
            raise ValueError("answer annotation references evidence outside the case")
        if self.expected_verdict != "abstain" and not evidence_ids:
            raise ValueError("answerable annotation requires gold evidence")

    def remap_evidence_ids(self, mapping: Mapping[UUID, UUID]) -> AnswerAnnotation:
        remapped: list[AnswerKeyPoint] = []
        for item in self.key_points:
            try:
                ids = [mapping[evidence_id] for evidence_id in item.evidence_ids]
            except KeyError as error:
                raise ValueError("cannot freeze answer annotation with an unmapped evidence id") from error
            remapped.append(item.model_copy(update={"evidence_ids": ids}))
        return self.model_copy(update={"key_points": remapped})
