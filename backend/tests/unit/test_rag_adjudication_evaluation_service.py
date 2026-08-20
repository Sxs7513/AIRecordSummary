from __future__ import annotations

from typing import Any
from uuid import uuid4

from l2_core.rag_adjudication_evaluation.service import RagAdjudicationEvaluationService


class _Result:
    def __init__(self, *, rows: list[dict[str, Any]] | None = None, scalar: int | None = None) -> None:
        self._rows = rows or []
        self._scalar = scalar

    def mappings(self) -> _Result:
        return self

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._rows)

    def scalar_one(self) -> int:
        assert self._scalar is not None
        return self._scalar


class _Connection:
    def __init__(self, evidence: list[dict[str, Any]]) -> None:
        self._evidence = evidence

    def execute(self, statement: object, parameters: object) -> _Result:
        del parameters
        sql = str(statement)
        if "rag_adjudication_evaluation_evidence_drafts" in sql:
            return _Result(rows=self._evidence)
        if "rag_adjudication_evaluation_correction_drafts" in sql:
            return _Result(scalar=1)
        raise AssertionError(f"Unexpected SQL: {sql}")


def test_case_validation_allows_cross_recording_reference_evidence() -> None:
    target_recording_id = uuid4()
    reference_recording_id = uuid4()
    connection = _Connection(
        [
            {"id": uuid4(), "role": "target", "source_recording_id": target_recording_id},
            {"id": uuid4(), "role": "reference", "source_recording_id": reference_recording_id},
        ]
    )

    RagAdjudicationEvaluationService._validate_case(connection, uuid4())  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage]
