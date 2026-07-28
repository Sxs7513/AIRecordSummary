from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeVar

TokenT = TypeVar("TokenT")
EditKind = Literal["equal", "substitute", "delete", "insert"]


@dataclass(frozen=True, slots=True)
class EditOperation:
    kind: EditKind
    reference: str | None
    hypothesis: str | None


@dataclass(frozen=True, slots=True)
class ErrorRate:
    substitutions: int
    deletions: int
    insertions: int
    reference_units: int
    value: float
    operations: tuple[EditOperation, ...]


def character_error_rate(reference: str, hypothesis: str) -> ErrorRate:
    return _error_rate(list(reference), list(hypothesis))


def word_error_rate(reference: str, hypothesis: str) -> ErrorRate:
    return _error_rate(reference.split(), hypothesis.split())


def _error_rate(reference: list[str], hypothesis: list[str]) -> ErrorRate:
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    distance = [[0] * columns for _ in range(rows)]
    choice: list[list[EditKind | None]] = [[None] * columns for _ in range(rows)]

    for row in range(1, rows):
        distance[row][0] = row
        choice[row][0] = "delete"
    for column in range(1, columns):
        distance[0][column] = column
        choice[0][column] = "insert"

    for row in range(1, rows):
        for column in range(1, columns):
            if reference[row - 1] == hypothesis[column - 1]:
                distance[row][column] = distance[row - 1][column - 1]
                choice[row][column] = "equal"
                continue
            candidates: tuple[tuple[int, EditKind], ...] = (
                (distance[row - 1][column - 1] + 1, "substitute"),
                (distance[row - 1][column] + 1, "delete"),
                (distance[row][column - 1] + 1, "insert"),
            )
            distance[row][column], choice[row][column] = min(candidates, key=lambda item: item[0])

    operations: list[EditOperation] = []
    substitutions = deletions = insertions = 0
    row = len(reference)
    column = len(hypothesis)
    while row > 0 or column > 0:
        kind = choice[row][column]
        if kind == "equal":
            operations.append(EditOperation(kind, reference[row - 1], hypothesis[column - 1]))
            row -= 1
            column -= 1
        elif kind == "substitute":
            substitutions += 1
            operations.append(EditOperation(kind, reference[row - 1], hypothesis[column - 1]))
            row -= 1
            column -= 1
        elif kind == "delete":
            deletions += 1
            operations.append(EditOperation(kind, reference[row - 1], None))
            row -= 1
        elif kind == "insert":
            insertions += 1
            operations.append(EditOperation(kind, None, hypothesis[column - 1]))
            column -= 1
        else:
            raise RuntimeError("Levenshtein backtrace is incomplete")

    operations.reverse()
    errors = substitutions + deletions + insertions
    value = errors / len(reference) if reference else (0.0 if not hypothesis else 1.0)
    return ErrorRate(substitutions, deletions, insertions, len(reference), value, tuple(operations))


def micro_error_rate(values: list[ErrorRate]) -> float:
    errors = sum(value.substitutions + value.deletions + value.insertions for value in values)
    reference_units = sum(value.reference_units for value in values)
    return errors / reference_units if reference_units else (0.0 if errors == 0 else 1.0)

