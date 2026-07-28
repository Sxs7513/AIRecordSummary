from uuid import UUID

import pytest

from evaluation.contracts import ApprovedAnnotation, DatasetSplit
from evaluation.datasets import build_dataset_preview


def _annotation(
    index: int,
    *,
    group: str,
    text: str = "项目验收",
    train_allowed: bool = True,
    evaluation_allowed: bool = True,
) -> ApprovedAnnotation:
    return ApprovedAnnotation(
        id=UUID(int=index),
        source_asset_id=UUID(int=100 + index),
        source_checksum=f"checksum-{index}",
        start_ms=index * 1000,
        end_ms=index * 1000 + 800,
        reference_text=text,
        language="zh",
        group_key=group,
        train_allowed=train_allowed,
        evaluation_allowed=evaluation_allowed,
    )


def test_dataset_preview_keeps_one_group_in_one_split_and_is_reproducible() -> None:
    annotations = [
        _annotation(1, group="recording-a"),
        _annotation(2, group="recording-a"),
        _annotation(3, group="recording-b"),
        _annotation(4, group="recording-c"),
    ]

    first = build_dataset_preview(annotations, normalization_name="zh_asr", normalization_version="v1", seed="fixed")
    second = build_dataset_preview(list(reversed(annotations)), normalization_name="zh_asr", normalization_version="v1", seed="fixed")

    assert first.checksum == second.checksum
    splits_by_group: dict[str, set[DatasetSplit]] = {}
    for case in first.cases:
        splits_by_group.setdefault(case.annotation.group_key, set()).add(case.split)
    assert all(len(splits) == 1 for splits in splits_by_group.values())
    assert first.train.case_count + first.validation.case_count + first.test.case_count == 4


def test_evaluation_only_group_never_enters_train() -> None:
    preview = build_dataset_preview(
        [_annotation(1, group="evaluation-only", train_allowed=False)],
        normalization_name="zh_asr",
        normalization_version="v1",
    )

    assert preview.train.case_count == 0
    assert preview.test.case_count == 1


def test_normalized_empty_reference_is_rejected_before_freeze() -> None:
    with pytest.raises(ValueError, match="empty after"):
        build_dataset_preview(
            [_annotation(1, group="punctuation", text="，。！？")],
            normalization_name="zh_asr",
            normalization_version="v1",
        )
