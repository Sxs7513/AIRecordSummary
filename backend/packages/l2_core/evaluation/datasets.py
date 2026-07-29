from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Literal

from l2_core.asr_lab.normalization import normalize_text
from l2_core.evaluation.contracts import (
    ApprovedAnnotation,
    DatasetSplit,
    DatasetVersionPreview,
    FrozenCase,
    SplitSummary,
)


def build_dataset_preview(
    annotations: list[ApprovedAnnotation],
    *,
    normalization_name: str,
    normalization_version: str,
    seed: str = "asr-lab-v1",
    split_strategy_name: Literal["deterministic_group_hash_v1", "all_train_v1"] = "deterministic_group_hash_v1",
    excluded_count: int = 0,
) -> DatasetVersionPreview:
    if not annotations:
        raise ValueError("At least one approved annotation is required")

    grouped: dict[str, list[ApprovedAnnotation]] = defaultdict(list)
    for annotation in annotations:
        grouped[annotation.group_key].append(annotation)

    split_cases: list[FrozenCase] = []
    for group_key in sorted(grouped):
        group = sorted(grouped[group_key], key=lambda item: (str(item.source_asset_id), item.start_ms, str(item.id)))
        preferred = DatasetSplit.TRAIN if split_strategy_name == "all_train_v1" else _preferred_split(group_key, seed)
        split = _allowed_split(group, preferred)
        for annotation in group:
            normalized = normalize_text(annotation.reference_text, normalization_name, normalization_version)
            if not normalized:
                raise ValueError(f"Annotation {annotation.id} is empty after {normalization_name}_{normalization_version} normalization")
            split_cases.append(FrozenCase(annotation=annotation, split=split, normalized_reference_text=normalized))

    split_cases.sort(key=lambda item: (item.split.value, item.annotation.group_key, item.annotation.start_ms, str(item.annotation.id)))
    checksum_payload = [
        {
            "annotation_id": str(item.annotation.id),
            "asset_id": str(item.annotation.source_asset_id),
            "source_checksum": item.annotation.source_checksum,
            "start_ms": item.annotation.start_ms,
            "end_ms": item.annotation.end_ms,
            "reference_text": item.annotation.reference_text,
            "normalized_reference_text": item.normalized_reference_text,
            "language": item.annotation.language,
            "group_key": item.annotation.group_key,
            "split": item.split.value,
        }
        for item in split_cases
    ]
    checksum_input: dict[str, object] = {
        "normalization": [normalization_name, normalization_version],
        "seed": seed,
        "cases": checksum_payload,
    }
    if split_strategy_name != "deterministic_group_hash_v1":
        checksum_input["split_strategy"] = split_strategy_name
    checksum = hashlib.sha256(
        json.dumps(
            checksum_input,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    return DatasetVersionPreview(
        cases=tuple(split_cases),
        train=_summary(split_cases, DatasetSplit.TRAIN),
        validation=_summary(split_cases, DatasetSplit.VALIDATION),
        test=_summary(split_cases, DatasetSplit.TEST),
        excluded_count=excluded_count,
        checksum=checksum,
    )


def _preferred_split(group_key: str, seed: str) -> DatasetSplit:
    bucket = int.from_bytes(hashlib.sha256(f"{seed}:{group_key}".encode()).digest()[:8], "big") % 100
    if bucket < 80:
        return DatasetSplit.TRAIN
    if bucket < 90:
        return DatasetSplit.VALIDATION
    return DatasetSplit.TEST


def _allowed_split(group: list[ApprovedAnnotation], preferred: DatasetSplit) -> DatasetSplit:
    train_allowed = all(item.train_allowed for item in group)
    evaluation_allowed = all(item.evaluation_allowed for item in group)
    if not train_allowed and not evaluation_allowed:
        raise ValueError(f"Annotation group {group[0].group_key!r} is not allowed for training or evaluation")
    if preferred == DatasetSplit.TRAIN and train_allowed:
        return preferred
    if preferred != DatasetSplit.TRAIN and evaluation_allowed:
        return preferred
    return DatasetSplit.TRAIN if train_allowed else DatasetSplit.TEST


def _summary(cases: list[FrozenCase], split: DatasetSplit) -> SplitSummary:
    selected = [item for item in cases if item.split == split]
    return SplitSummary(
        group_count=len({item.annotation.group_key for item in selected}),
        case_count=len(selected),
        duration_ms=sum(item.annotation.end_ms - item.annotation.start_ms for item in selected),
    )
