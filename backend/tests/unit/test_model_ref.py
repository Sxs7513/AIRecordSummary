from __future__ import annotations

import pytest

from l1_foundation.model_ref import OnlineModelRef


@pytest.mark.parametrize(
    ("value", "provider", "model"),
    [
        ("gemini-gemini-3.5-flash-lite", "gemini", "gemini-3.5-flash-lite"),
        ("qwen-qwen3.8-flash", "qwen", "qwen3.8-flash"),
        ("zhipu-glm-4.5-flash", "zhipu", "glm-4.5-flash"),
    ],
)
def test_online_model_ref_splits_only_the_provider_prefix(value: str, provider: str, model: str) -> None:
    reference = OnlineModelRef.parse(value)

    assert reference.provider == provider
    assert reference.model == model
    assert str(reference) == value


@pytest.mark.parametrize("value", ["gemini", "-gemini-3.5-flash-lite", "local-qwen3-4b"])
def test_online_model_ref_rejects_invalid_or_local_references(value: str) -> None:
    with pytest.raises(ValueError):
        OnlineModelRef.parse(value)
