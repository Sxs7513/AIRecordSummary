from __future__ import annotations

import pytest

from l1_foundation.embedding_profiles import embedding_profile_for_model, get_embedding_profile


def test_qwen_embedding_profiles_have_distinct_dimensions() -> None:
    large = get_embedding_profile("qwen3-4b")
    small = get_embedding_profile("qwen3-0.6b")

    assert (large.model_name, large.dimensions) == ("Qwen/Qwen3-Embedding-4B", 2560)
    assert (small.model_name, small.dimensions) == ("Qwen/Qwen3-Embedding-0.6B", 1024)
    assert embedding_profile_for_model(small.model_name, small.dimensions) == small


def test_unknown_embedding_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown embedding profile"):
        get_embedding_profile("unknown")
