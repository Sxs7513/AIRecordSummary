from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EmbeddingProfile:
    key: str
    provider: str
    model_name: str
    dimensions: int
    distance_metric: str = "cosine"


EMBEDDING_PROFILES: dict[str, EmbeddingProfile] = {
    "qwen3-4b": EmbeddingProfile(
        key="qwen3-4b",
        provider="sentence_transformers",
        model_name="Qwen/Qwen3-Embedding-4B",
        dimensions=2560,
    ),
    "qwen3-0.6b": EmbeddingProfile(
        key="qwen3-0.6b",
        provider="sentence_transformers",
        model_name="Qwen/Qwen3-Embedding-0.6B",
        dimensions=1024,
    ),
}


def get_embedding_profile(key: str) -> EmbeddingProfile:
    try:
        return EMBEDDING_PROFILES[key]
    except KeyError as error:
        supported = ", ".join(EMBEDDING_PROFILES)
        raise ValueError(f"Unknown embedding profile {key!r}; expected one of: {supported}") from error


def embedding_profile_for_model(model_name: str, dimensions: int) -> EmbeddingProfile:
    for profile in EMBEDDING_PROFILES.values():
        if profile.model_name == model_name and profile.dimensions == dimensions:
            return profile
    raise ValueError(f"No embedding profile matches model={model_name!r}, dimensions={dimensions}")
