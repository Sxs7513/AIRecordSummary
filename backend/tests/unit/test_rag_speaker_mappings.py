from uuid import uuid4

from rag.contracts import ResolvedFilters
from rag.retrieval import RagRetriever


def test_chunk_person_filter_uses_recording_speaker_mapping_and_chunk_clusters() -> None:
    clauses: list[str] = []
    values: dict[str, object] = {}

    RagRetriever._append_chunk_filters(clauses, values, ResolvedFilters(person_names=["张三"]))

    rendered = "\n".join(clauses)
    assert "recording_speaker_mappings" in rendered
    assert "speaker_cluster_id = any(chunks.speaker_cluster_ids)" in rendered
    assert values["person_patterns"] == ["%张三%"]


def test_speaker_profile_filter_does_not_depend_on_stale_chunk_metadata() -> None:
    profile_id = uuid4()
    clauses: list[str] = []
    values: dict[str, object] = {}

    RagRetriever._append_chunk_filters(clauses, values, ResolvedFilters(speaker_profile_ids=[profile_id]))

    rendered = "\n".join(clauses)
    assert "recording_speaker_mappings" in rendered
    assert "chunks.matched_speaker_profile_ids" not in rendered
    assert values["speaker_profile_ids"] == [str(profile_id)]
