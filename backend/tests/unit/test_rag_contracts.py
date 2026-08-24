from l2_core.rag.contracts import RetrievalTerms


def test_retrieval_terms_only_keep_anchors_present_in_source_and_prepared_queries() -> None:
    terms = RetrievalTerms(
        content_query="王总说 API v2 的上线时间定了吗？",
        terms=[" 王总 ", "API v2", "API v3"],
        phrases=["上线时间", "负责人"],
        evidence_queries=["负责人"],
    )

    sanitized = terms.with_faithful_anchors("王总说 API v2 的上线时间定了吗？")

    assert sanitized.terms == ["王总", "API v2"]
    assert sanitized.phrases == ["上线时间"]
    assert sanitized.evidence_queries == ["负责人"]
