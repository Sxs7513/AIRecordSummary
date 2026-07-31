from l2_core.rag.citations import normalize_answer_citations


def test_citations_are_filtered_and_renumbered_by_first_appearance() -> None:
    result = normalize_answer_citations(
        "延期风险[3]，成本风险[1][3]。",
        [
            {"index": 1, "title": "成本"},
            {"index": 2, "title": "未引用"},
            {"index": 3, "title": "延期"},
        ],
    )

    assert result.text == "延期风险[1]，成本风险[2][1]。"
    assert result.sources == [
        {"index": 1, "title": "延期"},
        {"index": 2, "title": "成本"},
    ]
    assert result.cited_indexes == (3, 1)
    assert result.invalid_indexes == ()


def test_invalid_citations_are_removed_from_the_final_answer() -> None:
    result = normalize_answer_citations("有依据[2]，无效依据[9]。", [{"index": 2, "title": "依据"}])

    assert result.text == "有依据[1]，无效依据。"
    assert result.sources == [{"index": 1, "title": "依据"}]
    assert result.invalid_indexes == (9,)
