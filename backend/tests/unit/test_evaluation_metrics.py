from l2_core.evaluation.metrics import character_error_rate, micro_error_rate, word_error_rate


def test_character_error_rate_records_substitution_deletion_and_insertion() -> None:
    result = character_error_rate("明天下午三点", "明天上午三点钟")

    assert result.substitutions == 1
    assert result.deletions == 0
    assert result.insertions == 1
    assert result.reference_units == 6
    assert result.value == 2 / 6
    assert [item.kind for item in result.operations].count("substitute") == 1
    assert [item.kind for item in result.operations].count("insert") == 1


def test_empty_reference_has_stable_error_rate() -> None:
    assert character_error_rate("", "").value == 0
    assert character_error_rate("", "额外内容").value == 1


def test_word_error_rate_and_micro_aggregation_do_not_average_cases() -> None:
    short = word_error_rate("one", "wrong")
    long = word_error_rate("one two three four", "one two three four")

    assert short.value == 1
    assert long.value == 0
    assert micro_error_rate([short, long]) == 0.2

