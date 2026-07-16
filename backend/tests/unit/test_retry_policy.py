from __future__ import annotations

from audio_processing.definition import recording_processing
from pipeline.contracts import RetryPolicy


def test_recording_processing_has_no_retry_attempt_limit() -> None:
    assert all(node.retry_policy.max_attempts is None for node in recording_processing.nodes)


def test_retry_delay_is_capped_even_when_attempts_are_unlimited() -> None:
    policy = RetryPolicy(initial_backoff_seconds=10, max_backoff_seconds=60)

    assert policy.retry_delay_seconds(1) == 10
    assert policy.retry_delay_seconds(4) == 60
