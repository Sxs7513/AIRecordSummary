from __future__ import annotations

import pytest

from l1_foundation.llm import ChatMessage, ChatRole, LlmGenerateResult, LlmProvider
from l2_core.rag.token_budget import RagTokenBudgetExceeded, RagTokenBudgetMiddleware


def test_actual_usage_uses_only_provider_or_tokenizer_result() -> None:
    result = LlmGenerateResult(
        text="完成",
        provider=LlmProvider.GEMINI,
        model="gemini-test",
        prompt_tokens=120,
        completion_tokens=30,
    )

    assert RagTokenBudgetMiddleware.actual_usage(result) == 150


def test_soft_cap_blocks_the_next_model_call() -> None:
    middleware = RagTokenBudgetMiddleware(30_000)

    middleware.before_model(29_999, "answer")
    with pytest.raises(RagTokenBudgetExceeded, match="before plan"):
        middleware.before_model(30_000, "plan")


def test_input_estimate_is_reserved_for_provider_routing() -> None:
    short = [ChatMessage(ChatRole.USER, "请判断证据是否足够")]
    long = [ChatMessage(ChatRole.USER, "证据" * 3_000)]

    assert RagTokenBudgetMiddleware.estimate_input_tokens(short) < 100
    assert RagTokenBudgetMiddleware.estimate_input_tokens(long) > 2_500
