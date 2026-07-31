from __future__ import annotations

import math
import re
from collections.abc import Sequence

from l1_foundation.llm import ChatMessage, LlmGenerateResult

_CJK_CHARACTER = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")


class RagTokenBudgetExceeded(RuntimeError):
    pass


class RagTokenBudgetMiddleware:
    """Pre-call soft-cap guard and actual usage accounting for RAG model nodes."""

    def __init__(self, max_total_tokens: int) -> None:
        self.max_total_tokens = max_total_tokens

    def before_model(self, current_total: int, node: str) -> None:
        if current_total >= self.max_total_tokens:
            raise RagTokenBudgetExceeded(
                f"RAG token limit reached before {node}: used={current_total}, limit={self.max_total_tokens}"
            )

    @staticmethod
    def actual_usage(result: LlmGenerateResult) -> int:
        return (result.prompt_tokens or 0) + (result.completion_tokens or 0)

    @staticmethod
    def estimate_input_tokens(messages: Sequence[ChatMessage]) -> int:
        """Conservative dependency-free estimate used only for provider routing."""
        total = 0
        for message in messages:
            cjk = len(_CJK_CHARACTER.findall(message.content))
            non_cjk_bytes = len(_CJK_CHARACTER.sub("", message.content).encode("utf-8"))
            total += cjk + math.ceil(non_cjk_bytes / 4) + 4
        return total + 2
