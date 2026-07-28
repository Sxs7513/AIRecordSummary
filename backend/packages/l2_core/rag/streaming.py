from __future__ import annotations

from collections.abc import Callable


class ThinkTagFilter:
    """Remove ``<think>…</think>`` blocks from incrementally produced model text."""

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self, on_text: Callable[[str], None]) -> None:
        self._on_text = on_text
        self._pending = ""
        self._inside_thinking = False

    def feed(self, value: str) -> None:
        if not value:
            return
        self._pending += value
        while self._pending:
            if self._inside_thinking:
                closing_at = self._pending.lower().find(self._CLOSE)
                if closing_at < 0:
                    self._pending = self._tag_prefix_tail(self._pending, (self._CLOSE,))
                    return
                self._pending = self._pending[closing_at + len(self._CLOSE) :]
                self._inside_thinking = False
                continue

            lower_pending = self._pending.lower()
            opening_at = lower_pending.find(self._OPEN)
            closing_at = lower_pending.find(self._CLOSE)
            tag_at = min(index for index in (opening_at, closing_at) if index >= 0) if opening_at >= 0 or closing_at >= 0 else -1
            if tag_at < 0:
                tail = self._tag_prefix_tail(self._pending, (self._OPEN, self._CLOSE))
                visible = self._pending[: len(self._pending) - len(tail)] if tail else self._pending
                if visible:
                    self._on_text(visible)
                self._pending = tail
                return

            if tag_at:
                self._on_text(self._pending[:tag_at])
            if tag_at == opening_at:
                self._pending = self._pending[tag_at + len(self._OPEN) :]
                self._inside_thinking = True
            else:
                self._pending = self._pending[tag_at + len(self._CLOSE) :]

    def finish(self) -> None:
        # `_pending` is either a possible split tag or text inside an unclosed think block.
        self._pending = ""

    @staticmethod
    def _tag_prefix_tail(value: str, tags: tuple[str, ...]) -> str:
        lower_value = value.lower()
        for size in range(min(len(value), max(len(tag) for tag in tags) - 1), 0, -1):
            tail = lower_value[-size:]
            if any(tag.startswith(tail) for tag in tags):
                return value[-size:]
        return ""
