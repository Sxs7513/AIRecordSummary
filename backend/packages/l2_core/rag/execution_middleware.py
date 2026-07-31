from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from functools import wraps
from inspect import isawaitable
from time import monotonic
from typing import cast

from l2_core.rag.checkpoint import RagCheckpointSession, completed_state
from l2_core.rag.contracts import RagGraphState


class RagExecutionCancelled(Exception):
    """Raised at a cooperative RAG boundary after the owning execution is cancelled."""


CancellationCheck = Callable[[], bool]
_CURRENT_CANCELLATION_CHECK: ContextVar[CancellationCheck | None] = ContextVar("rag_cancellation_check", default=None)
_CURRENT_CHECKPOINT_SESSION: ContextVar[RagCheckpointSession | None] = ContextVar("rag_checkpoint_session", default=None)


def raise_if_rag_cancelled() -> None:
    check = _CURRENT_CANCELLATION_CHECK.get()
    if check is not None and check():
        raise RagExecutionCancelled("RAG execution was cancelled")


class RagExecutionMiddleware:
    """Apply checkpoint and cooperative-cancellation behavior at RAG execution boundaries."""

    def wrap_node(
        self,
        node: Callable[..., object],
        *,
        graph_name: str,
        node_name: str,
    ) -> Callable[..., Awaitable[object]]:
        key = f"{graph_name}-{node_name}"

        @wraps(node)
        async def wrapped(*args: object, **kwargs: object) -> object:
            raise_if_rag_cancelled()
            checkpoint = _CURRENT_CHECKPOINT_SESSION.get()
            if checkpoint is not None and checkpoint.should_skip(key):
                return {}
            result = node(*args, **kwargs)
            if isawaitable(result):
                result = await cast(Awaitable[object], result)
            raise_if_rag_cancelled()
            if checkpoint is not None:
                current_state = cast(RagGraphState, args[0])
                output = cast(Mapping[str, object], result)
                await asyncio.to_thread(checkpoint.save, key, completed_state(current_state, output))
            return result

        return wrapped

    def wrap_delta(self, emit: Callable[[str], None]) -> Callable[[str], None]:
        @wraps(emit)
        def wrapped(delta: str) -> None:
            raise_if_rag_cancelled()
            emit(delta)

        return wrapped


rag_execution_middleware = RagExecutionMiddleware()


def throttled_cancellation_check(
    check: CancellationCheck,
    *,
    poll_interval_seconds: float = 0.1,
) -> CancellationCheck:
    """Turn a shared-store check into a cheap, sticky per-execution signal."""

    last_checked_at = float("-inf")
    cancelled = False

    def throttled() -> bool:
        nonlocal cancelled, last_checked_at
        if cancelled:
            return True
        now = monotonic()
        if now - last_checked_at < poll_interval_seconds:
            return False
        last_checked_at = now
        cancelled = check()
        return cancelled

    return throttled


@contextmanager
def rag_cancellation_scope(check: CancellationCheck) -> Generator[None]:
    token: Token[CancellationCheck | None] = _CURRENT_CANCELLATION_CHECK.set(check)
    try:
        yield
    finally:
        _CURRENT_CANCELLATION_CHECK.reset(token)


@contextmanager
def rag_checkpoint_scope(session: RagCheckpointSession) -> Generator[None]:
    token: Token[RagCheckpointSession | None] = _CURRENT_CHECKPOINT_SESSION.set(session)
    try:
        yield
    finally:
        _CURRENT_CHECKPOINT_SESSION.reset(token)
