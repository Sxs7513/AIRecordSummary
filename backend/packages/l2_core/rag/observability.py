from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from time import perf_counter
from typing import Any
from uuid import UUID

logger = logging.getLogger("rag")


def started_at() -> float:
    return perf_counter()


def elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1_000, 2)


def log_event(
    event: str,
    run_id: str | UUID,
    *,
    level: int = logging.INFO,
    exc_info: bool = False,
    data: Mapping[str, Any] | None = None,
    **fields: Any,
) -> None:
    payload = {
        "event": event,
        "run_id": str(run_id),
        **{key: value for key, value in (data or {}).items() if value is not None},
        **{key: value for key, value in fields.items() if value is not None},
    }
    logger.log(
        level,
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str),
        exc_info=exc_info,
    )
