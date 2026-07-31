from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ResourceQueue(StrEnum):
    IO = "io"
    CPU = "cpu"
    GPU_NORMAL = "gpu_normal"
    GPU_HIGH = "gpu_high"


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int | None = None
    initial_backoff_seconds: int = 15
    max_backoff_seconds: int = 600

    def __post_init__(self) -> None:
        if self.max_attempts is not None and self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_backoff_seconds < 0:
            raise ValueError("initial_backoff_seconds cannot be negative")
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("max_backoff_seconds cannot be smaller than initial_backoff_seconds")

    def retry_delay_seconds(self, attempt_count: int) -> int:
        exponential_delay = self.initial_backoff_seconds * (2 ** max(0, attempt_count - 1))
        return min(exponential_delay, self.max_backoff_seconds)
