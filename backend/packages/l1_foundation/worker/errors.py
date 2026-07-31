from __future__ import annotations

from l1_foundation.worker.contracts import ComputeTaskError


class ComputeError(RuntimeError):
    """Base error for the internal compute protocol."""


class ComputeConfigurationError(ComputeError):
    """The compute client or worker was configured incorrectly."""


class ComputeTransportError(ComputeError):
    """The worker could not be reached or returned an invalid transport response."""


class ComputeStreamDisconnectedError(ComputeTransportError):
    """An internal SSE stream disconnected after transmission had started."""


class ComputeStateTimeoutError(ComputeTransportError):
    """Kafka accepted a task but its initial Redis state did not appear in time."""


class ComputeTaskNotFoundError(ComputeError):
    """The in-memory worker task does not exist."""


class ComputeTaskConflictError(ComputeError):
    """A task ID was reused with a different request."""


class ComputeQueueFullError(ComputeError):
    """The worker cannot accept more in-memory tasks."""


class ComputeRemoteError(ComputeError):
    """The worker rejected or failed a compute operation."""

    def __init__(self, error: ComputeTaskError, *, status_code: int | None = None) -> None:
        super().__init__(error.message)
        self.error = error
        self.status_code = status_code
