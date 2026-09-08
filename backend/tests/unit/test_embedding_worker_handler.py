from __future__ import annotations

from typing import Any, cast

import pytest

from l1_foundation.worker import WorkerExecutionContext
from l2_core.audio_processing.worker_tasks import EmbeddingEncodeTaskInput
from l3_app.compute_worker.audio_handlers import EmbeddingEncodeHandler


def test_embedding_worker_rejects_a_different_profile_before_inference() -> None:
    handler = object.__new__(EmbeddingEncodeHandler)
    handler._embedding_profile = "qwen3-4b"  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(ValueError, match="requested=qwen3-0.6b, worker_active=qwen3-4b"):
        handler(
            EmbeddingEncodeTaskInput(texts=["测试"], embedding_profile="qwen3-0.6b"),
            cast(WorkerExecutionContext, cast(Any, object())),
        )
