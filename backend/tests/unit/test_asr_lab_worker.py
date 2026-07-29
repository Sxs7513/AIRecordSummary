from __future__ import annotations

from threading import Event, Thread
from typing import Any, cast

from l2_core.asr_lab.training import AsrTrainingWorker
from l2_core.asr_lab.worker import AsrLabWorker


class BlockingEvaluationWorker:
    def __init__(self) -> None:
        self.started = Event()
        self.received_stop_event: Event | None = None

    def run_once(self, stop_event: Event | None = None) -> bool:
        self.received_stop_event = stop_event
        self.started.set()
        assert stop_event is not None
        stop_event.wait(5)
        return True


class UnexpectedTrainingWorker:
    def run_once(self, _stop_event: Event | None = None) -> bool:
        raise AssertionError("training worker must not run after shutdown is requested")


def test_asr_lab_worker_stop_wakes_active_loop_and_propagates_signal() -> None:
    evaluation = BlockingEvaluationWorker()
    worker = object.__new__(AsrLabWorker)
    worker_state = cast(Any, worker)
    worker_state._evaluation = evaluation
    worker_state._training = UnexpectedTrainingWorker()
    worker_state._poll_seconds = 60.0
    worker_state._stop_event = Event()

    thread = Thread(target=worker.run_forever)
    thread.start()
    assert evaluation.started.wait(1)

    worker.stop()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert evaluation.received_stop_event is worker_state._stop_event


def test_training_worker_parses_structured_subprocess_progress() -> None:
    progress = AsrTrainingWorker._parse_training_progress(  # pyright: ignore[reportPrivateUsage]
        '2026-07-29 INFO train: TRAIN_PROGRESS {"step":2,"max_steps":5,"percent":40.0,"loss":1.25,"learning_rate":0.0002}'
    )

    assert progress == (2, 5, 40.0, 1.25, 0.0002)
