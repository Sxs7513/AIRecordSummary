from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from l2_core.audio_processing.stages.summary.stage import SafeTextStream
from l2_core.generation.contracts import GenerationEvent
from l2_core.generation.hub import GenerationStreamHub
from routes.generations import encode_sse


def test_safe_text_stream_never_emits_a_leading_thinking_block() -> None:
    visible: list[str] = []
    stream = SafeTextStream(visible.append)

    stream.feed("<thi")
    stream.feed("nk>内部推理")
    stream.feed("内容</think>最终")
    stream.feed("总结")
    stream.finish()

    assert "".join(visible) == "最终总结"


def test_hub_broadcasts_to_only_the_matching_generation_subscription() -> None:
    hub = GenerationStreamHub()
    run_id = uuid4()
    other_run_id = uuid4()
    subscription = hub.subscribe(run_id)
    other = hub.subscribe(other_run_id)
    event = _event(run_id, 1)

    hub.publish(event)

    assert subscription.get(0) == event
    assert other.get(0) is None
    hub.unsubscribe(run_id, subscription)


def test_sse_event_uses_sequence_as_the_browser_reconnect_id() -> None:
    encoded = encode_sse(_event(uuid4(), 9))

    assert encoded.startswith("id: 9\nevent: content.delta\ndata: ")
    assert encoded.endswith("\n\n")


def _event(run_id: UUID, sequence: int) -> GenerationEvent:
    return GenerationEvent(
        run_id=run_id,
        seq=sequence,
        type="content.delta",
        at=datetime.now(UTC),
        data={"blocks": [{"type": "text", "value": "测试"}]},
    )
