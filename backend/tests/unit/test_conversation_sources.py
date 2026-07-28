from l2_core.conversations.service import ConversationService


def test_conversation_source_persistence_removes_retrieved_text() -> None:
    source: dict[str, object] = {
        "recording": {"id": "recording-1", "title": "项目周会"},
        "chunk": {"id": "chunk-1", "text": "不应重复落库的转写正文", "startMs": 100, "endMs": 200},
        "url": "/recordings/recording-1?t=100",
    }

    sanitized = ConversationService._persistent_sources([source])  # pyright: ignore[reportPrivateUsage]

    assert sanitized[0]["chunk"] == {"id": "chunk-1", "startMs": 100, "endMs": 200}
    assert source["chunk"] == {"id": "chunk-1", "text": "不应重复落库的转写正文", "startMs": 100, "endMs": 200}
