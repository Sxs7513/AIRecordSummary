from __future__ import annotations

# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
import json
from typing import Any, cast
from uuid import UUID

from redis import Redis

from l2_core.rag.contracts import RagHistoryMessage, RagHistorySource


class ConversationHistoryStore:
    """Best-effort Redis cache for the recent completed turns of a conversation."""

    MAX_TURNS = 10
    MAX_TURN_CHARS = 3_000
    TTL_SECONDS = 86_400

    def __init__(self, client: Redis) -> None:
        self._client = client

    @classmethod
    def from_url(cls, url: str) -> ConversationHistoryStore:
        return cls(Redis.from_url(url, decode_responses=True))

    def close(self) -> None:
        self._client.close()

    def get(self, conversation_id: UUID) -> list[RagHistoryMessage] | None:
        values = cast(list[str], self._client.lrange(self._key(conversation_id), 0, self.MAX_TURNS - 1))
        if not values:
            return None
        history: list[RagHistoryMessage] = []
        for value in reversed(values):
            entry = cast(dict[str, Any], json.loads(value))
            history.extend((self._message(entry["user"]), self._message(entry["assistant"])))
        return history

    def put(self, conversation_id: UUID, turns: list[tuple[RagHistoryMessage, RagHistoryMessage]]) -> None:
        key = self._key(conversation_id)
        payloads = [json.dumps(self._turn(user, assistant), ensure_ascii=False, separators=(",", ":")) for user, assistant in turns[-self.MAX_TURNS :]]
        pipeline = self._client.pipeline(transaction=True)
        pipeline.delete(key)
        if payloads:
            # LPUSH makes the newest turn index 0; get() reverses it for prompt order.
            pipeline.lpush(key, *payloads)
            pipeline.ltrim(key, 0, self.MAX_TURNS - 1)
            pipeline.expire(key, self.TTL_SECONDS)
        pipeline.execute()

    def delete(self, conversation_id: UUID) -> None:
        self._client.delete(self._key(conversation_id))

    @classmethod
    def _turn(cls, user: RagHistoryMessage, assistant: RagHistoryMessage) -> dict[str, dict[str, object]]:
        user_content = user.content[: cls.MAX_TURN_CHARS]
        assistant_content = assistant.content[: max(0, cls.MAX_TURN_CHARS - len(user_content))]
        return {
            "user": cls._serialized_message(user, user_content),
            "assistant": cls._serialized_message(assistant, assistant_content),
        }

    @staticmethod
    def _serialized_message(message: RagHistoryMessage, content: str) -> dict[str, object]:
        return {
            "role": message.role,
            "content": content,
            "sources": [{"recording_id": str(source.recording_id)} for source in message.sources],
        }

    @staticmethod
    def _message(value: object) -> RagHistoryMessage:
        raw = cast(dict[str, Any], value)
        sources = []
        for source in cast(list[object], raw.get("sources", [])):
            if not isinstance(source, dict) or not source.get("recording_id"):
                continue
            sources.append(RagHistorySource(recording_id=UUID(str(source["recording_id"]))))
        return RagHistoryMessage(role=raw["role"], content=str(raw.get("content", "")), sources=sources)

    @staticmethod
    def _key(conversation_id: UUID) -> str:
        return f"conversation:{conversation_id}:recent_turns"
