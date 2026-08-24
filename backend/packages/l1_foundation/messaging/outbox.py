from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import cast
from uuid import UUID

from sqlalchemy import Connection, Engine, RowMapping, text

from l1_foundation.messaging.contracts import EventEnvelope


@dataclass(frozen=True, slots=True)
class OutboxMessage:
    event_id: UUID
    channel: str
    topic: str
    partition_key: str
    event: EventEnvelope
    attempt_count: int
    created_at: datetime


class OutboxRepository:
    def __init__(self, engine: Engine | None = None) -> None:
        self._engine = engine

    def enqueue(
        self,
        connection: Connection,
        *,
        channel: str,
        topic: str,
        partition_key: str,
        aggregate_type: str,
        aggregate_id: UUID | str,
        event: EventEnvelope,
    ) -> None:
        connection.execute(
            text(
                """
                insert into integration_outbox (
                    event_id, channel, topic, partition_key, event_type,
                    aggregate_type, aggregate_id, payload, occurred_at
                ) values (
                    :event_id, :channel, :topic, :partition_key, :event_type,
                    :aggregate_type, :aggregate_id, cast(:payload as jsonb), :occurred_at
                ) on conflict (event_id) do nothing
                """
            ),
            {
                "event_id": event.event_id,
                "channel": channel,
                "topic": topic,
                "partition_key": partition_key,
                "event_type": event.event_type,
                "aggregate_type": aggregate_type,
                "aggregate_id": str(aggregate_id),
                "payload": event.model_dump_json(),
                "occurred_at": event.occurred_at,
            },
        )

    def claim(self, relay_id: str, *, batch_size: int, lease_seconds: int) -> list[OutboxMessage]:
        engine = self._require_engine()
        with engine.begin() as connection:
            rows = (
                connection.execute(
                    text(
                        """
                    with picked as (
                        select candidate.event_id
                        from integration_outbox candidate
                        where candidate.published_at is null
                          and candidate.exhausted_at is null
                          and candidate.available_at <= now()
                          and (candidate.locked_until is null or candidate.locked_until < now())
                          and not exists (
                              select 1 from integration_outbox earlier
                              where earlier.aggregate_type = candidate.aggregate_type
                                and earlier.aggregate_id = candidate.aggregate_id
                                and earlier.published_at is null
                                and earlier.exhausted_at is null
                                and (earlier.created_at, earlier.event_id) < (candidate.created_at, candidate.event_id)
                          )
                        order by candidate.created_at, candidate.event_id
                        for update skip locked
                        limit :batch_size
                    )
                    update integration_outbox outgoing
                    set locked_by = :relay_id,
                        locked_until = now() + make_interval(secs => :lease_seconds)
                    from picked
                    where outgoing.event_id = picked.event_id
                    returning outgoing.*
                    """
                    ),
                    {"relay_id": relay_id, "lease_seconds": lease_seconds, "batch_size": batch_size},
                )
                .mappings()
                .all()
            )
        return [self._message(row) for row in rows]

    def mark_published(self, event_id: UUID, relay_id: str) -> None:
        with self._require_engine().begin() as connection:
            connection.execute(
                text(
                    """
                    update integration_outbox
                    set published_at = now(), locked_by = null, locked_until = null, last_error = null
                    where event_id = :event_id and locked_by = :relay_id and published_at is null
                    """
                ),
                {"event_id": event_id, "relay_id": relay_id},
            )

    def mark_failed(self, event_id: UUID, relay_id: str, error: str, *, max_attempts: int) -> None:
        with self._require_engine().begin() as connection:
            connection.execute(
                text(
                    """
                    update integration_outbox
                    set attempt_count = attempt_count + 1,
                        available_at = now() + make_interval(secs => least(300, power(2, least(attempt_count, 8))::integer)),
                        exhausted_at = case when attempt_count + 1 >= :max_attempts then now() else null end,
                        last_error = :error,
                        locked_by = null,
                        locked_until = null
                    where event_id = :event_id and locked_by = :relay_id and published_at is null
                    """
                ),
                {"event_id": event_id, "relay_id": relay_id, "error": error[:2000], "max_attempts": max_attempts},
            )

    def metrics(self) -> list[dict[str, object]]:
        with self._require_engine().connect() as connection:
            rows = (
                connection.execute(
                    text(
                        """
                    select channel,
                           count(*) filter (where published_at is null and exhausted_at is null) as pending,
                           count(*) filter (where exhausted_at is not null) as exhausted,
                           coalesce(max(extract(epoch from now() - created_at))
                               filter (where published_at is null and exhausted_at is null), 0) as oldest_pending_seconds,
                           coalesce(avg(extract(epoch from published_at - created_at))
                               filter (where published_at >= now() - interval '5 minutes'), 0) as recent_publish_latency_seconds,
                           coalesce(sum(attempt_count) filter (where published_at is null), 0) as retry_count
                    from integration_outbox
                    group by channel order by channel
                    """
                    )
                )
                .mappings()
                .all()
            )
        return [dict(row) for row in rows]

    def delete_published_before(self, cutoff: datetime) -> int:
        with self._require_engine().begin() as connection:
            result = connection.execute(text("delete from integration_outbox where published_at < :cutoff"), {"cutoff": cutoff})
            return result.rowcount

    def _require_engine(self) -> Engine:
        if self._engine is None:
            raise RuntimeError("OutboxRepository requires an engine for relay operations")
        return self._engine

    @staticmethod
    def _message(row: RowMapping) -> OutboxMessage:
        return OutboxMessage(
            event_id=cast(UUID, row["event_id"]),
            channel=cast(str, row["channel"]),
            topic=cast(str, row["topic"]),
            partition_key=cast(str, row["partition_key"]),
            event=EventEnvelope.model_validate(row["payload"]),
            attempt_count=cast(int, row["attempt_count"]),
            created_at=cast(datetime, row["created_at"]),
        )
