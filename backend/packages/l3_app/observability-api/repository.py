from __future__ import annotations

import json
from datetime import datetime
from typing import cast
from uuid import UUID

from sqlalchemy import Engine, RowMapping, text

from l1_foundation.observability.contracts import ModelInvocationRecord, RagExecutionSpanRecord


class ObservabilityRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def abandon_stale_records(self) -> None:
        """Close records left running by a caller that can no longer publish a terminal state."""
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    update rag_execution_spans
                    set status = 'abandoned',
                        finished_at = now(),
                        elapsed_ms = extract(epoch from (now() - started_at)) * 1000,
                        error_type = coalesce(error_type, 'stale_running_record'),
                        updated_at = now()
                    where status = 'running' and started_at < now() - interval '24 hours'
                    """
                )
            )
            connection.execute(
                text(
                    """
                    update model_invocations
                    set status = 'abandoned',
                        finished_at = now(),
                        elapsed_ms = extract(epoch from (now() - started_at)) * 1000,
                        error_type = coalesce(error_type, 'stale_running_record'),
                        updated_at = now()
                    where status = 'running' and started_at < now() - interval '24 hours'
                    """
                )
            )

    def upsert_span(self, record: RagExecutionSpanRecord) -> None:
        payload = record.model_dump(mode="python")
        payload["metadata"] = json.dumps(record.metadata)
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    insert into rag_execution_spans (
                        id, workspace_id, generation_run_id, parent_span_id, component,
                        operation, operation_version, attempt, status, started_at,
                        finished_at, elapsed_ms, error_type, metadata
                    ) values (
                        :id, :workspace_id, :generation_run_id, :parent_span_id, :component,
                        :operation, :operation_version, :attempt, :status, :started_at,
                        :finished_at, :elapsed_ms, :error_type, cast(:metadata as jsonb)
                    )
                    on conflict (id) do update set
                        status = case
                            when rag_execution_spans.status <> 'running' then rag_execution_spans.status
                            else excluded.status
                        end,
                        finished_at = coalesce(rag_execution_spans.finished_at, excluded.finished_at),
                        elapsed_ms = coalesce(rag_execution_spans.elapsed_ms, excluded.elapsed_ms),
                        error_type = coalesce(rag_execution_spans.error_type, excluded.error_type),
                        metadata = rag_execution_spans.metadata || excluded.metadata,
                        updated_at = now()
                    """
                ),
                payload,
            )

    def upsert_model_invocation(self, record: ModelInvocationRecord) -> None:
        payload = record.model_dump(mode="python")
        payload["metadata"] = json.dumps(record.metadata)
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    insert into model_invocations (
                        id, workspace_id, generation_run_id, span_id, component,
                        operation, operation_version, attempt, usage_kind, provider,
                        model, stream, status, prompt_tokens, completion_tokens,
                        cached_input_tokens, reasoning_tokens, usage_source, finish_reason,
                        provider_request_id, error_type, started_at, finished_at, elapsed_ms, metadata
                    ) values (
                        :id, :workspace_id, :generation_run_id, :span_id, :component,
                        :operation, :operation_version, :attempt, :usage_kind, :provider,
                        :model, :stream, :status, :prompt_tokens, :completion_tokens,
                        :cached_input_tokens, :reasoning_tokens, :usage_source, :finish_reason,
                        :provider_request_id, :error_type, :started_at, :finished_at, :elapsed_ms, cast(:metadata as jsonb)
                    )
                    on conflict (id) do update set
                        status = case
                            when model_invocations.status <> 'running' then model_invocations.status
                            else excluded.status
                        end,
                        model = coalesce(model_invocations.model, excluded.model),
                        prompt_tokens = coalesce(model_invocations.prompt_tokens, excluded.prompt_tokens),
                        completion_tokens = coalesce(model_invocations.completion_tokens, excluded.completion_tokens),
                        cached_input_tokens = coalesce(model_invocations.cached_input_tokens, excluded.cached_input_tokens),
                        reasoning_tokens = coalesce(model_invocations.reasoning_tokens, excluded.reasoning_tokens),
                        usage_source = case
                            when model_invocations.usage_source <> 'unavailable' then model_invocations.usage_source
                            else excluded.usage_source
                        end,
                        finish_reason = coalesce(model_invocations.finish_reason, excluded.finish_reason),
                        provider_request_id = coalesce(model_invocations.provider_request_id, excluded.provider_request_id),
                        error_type = coalesce(model_invocations.error_type, excluded.error_type),
                        finished_at = coalesce(model_invocations.finished_at, excluded.finished_at),
                        elapsed_ms = coalesce(model_invocations.elapsed_ms, excluded.elapsed_ms),
                        metadata = model_invocations.metadata || excluded.metadata,
                        updated_at = now()
                    """
                ),
                payload,
            )

    def overview(self, workspace_id: UUID, start: datetime, end: datetime) -> dict[str, object]:
        with self._engine.connect() as connection:
            row = (
                connection.execute(
                    text(
                        """
                        with completed_runs as (
                            select distinct generation_run_id
                            from rag_execution_spans
                            where workspace_id = :workspace_id
                              and operation = 'answer'
                              and status = 'succeeded'
                              and started_at >= :start and started_at < :end
                        )
                        select
                            count(distinct invocations.generation_run_id)::integer as run_count,
                            count(*)::integer as invocation_count,
                            count(*) filter (where invocations.status = 'failed')::integer
                                as failed_invocation_count,
                            coalesce(sum(invocations.prompt_tokens), 0)::bigint as prompt_tokens,
                            coalesce(sum(invocations.completion_tokens), 0)::bigint as completion_tokens,
                            avg(invocations.elapsed_ms) filter (where invocations.status <> 'running')
                                as average_invocation_elapsed_ms
                        from model_invocations invocations
                        join completed_runs using (generation_run_id)
                        where invocations.workspace_id = :workspace_id
                        """
                    ),
                    {"workspace_id": workspace_id, "start": start, "end": end},
                )
                .mappings()
                .one()
            )
            token_p90_rows = list(
                connection.execute(
                    text(
                        """
                    with completed_runs as (
                        select distinct generation_run_id
                        from rag_execution_spans
                        where workspace_id = :workspace_id
                          and operation = 'answer'
                          and status = 'succeeded'
                          and started_at >= :start and started_at < :end
                    ),
                    usage_by_run_and_operation as (
                        select
                            invocations.generation_run_id,
                            coalesce(spans.operation, invocations.operation) as operation,
                            count(*)::integer as invocation_count,
                            coalesce(sum(invocations.prompt_tokens), 0)::bigint as prompt_tokens,
                            coalesce(sum(invocations.completion_tokens), 0)::bigint as completion_tokens
                        from model_invocations invocations
                        join completed_runs using (generation_run_id)
                        left join rag_execution_spans spans
                          on spans.id = invocations.span_id
                         and spans.workspace_id = :workspace_id
                        where invocations.workspace_id = :workspace_id
                          and (
                              invocations.prompt_tokens is not null
                              or invocations.completion_tokens is not null
                          )
                        group by
                            invocations.generation_run_id,
                            coalesce(spans.operation, invocations.operation)
                    )
                    select
                        operation,
                        count(*)::integer as sample_run_count,
                        sum(invocation_count)::integer as invocation_count,
                        (percentile_disc(0.9) within group (order by prompt_tokens))::bigint
                            as prompt_tokens_p90,
                        (percentile_disc(0.9) within group (order by completion_tokens))::bigint
                            as completion_tokens_p90,
                        (percentile_disc(0.9) within group (
                            order by prompt_tokens + completion_tokens
                        ))::bigint as total_tokens_p90
                    from usage_by_run_and_operation
                    group by operation
                    order by operation
                        """
                    ),
                    {"workspace_id": workspace_id, "start": start, "end": end},
                ).mappings()
            )
        result = dict(row)
        result["total_tokens"] = cast(int, result["prompt_tokens"]) + cast(int, result["completion_tokens"])
        result["token_p90_by_operation"] = [dict(token_p90_row) for token_p90_row in token_p90_rows]
        result["start"] = start
        result["end"] = end
        return cast(dict[str, object], result)

    def list_runs(
        self,
        workspace_id: UUID,
        user_id: UUID,
        start: datetime,
        end: datetime,
        limit: int,
        offset: int,
    ) -> list[dict[str, object]]:
        with self._engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    with completed_runs as (
                        select distinct generation_run_id
                        from rag_execution_spans
                        where workspace_id = :workspace_id
                          and operation = 'answer'
                          and status = 'succeeded'
                          and started_at >= :start and started_at < :end
                    ),
                    runs as (
                        select
                            invocations.generation_run_id,
                            min(invocations.started_at) as started_at,
                            max(invocations.finished_at) as finished_at,
                            count(*)::integer as invocation_count,
                            count(*) filter (where invocations.status = 'failed')::integer
                                as failed_invocation_count,
                            coalesce(sum(invocations.prompt_tokens), 0)::bigint as prompt_tokens,
                            coalesce(sum(invocations.completion_tokens), 0)::bigint as completion_tokens,
                            case
                                when bool_or(invocations.status = 'failed') then 'failed'
                                when bool_or(invocations.status = 'cancelled') then 'cancelled'
                                when bool_or(invocations.status = 'running') then 'running'
                                when bool_or(invocations.status = 'abandoned') then 'abandoned'
                                else 'succeeded'
                            end as run_status
                        from model_invocations invocations
                        join completed_runs using (generation_run_id)
                        where invocations.workspace_id = :workspace_id
                        group by invocations.generation_run_id
                    )
                    select runs.*,
                        conversation.id as conversation_id,
                        coalesce(
                            conversation.owner_user_id = :user_id and conversation.archived_at is null,
                            false
                        ) as conversation_navigable,
                        coalesce(
                            conversation.id is not null
                                and (conversation.owner_user_id is null or conversation.archived_at is not null),
                            false
                        ) as conversation_deleted
                    from runs
                    left join lateral (
                            select conversations.id, conversations.owner_user_id, conversations.archived_at
                            from conversation_messages messages
                            join conversations on conversations.id = messages.conversation_id
                            where messages.generation_run_id = runs.generation_run_id
                              and conversations.workspace_id = :workspace_id
                            limit 1
                        ) conversation on true
                    order by runs.started_at desc
                    limit :limit offset :offset
                    """
                ),
                {
                    "workspace_id": workspace_id,
                    "user_id": user_id,
                    "start": start,
                    "end": end,
                    "limit": limit,
                    "offset": offset,
                },
            ).mappings()
            return [self._run_row(row) for row in rows]

    def run_detail(self, workspace_id: UUID, run_id: UUID) -> dict[str, object] | None:
        with self._engine.connect() as connection:
            spans = list(
                connection.execute(
                    text(
                        """
                        select * from rag_execution_spans
                        where workspace_id = :workspace_id and generation_run_id = :run_id
                        order by started_at, id
                        """
                    ),
                    {"workspace_id": workspace_id, "run_id": run_id},
                ).mappings()
            )
            invocations = list(
                connection.execute(
                    text(
                        """
                        select * from model_invocations
                        where workspace_id = :workspace_id and generation_run_id = :run_id
                        order by started_at, id
                        """
                    ),
                    {"workspace_id": workspace_id, "run_id": run_id},
                ).mappings()
            )
        if not spans and not invocations:
            return None
        return {
            "generation_run_id": run_id,
            "spans": [dict(row) for row in spans],
            "model_invocations": [dict(row) for row in invocations],
        }

    def run_conversation(self, workspace_id: UUID, run_id: UUID) -> dict[str, object] | None:
        with self._engine.connect() as connection:
            conversation = (
                connection.execute(
                    text(
                        """
                        select conversations.id, conversations.title, conversations.owner_user_id,
                               conversations.archived_at, conversations.created_at, conversations.updated_at
                        from conversation_messages trigger_message
                        join conversations on conversations.id = trigger_message.conversation_id
                        where trigger_message.generation_run_id = :run_id
                          and conversations.workspace_id = :workspace_id
                        limit 1
                        """
                    ),
                    {"workspace_id": workspace_id, "run_id": run_id},
                )
                .mappings()
                .one_or_none()
            )
            if conversation is None:
                return None
            messages = list(
                connection.execute(
                    text(
                        """
                        select id, conversation_id, role, sequence, content_blocks, sources,
                               generation_run_id, status, error_message, created_at, updated_at
                        from conversation_messages
                        where conversation_id = :conversation_id
                        order by sequence
                        """
                    ),
                    {"conversation_id": conversation["id"]},
                ).mappings()
            )
        conversation_data = dict(conversation)
        conversation_data["deleted"] = conversation_data["owner_user_id"] is None or conversation_data["archived_at"] is not None
        return {
            "conversation": conversation_data,
            "messages": [dict(row) for row in messages],
        }

    @staticmethod
    def _run_row(row: RowMapping) -> dict[str, object]:
        result = dict(row)
        result["total_tokens"] = cast(int, result["prompt_tokens"]) + cast(int, result["completion_tokens"])
        result["status"] = result.pop("run_status")
        return cast(dict[str, object], result)
