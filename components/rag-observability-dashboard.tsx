"use client";

import Link from "next/link";
import { CircleAlert } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import {
  getObservabilityOverview,
  getObservabilityRun,
  getObservabilityRunConversation,
  listObservabilityRuns
} from "@/app/sdk/observability/client";
import type {
  ModelInvocation,
  ObservabilityConversationSnapshot,
  ObservabilityOverview,
  ObservabilityRun,
  ObservabilityRunDetail,
  ObservabilityStatus
} from "@/app/sdk/observability/types";
import { MarkdownContent } from "@/components/markdown-content";
import { selectContentBlocksText } from "@/app/sdk/generation/selectors";

const number = new Intl.NumberFormat("zh-CN");

function statusLabel(status: ObservabilityStatus) {
  return { running: "运行中", succeeded: "成功", failed: "失败", cancelled: "已取消", abandoned: "已中断" }[status];
}

function duration(value: number | null) {
  if (value === null) return "—";
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`;
}

function dateTime(value: string | null) {
  return value ? new Date(value).toLocaleString("zh-CN") : "—";
}

function tokenTotal(invocation: ModelInvocation) {
  if (invocation.prompt_tokens === null && invocation.completion_tokens === null) return "usage 不可用";
  return number.format((invocation.prompt_tokens ?? 0) + (invocation.completion_tokens ?? 0));
}

function operationLabel(operation: string) {
  const labels: Record<string, string> = {
    answer: "Answer",
    grade: "Grade",
    plan: "Plan",
    rerank: "Rerank",
    rewrite: "Rewrite",
    route: "Route"
  };
  return labels[operation] ?? operation;
}

function formatMetadata(metadata: Record<string, unknown>) {
  try {
    return JSON.stringify(metadata, null, 2);
  } catch {
    return "{\n  \"error\": \"metadata 无法序列化为 JSON\"\n}";
  }
}

export function RagObservabilityDashboard() {
  const [overview, setOverview] = useState<ObservabilityOverview | null>(null);
  const [runs, setRuns] = useState<ObservabilityRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ObservabilityRunDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [conversationSnapshot, setConversationSnapshot] = useState<ObservabilityConversationSnapshot | null>(null);
  const [conversationLoading, setConversationLoading] = useState(false);
  const [conversationError, setConversationError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [nextOverview, nextRuns] = await Promise.all([getObservabilityOverview(), listObservabilityRuns()]);
      setOverview(nextOverview);
      setRuns(nextRuns);
      const nextSelected = nextRuns[0]?.generation_run_id ?? null;
      setSelectedRunId(nextSelected);
      setDetail(nextSelected ? await getObservabilityRun(nextSelected) : null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "监控数据加载失败");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  const selectRun = async (runId: string) => {
    setSelectedRunId(runId);
    setError(null);
    try {
      setDetail(await getObservabilityRun(runId));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Run 详情加载失败");
    }
  };

  const openConversationSnapshot = async (runId: string) => {
    setConversationSnapshot(null);
    setConversationError(null);
    setConversationLoading(true);
    try {
      setConversationSnapshot(await getObservabilityRunConversation(runId));
    } catch (cause) {
      setConversationError(cause instanceof Error ? cause.message : "对话快照加载失败");
    } finally {
      setConversationLoading(false);
    }
  };

  const closeConversationSnapshot = () => {
    setConversationSnapshot(null);
    setConversationError(null);
    setConversationLoading(false);
  };

  return (
    <div className="observability-page grid">
      <div className="topbar">
        <div><h1>RAG 运行监控</h1><p className="subtle">当前工作区最近 7 天已完成 Answer 的节点耗时和模型 Token。此页面不保存 Prompt 或录音正文。</p></div>
        <button type="button" onClick={() => void load()} disabled={loading}>{loading ? "刷新中…" : "刷新"}</button>
      </div>

      {error ? <div className="panel observability-error">{error}</div> : null}

      <div className="stats grid">
        <div className="card stat"><span className="subtle">RAG Runs</span><strong>{number.format(overview?.run_count ?? 0)}</strong></div>
        <div className="card stat"><span className="subtle">模型调用</span><strong>{number.format(overview?.invocation_count ?? 0)}</strong></div>
        <div className="card stat"><span className="subtle">总 Token</span><strong>{number.format(overview?.total_tokens ?? 0)}</strong></div>
        <div className="card stat"><span className="subtle">平均模型耗时</span><strong>{duration(overview?.average_invocation_elapsed_ms ?? null)}</strong></div>
      </div>

      <section className="panel observability-p90">
        <div>
          <h2>节点 Token P90</h2>
          <p className="subtle">仅统计 Answer 成功完成的 Run；每个样本为单个 Run 在该节点上的 Token 总和。</p>
        </div>
        <div className="observability-table-scroll">
          <table className="observability-p90-table">
            <thead>
              <tr>
                <th>节点</th>
                <th>样本 Run</th>
                <th>模型调用</th>
                <th>输入 Token P90</th>
                <th>输出 Token P90</th>
                <th>总 Token P90</th>
              </tr>
            </thead>
            <tbody>
              {(overview?.token_p90_by_operation ?? []).map((row) => (
                <tr key={row.operation}>
                  <td><strong>{operationLabel(row.operation)}</strong></td>
                  <td>{number.format(row.sample_run_count)}</td>
                  <td>{number.format(row.invocation_count)}</td>
                  <td>{number.format(row.prompt_tokens_p90)}</td>
                  <td>{number.format(row.completion_tokens_p90)}</td>
                  <td><strong>{number.format(row.total_tokens_p90)}</strong></td>
                </tr>
              ))}
              {(overview?.token_p90_by_operation.length ?? 0) === 0 ? (
                <tr><td className="subtle observability-empty-cell" colSpan={6}>暂无可计算 P90 的 Token 数据。</td></tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>

      <div className="observability-layout">
        <section className="panel observability-runs">
          <div><h2>Run 列表</h2><p className="subtle">仅展示 Answer 成功完成的 Run。</p></div>
          {runs.length === 0 ? <p className="subtle">还没有可观测数据。</p> : runs.map((run) => (
            <article
              className={run.generation_run_id === selectedRunId ? "observability-run selected" : "observability-run"}
              key={run.generation_run_id}
            >
              <button className="observability-run-select" onClick={() => void selectRun(run.generation_run_id)} type="button">
                <span className={`observability-status ${run.status}`}>{statusLabel(run.status)}</span>
                <strong>{dateTime(run.started_at)}</strong>
                <span>{run.invocation_count} 次调用 · {number.format(run.total_tokens)} Token</span>
                <code>{run.generation_run_id.slice(0, 8)}</code>
              </button>
              {run.conversation_id && run.conversation_navigable ? (
                <Link className="observability-run-chat" href={`/chat/${run.conversation_id}`}>查看对话</Link>
              ) : run.conversation_id ? (
                <button className="observability-run-chat" onClick={() => void openConversationSnapshot(run.generation_run_id)} type="button">
                  {run.conversation_deleted ? "查看已删除对话" : "查看对话快照"}
                </button>
              ) : null}
            </article>
          ))}
        </section>

        <section className="panel observability-detail">
          <h2>执行时间线</h2>
          {!detail ? <p className="subtle">选择一个 Run 查看详情。</p> : (
            <div className="observability-timeline">
              {detail.spans.map((span) => {
                const invocations = detail.model_invocations.filter((item) => item.span_id === span.id);
                return (
                  <article className="observability-span" key={span.id}>
                    <div className="observability-span-head">
                      <div className="observability-span-title">
                        <strong>{span.operation}</strong><span className="subtle"> attempt {span.attempt}</span>
                        <span className="observability-metadata-tooltip">
                          <button aria-label={`查看 ${span.operation} 节点 metadata`} className="observability-metadata-trigger" type="button">
                            <CircleAlert aria-hidden="true" size={16} />
                          </button>
                          <pre className="observability-metadata-json">{formatMetadata(span.metadata)}</pre>
                        </span>
                      </div>
                      <div><span className={`observability-status ${span.status}`}>{statusLabel(span.status)}</span> {duration(span.elapsed_ms)}</div>
                    </div>
                    {span.error_type ? <p className="observability-error">{span.error_type}</p> : null}
                    {invocations.map((invocation) => (
                      <div className="model-invocation" key={invocation.id}>
                        <span>{invocation.provider} / {invocation.model ?? "unknown model"}</span>
                        <span>{tokenTotal(invocation)} Token</span>
                        <span>{duration(invocation.elapsed_ms)}</span>
                        <span className="subtle">{invocation.usage_source}</span>
                      </div>
                    ))}
                  </article>
                );
              })}
            </div>
          )}
        </section>
      </div>

      {conversationLoading || conversationSnapshot || conversationError ? (
        <div className="observability-modal-backdrop" onClick={closeConversationSnapshot}>
          <section
            aria-labelledby="observability-conversation-title"
            aria-modal="true"
            className="observability-conversation-modal"
            onClick={(event) => event.stopPropagation()}
            role="dialog"
          >
            <div className="observability-conversation-head">
              <div>
                <h2 id="observability-conversation-title">{conversationSnapshot?.conversation.title ?? "对话快照"}</h2>
                {conversationSnapshot?.conversation.deleted ? <span className="observability-deleted-badge">已从用户对话列表删除</span> : null}
              </div>
              <button className="secondary" onClick={closeConversationSnapshot} type="button">关闭</button>
            </div>
            {conversationLoading ? <p className="subtle">正在加载对话…</p> : null}
            {conversationError ? <p className="observability-error">{conversationError}</p> : null}
            {conversationSnapshot ? (
              <div className="observability-conversation-messages">
                {conversationSnapshot.messages.map((message) => {
                  const text = selectContentBlocksText(message.content_blocks);
                  return (
                    <article className={`chat-message ${message.role}`} key={message.id}>
                      <div className="message-bubble">
                        {message.role === "assistant" ? <MarkdownContent className="chat-markdown" markdown={text || "（无文本内容）"} /> : text || "（无文本内容）"}
                      </div>
                      {message.error_message ? <p className="chat-error">{message.error_message}</p> : null}
                      <span className="observability-message-meta">{dateTime(message.created_at)} · {message.status}</span>
                    </article>
                  );
                })}
              </div>
            ) : null}
          </section>
        </div>
      ) : null}
    </div>
  );
}
