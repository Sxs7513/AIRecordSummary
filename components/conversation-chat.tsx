"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { FormEvent, useEffect, useState } from "react";

import { deleteConversation, getMessages, listConversations, restartGeneration, sendMessage, submitAdjudicationDecision } from "@/app/sdk/conversations/client";
import { useConversationStore } from "@/app/sdk/conversations/store";
import type { Conversation, ConversationMessage, ConversationTurn } from "@/app/sdk/conversations/types";
import { MarkdownContent } from "@/components/markdown-content";
import { GenerationStreamClient } from "@/app/sdk/generation/client";
import { useGenerationStore } from "@/app/sdk/generation/store";
import type { AdjudicationConfirmationBlock, AggreMessageBlock, GenerationEvent, SubMessage } from "@/app/sdk/generation/types";

const generationClient = new GenerationStreamClient();

export function ConversationChat({ conversationId }: { conversationId?: string }) {
  const router = useRouter();
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const {
    conversations, activeConversationId, messagesByConversation, setConversations, removeConversation,
    setActiveConversation, hydrateMessages, reconcileTurn, createOptimisticTurn, reconcileInitialTurn,
  } = useConversationStore();
  const effectiveConversationId = conversationId ?? activeConversationId;
  const creatingConversation = effectiveConversationId?.startsWith("temporary:") ?? false;
  const messages = effectiveConversationId ? messagesByConversation[effectiveConversationId] ?? [] : [];
  const latestAssistant = [...messages].reverse().find((message) => message.role === "assistant") ?? null;
  const latestRun = useGenerationStore((state) => latestAssistant?.generation_run_id ? state.runs[latestAssistant.generation_run_id] : undefined);
  const latestStatus = latestRun?.status ?? latestAssistant?.status ?? null;
  const latestBlocks = latestRun?.blocks ?? latestAssistant?.content_blocks ?? [];
  const interactionPending = latestBlocks.some((block) => block.type === "adjudication_confirmation");
  const generationActive = latestAssistant !== null && ["pending", "streaming", "queued", "running"].includes(latestStatus ?? "");

  useEffect(() => { void listConversations().then(setConversations).catch((reason) => setError(String(reason))); }, [setConversations]);
  useEffect(() => {
    if (!conversationId) return;
    void getMessages(conversationId).then((page) => {
      hydrateMessages(conversationId, page);
      for (const message of page.items) if (message.generation_run_id && ["pending", "streaming"].includes(message.status)) generationClient.connect(message.generation_run_id);
    }).catch((reason) => setError(String(reason)));
  }, [conversationId, hydrateMessages]);

  async function submit(event: FormEvent) {
    event.preventDefault(); const text = draft.trim(); if (!text || generationActive || submitting) return;
    setSubmitting(true);
    try {
      setError(null);
      const clientMessageId = crypto.randomUUID();
      if (!effectiveConversationId) {
        const temporaryId = `temporary:${crypto.randomUUID()}`;
        createOptimisticTurn(temporaryId, clientMessageId, text);
        setDraft("");
        generationClient.startConversationTurn(
          {
            client_conversation_id: temporaryId.slice("temporary:".length),
            client_message_id: clientMessageId,
            content_blocks: [{ type: "text", value: text }],
            limit: 10,
          },
          (event) => {
            const ready = conversationReady(event);
            if (ready === null) {
              setError("创建对话接口返回了无效数据");
              return;
            }
            setError(null);
            reconcileInitialTurn(temporaryId, ready.conversation, ready.turn);
            router.replace(`/chat/${ready.conversation.id}`, { scroll: false });
          },
          (reason) => { setError(reason.message); setSubmitting(false); },
        );
        return;
      }
      if (creatingConversation) return;
      const turn = await sendMessage(effectiveConversationId, text, clientMessageId);
      reconcileTurn(turn); setDraft(""); generationClient.connect(turn.generation_run_id);
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { if (!creatingConversation) setSubmitting(false); }
  }

  async function stopLatestGeneration() {
    if (latestAssistant?.generation_run_id === null || latestAssistant?.generation_run_id === undefined || stopping) return;
    setStopping(true); setError(null);
    try { await generationClient.cancel(latestAssistant.generation_run_id); }
    catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); setStopping(false); }
  }

  useEffect(() => {
    if (!generationActive) {
      setStopping(false);
      setSubmitting(false);
    }
  }, [generationActive]);

  async function remove(id: string) {
    if (!window.confirm("删除后会将该对话从你的对话列表中移除，确定删除吗？")) return;
    try {
      setError(null);
      await deleteConversation(id);
      removeConversation(id);
      if (id === conversationId) router.replace("/chat");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  return <div className="chat-page"><header className="chat-header"><Link className="button secondary" href="/recordings">返回录音管理</Link></header><div className={`chat-layout ${conversations.length === 0 ? "without-conversation-list" : ""}`}>{conversations.length > 0 ? <aside className="conversation-list"><Link className="button" href="/chat" onClick={() => setActiveConversation(null)}>新建对话</Link>{conversations.map((item) => <div className="conversation-list-item" key={item.id}>{item.id.startsWith("temporary:") ? <span className="selected">{item.title}</span> : <Link className={item.id === effectiveConversationId ? "selected" : ""} href={`/chat/${item.id}`}>{item.title}</Link>}<button aria-label={`删除对话：${item.title}`} className="conversation-delete" disabled={item.id.startsWith("temporary:")} onClick={() => void remove(item.id)} type="button">删除</button></div>)}</aside> : null}<section className="chat-panel"><div className="message-list">{effectiveConversationId ? messages.map((message) => <MessageItem isLatestAssistant={message.id === latestAssistant?.id} key={message.id} message={message} />) : <p className="chat-empty">开始一个新对话，向已处理的录音提问。</p>}</div>{error ? <p className="chat-error">{error}</p> : null}<form className="chat-composer" onSubmit={submit}><textarea disabled={creatingConversation || generationActive || interactionPending || submitting} value={draft} onChange={(event) => setDraft(event.target.value)} placeholder={interactionPending ? "请先处理上方的转写确认" : generationActive ? "请先停止当前回答，再发送新消息" : "输入一个关于录音的问题…"} rows={3} />{generationActive ? <button disabled={stopping || latestAssistant?.generation_run_id === null} onClick={() => void stopLatestGeneration()} type="button">{stopping ? "正在停止…" : "停止生成"}</button> : <button disabled={creatingConversation || interactionPending || submitting || !draft.trim()} type="submit">{interactionPending ? "等待确认" : submitting ? "正在发送…" : "发送"}</button>}</form></section></div></div>;
}

function conversationReady(event: GenerationEvent): { conversation: Conversation; turn: ConversationTurn } | null {
  if (event.type !== "conversation.ready") return null;
  const conversation = event.data.conversation;
  const userMessage = event.data.user_message;
  const assistantMessage = event.data.assistant_message;
  if (!isRecord(conversation) || typeof conversation.id !== "string" || !isRecord(userMessage) || !isRecord(assistantMessage)) return null;
  if (typeof event.data.generation_run_id !== "string") return null;
  return {
    conversation: conversation as Conversation,
    turn: {
      user_message: userMessage as ConversationMessage,
      assistant_message: assistantMessage as ConversationMessage,
      generation_run_id: event.data.generation_run_id,
    },
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function MessageItem({ message, isLatestAssistant }: { message: ConversationMessage; isLatestAssistant: boolean }) {
  const reconcileTurn = useConversationStore((state) => state.reconcileTurn);
  const run = useGenerationStore((state) => message.generation_run_id ? state.runs[message.generation_run_id] : undefined);
  const [sourcesExpanded, setSourcesExpanded] = useState(false);
  const [stopError, setStopError] = useState<string | null>(null);
  const [restarting, setRestarting] = useState(false);
  const blocks = run?.blocks ?? message.content_blocks;
  const text = blocks.filter((block) => block.type === "text").map((block) => block.value).join("");
  const aggregateBlocks = blocks.filter((block): block is AggreMessageBlock => block.type === "AGGRE_MSG");
  const confirmationBlocks = blocks.filter((block): block is AdjudicationConfirmationBlock => block.type === "adjudication_confirmation");
  const status = run?.status ?? message.status; const sources = run?.sources ?? message.sources;
  const streaming = status === "streaming" || status === "running";
  const sourceLinks = sources
    .map(toSourceLink)
    .filter((source): source is SourceLink => source !== null)
    .sort((left, right) => left.index - right.index);
  const placeholder = run?.phase?.label
    ?? (run?.status === "queued" ? "等待生成任务开始…"
      : run?.connection === "reconnecting" ? "正在重新连接生成服务…"
        : run?.status === "failed" || message.status === "failed" ? "回答生成失败"
          : "正在检索录音资料…");
  async function restart() {
    if (message.generation_run_id === null || restarting) return;
    setRestarting(true); setStopError(null);
    try {
      const turn = await restartGeneration(message.conversation_id, message.generation_run_id, "resume");
      reconcileTurn(turn); generationClient.connect(turn.generation_run_id);
    } catch (reason) { setStopError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setRestarting(false); }
  }
  const canRestart = isLatestAssistant && message.generation_run_id !== null && ["cancelled", "failed"].includes(status ?? "");
  const resumeLabel = status === "failed" ? "重试" : "继续生成";
  const resumingLabel = status === "failed" ? "正在重试…" : "正在继续…";
  return <article className={`chat-message ${message.role}`}><div className="message-bubble">{message.role === "assistant" ? <>{!text && aggregateBlocks.length === 0 && confirmationBlocks.length === 0 ? <span className="subtle">{placeholder}</span> : null}{text ? <MarkdownContent className="chat-markdown" markdown={text} streaming={streaming} citations={sourceLinks} /> : null}{aggregateBlocks.map((block) => <AggregateMessageCards block={block} key={block.id} />)}{confirmationBlocks.map((block) => <AdjudicationConfirmationCard block={block} conversationId={message.conversation_id} key={block.request_id} onError={setStopError} onSubmitted={(turn) => { reconcileTurn(turn); generationClient.connect(turn.generation_run_id); }} />)}</> : text}</div>{canRestart ? <div className="generation-actions"><button disabled={restarting} onClick={() => void restart()} type="button">{restarting ? resumingLabel : resumeLabel}</button></div> : null}{stopError ? <p className="chat-error">{stopError}</p> : null}{message.role === "assistant" && status === "failed" ? <p className="chat-error">{run?.error?.message ?? message.error_message ?? "回答生成失败"}</p> : null}{message.role === "assistant" && aggregateBlocks.length === 0 && sourceLinks.length > 0 ? <div className="message-sources"><button aria-expanded={sourcesExpanded} className="message-sources-toggle" onClick={() => setSourcesExpanded((expanded) => !expanded)} type="button">引用了 {sourceLinks.length} 条录音资料</button>{sourcesExpanded ? <ul className="message-source-list">{sourceLinks.map((source) => <li key={`${source.index}-${source.recordingId}-${source.href}`}><a href={source.href} rel="noreferrer" target="_blank"><span className="message-source-index">[{source.index}]</span>{source.title}{source.timeRange ? <span>{source.timeRange}</span> : null}</a></li>)}</ul> : null}</div> : null}</article>;
}

function AggregateMessageCards({ block }: { block: AggreMessageBlock }) {
  const group = block.sub_message.message_group;
  const messages = new Map(block.sub_message.sub_message_list.map((item) => [item.id, item]));
  return <section className="aggregate-message-grid">{group.sub_message_ids.map((id) => {
    const message = messages.get(id);
    if (!message) return null;
    return <AggregateMessageCard isPrimary={id === group.primary_sub_message_id} key={id} message={message} />;
  })}</section>;
}

function AggregateMessageCard({ message, isPrimary }: { message: SubMessage; isPrimary: boolean }) {
  const [sourcesExpanded, setSourcesExpanded] = useState(false);
  const text = message.blocks.map((block) => block.value).join("");
  const citations = message.sources.map(toSourceLink).filter((source): source is SourceLink => source !== null);
  const corrections = isPrimary ? message.sources.flatMap(toCorrectionLinks) : [];
  const streaming = message.status === "pending" || message.status === "streaming";
  return <section className={`aggregate-message-card ${isPrimary ? "primary" : ""}`}><header><h3>{message.title}</h3>{isPrimary ? <span>推荐</span> : null}</header>{corrections.length > 0 ? <div className="aggregate-corrections">{corrections.map((correction) => <a href={correction.href} key={`${correction.proposalId}-${correction.href}`} rel="noreferrer" target="_blank">{correction.originalExpression} → {correction.resolvedExpression}</a>)}</div> : null}{text ? <MarkdownContent className="chat-markdown" markdown={text} streaming={streaming} citations={citations} /> : message.status === "failed" ? <p className="chat-error">{message.error ?? "该版本生成失败"}</p> : <p className="subtle">正在生成…</p>}{message.status === "failed" && text ? <p className="chat-error">{message.error ?? "该版本生成失败"}</p> : null}{citations.length > 0 ? <div className="message-sources"><button aria-expanded={sourcesExpanded} className="message-sources-toggle" onClick={() => setSourcesExpanded((expanded) => !expanded)} type="button">引用了 {citations.length} 条录音资料</button>{sourcesExpanded ? <ul className="message-source-list">{citations.map((source) => <li key={`${source.index}-${source.recordingId}-${source.href}`}><a href={source.href} rel="noreferrer" target="_blank"><span className="message-source-index">[{source.index}]</span>{source.title}{source.timeRange ? <span>{source.timeRange}</span> : null}</a></li>)}</ul> : null}</div> : null}</section>;
}

function AdjudicationConfirmationCard({ block, conversationId, onSubmitted, onError }: { block: AdjudicationConfirmationBlock; conversationId: string; onSubmitted: (turn: ConversationTurn) => void; onError: (message: string | null) => void }) {
  const [choices, setChoices] = useState<Record<string, string>>({});
  const [submitting, setSubmitting] = useState(false);
  const ready = block.items.every((item) => Boolean(choices[item.id]));
  async function submitDecision() {
    if (!ready || submitting) return;
    setSubmitting(true); onError(null);
    try {
      const decisions = block.items.map((item) => {
        const choice = choices[item.id];
        if (choice.startsWith("candidate:")) return { item_id: item.id, action: "accept_candidate" as const, candidate_id: choice.slice("candidate:".length) };
        return { item_id: item.id, action: choice === "keep" ? "keep_original" as const : "unresolved" as const };
      });
      onSubmitted(await submitAdjudicationDecision(conversationId, block.source_generation_id, block.request_id, decisions));
    } catch (reason) { onError(reason instanceof Error ? reason.message : String(reason)); setSubmitting(false); }
  }
  return <section className="adjudication-card"><h3>需要确认录音转写</h3><p>系统发现关键表达可能存在识别错误。请选择后再继续回答。</p>{block.items.map((item) => <fieldset key={item.id}><legend>原转写：{item.original_expression}</legend><a href={`/recordings/${encodeURIComponent(item.recording_id)}?t=${item.start_ms}&end=${item.end_ms}&highlight=${encodeURIComponent(item.original_expression)}`} rel="noreferrer" target="_blank">播放并高亮原词</a>{item.candidates.map((candidate) => <label key={candidate.id}><input checked={choices[item.id] === `candidate:${candidate.id}`} name={`adjudication-${item.id}`} onChange={() => setChoices((value) => ({ ...value, [item.id]: `candidate:${candidate.id}` }))} type="radio" />使用建议：{candidate.expression}</label>)}<label><input checked={choices[item.id] === "keep"} name={`adjudication-${item.id}`} onChange={() => setChoices((value) => ({ ...value, [item.id]: "keep" }))} type="radio" />保留原转写</label><label><input checked={choices[item.id] === "unresolved"} name={`adjudication-${item.id}`} onChange={() => setChoices((value) => ({ ...value, [item.id]: "unresolved" }))} type="radio" />暂不确认</label>{item.reason ? <small>{item.reason}</small> : null}</fieldset>)}<button disabled={!ready || submitting} onClick={() => void submitDecision()} type="button">{submitting ? "正在继续…" : "确认并继续回答"}</button></section>;
}

type SourceLink = { index: number; recordingId: string; title: string; href: string; timeRange: string | null };
type CorrectionLink = { proposalId: string; originalExpression: string; resolvedExpression: string; href: string };

function toCorrectionLinks(source: Record<string, unknown>): CorrectionLink[] {
  const recording = isRecord(source.recording) ? source.recording : null;
  const chunk = isRecord(source.chunk) ? source.chunk : null;
  const recordingId = recording?.id;
  const adjudication = source.adjudication;
  if (typeof recordingId !== "string" || !Array.isArray(adjudication)) return [];
  const startMs = typeof chunk?.startMs === "number" ? chunk.startMs : null;
  const endMs = typeof chunk?.endMs === "number" ? chunk.endMs : null;
  return adjudication.flatMap((value) => {
    if (!isRecord(value)) return [];
    const proposalId = value.proposal_id;
    const originalExpression = value.original_expression;
    const resolvedExpression = value.resolved_expression;
    if (typeof proposalId !== "string" || typeof originalExpression !== "string" || typeof resolvedExpression !== "string") return [];
    const parameters = new URLSearchParams({ highlight: originalExpression });
    if (startMs !== null) parameters.set("t", String(startMs));
    if (endMs !== null) parameters.set("end", String(endMs));
    return [{
      proposalId,
      originalExpression,
      resolvedExpression,
      href: `/recordings/${encodeURIComponent(recordingId)}?${parameters.toString()}`,
    }];
  });
}

function toSourceLink(source: Record<string, unknown>): SourceLink | null {
  const index = source.index;
  const recording = source.recording;
  if (typeof recording !== "object" || recording === null || Array.isArray(recording)) return null;
  const data = recording as Record<string, unknown>;
  const recordingId = data.id;
  const title = typeof data.title === "string" ? data.title : data.fileName;
  if (typeof index !== "number" || !Number.isInteger(index) || typeof recordingId !== "string" || typeof title !== "string") return null;
  const chunk = source.chunk;
  const chunkData = typeof chunk === "object" && chunk !== null && !Array.isArray(chunk) ? chunk as Record<string, unknown> : null;
  const startMs = typeof chunkData?.startMs === "number" ? chunkData.startMs : null;
  const endMs = typeof chunkData?.endMs === "number" ? chunkData.endMs : null;
  const url = typeof source.url === "string" && source.url.startsWith("/recordings/") ? source.url : `/recordings/${recordingId}`;
  const href = withCitationRange(url, startMs, endMs);
  return { index, recordingId, title, href, timeRange: startMs === null || endMs === null ? null : `${formatTime(startMs)}–${formatTime(endMs)}` };
}

function withCitationRange(url: string, startMs: number | null, endMs: number | null): string {
  if (startMs === null || endMs === null) return url;
  const [path, query = ""] = url.split("?", 2);
  const parameters = new URLSearchParams(query);
  parameters.set("t", String(startMs));
  parameters.set("end", String(endMs));
  return `${path}?${parameters.toString()}`;
}

function formatTime(milliseconds: number): string {
  const totalSeconds = Math.max(0, Math.floor(milliseconds / 1_000));
  return `${Math.floor(totalSeconds / 60)}:${String(totalSeconds % 60).padStart(2, "0")}`;
}
