"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { FormEvent, useEffect, useState } from "react";

import { deleteConversation, getMessages, listConversations, restartGeneration, sendMessage } from "@/app/sdk/conversations/client";
import { useConversationStore } from "@/app/sdk/conversations/store";
import type { Conversation, ConversationMessage, ConversationTurn } from "@/app/sdk/conversations/types";
import { MarkdownContent } from "@/components/markdown-content";
import { GenerationStreamClient } from "@/app/sdk/generation/client";
import { useGenerationStore } from "@/app/sdk/generation/store";
import type { GenerationEvent } from "@/app/sdk/generation/types";

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

  return <div className="chat-page"><header className="chat-header"><Link className="button secondary" href="/recordings">返回录音管理</Link></header><div className={`chat-layout ${conversations.length === 0 ? "without-conversation-list" : ""}`}>{conversations.length > 0 ? <aside className="conversation-list"><Link className="button" href="/chat" onClick={() => setActiveConversation(null)}>新建对话</Link>{conversations.map((item) => <div className="conversation-list-item" key={item.id}>{item.id.startsWith("temporary:") ? <span className="selected">{item.title}</span> : <Link className={item.id === effectiveConversationId ? "selected" : ""} href={`/chat/${item.id}`}>{item.title}</Link>}<button aria-label={`删除对话：${item.title}`} className="conversation-delete" disabled={item.id.startsWith("temporary:")} onClick={() => void remove(item.id)} type="button">删除</button></div>)}</aside> : null}<section className="chat-panel"><div className="message-list">{effectiveConversationId ? messages.map((message) => <MessageItem isLatestAssistant={message.id === latestAssistant?.id} key={message.id} message={message} />) : <p className="chat-empty">开始一个新对话，向已处理的录音提问。</p>}</div>{error ? <p className="chat-error">{error}</p> : null}<form className="chat-composer" onSubmit={submit}><textarea disabled={creatingConversation || generationActive || submitting} value={draft} onChange={(event) => setDraft(event.target.value)} placeholder={generationActive ? "请先停止当前回答，再发送新消息" : "输入一个关于录音的问题…"} rows={3} />{generationActive ? <button disabled={stopping || latestAssistant?.generation_run_id === null} onClick={() => void stopLatestGeneration()} type="button">{stopping ? "正在停止…" : "停止生成"}</button> : <button disabled={creatingConversation || submitting || !draft.trim()} type="submit">{submitting ? "正在发送…" : "发送"}</button>}</form></section></div></div>;
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
  const blocks = run?.blocks ?? message.content_blocks; const text = blocks.map((block) => block.value).join("");
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
  return <article className={`chat-message ${message.role}`}><div className="message-bubble">{message.role === "assistant" ? <>{!text ? <span className="subtle">{placeholder}</span> : null}<MarkdownContent className="chat-markdown" markdown={text} streaming={streaming} citations={sourceLinks} /></> : text}</div>{canRestart ? <div className="generation-actions"><button disabled={restarting} onClick={() => void restart()} type="button">{restarting ? resumingLabel : resumeLabel}</button></div> : null}{stopError ? <p className="chat-error">{stopError}</p> : null}{message.role === "assistant" && status === "failed" ? <p className="chat-error">{run?.error?.message ?? message.error_message ?? "回答生成失败"}</p> : null}{message.role === "assistant" && sourceLinks.length > 0 ? <div className="message-sources"><button aria-expanded={sourcesExpanded} className="message-sources-toggle" onClick={() => setSourcesExpanded((expanded) => !expanded)} type="button">引用了 {sourceLinks.length} 条录音资料</button>{sourcesExpanded ? <ul className="message-source-list">{sourceLinks.map((source) => <li key={`${source.index}-${source.recordingId}-${source.href}`}><a href={source.href} rel="noreferrer" target="_blank"><span className="message-source-index">[{source.index}]</span>{source.title}{source.timeRange ? <span>{source.timeRange}</span> : null}</a></li>)}</ul> : null}</div> : null}</article>;
}

type SourceLink = { index: number; recordingId: string; title: string; href: string; timeRange: string | null };

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
