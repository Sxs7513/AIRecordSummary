"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { FormEvent, useEffect, useRef, useState } from "react";

import { createConversation, deleteConversation, getMessages, listConversations, sendMessage } from "@/app/sdk/conversations/client";
import { useConversationStore } from "@/app/sdk/conversations/store";
import type { ConversationMessage } from "@/app/sdk/conversations/types";
import { MarkdownContent } from "@/components/markdown-content";
import { GenerationStreamClient } from "@/app/sdk/generation/client";
import { useGenerationStore } from "@/app/sdk/generation/store";

export function ConversationChat({ conversationId }: { conversationId?: string }) {
  const router = useRouter();
  const client = useRef<GenerationStreamClient | null>(null);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const { conversations, messagesByConversation, setConversations, removeConversation, hydrateMessages, reconcileTurn } = useConversationStore();
  if (client.current === null) client.current = new GenerationStreamClient();

  useEffect(() => { void listConversations().then(setConversations).catch((reason) => setError(String(reason))); }, [setConversations]);
  useEffect(() => {
    if (!conversationId) return;
    void getMessages(conversationId).then((page) => {
      hydrateMessages(conversationId, page);
      for (const message of page.items) if (message.generation_run_id && ["pending", "streaming"].includes(message.status)) client.current?.connect(message.generation_run_id);
    }).catch((reason) => setError(String(reason)));
  }, [conversationId, hydrateMessages]);

  async function submit(event: FormEvent) {
    event.preventDefault(); const text = draft.trim(); if (!text) return;
    try {
      setError(null);
      const created = conversationId ? null : await createConversation();
      if (created !== null) setConversations([created, ...conversations]);
      const id = conversationId ?? created!.id;
      const turn = await sendMessage(id, text, crypto.randomUUID());
      reconcileTurn(turn); setDraft(""); client.current?.connect(turn.generation_run_id);
      if (!conversationId) router.replace(`/chat/${id}`);
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
  }

  async function remove(id: string) {
    if (!window.confirm("删除后会永久清除该对话及其消息，确定删除吗？")) return;
    try {
      setError(null);
      await deleteConversation(id);
      removeConversation(id);
      if (id === conversationId) router.replace("/chat");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  const messages = conversationId ? messagesByConversation[conversationId] ?? [] : [];
  return <div className="chat-page"><header className="chat-header"><Link className="button secondary" href="/recordings">返回录音管理</Link></header><div className={`chat-layout ${conversations.length === 0 ? "without-conversation-list" : ""}`}>{conversations.length > 0 ? <aside className="conversation-list"><Link className="button" href="/chat">新建对话</Link>{conversations.map((item) => <div className="conversation-list-item" key={item.id}><Link className={item.id === conversationId ? "selected" : ""} href={`/chat/${item.id}`}>{item.title}</Link><button aria-label={`删除对话：${item.title}`} className="conversation-delete" onClick={() => void remove(item.id)} type="button">删除</button></div>)}</aside> : null}<section className="chat-panel"><div className="message-list">{conversationId ? messages.map((message) => <MessageItem key={message.id} message={message} />) : <p className="chat-empty">开始一个新对话，向已处理的录音提问。</p>}</div>{error ? <p className="chat-error">{error}</p> : null}<form className="chat-composer" onSubmit={submit}><textarea value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="输入一个关于录音的问题…" rows={3} /><button type="submit">发送</button></form></section></div></div>;
}

function MessageItem({ message }: { message: ConversationMessage }) {
  const run = useGenerationStore((state) => message.generation_run_id ? state.runs[message.generation_run_id] : undefined);
  const [sourcesExpanded, setSourcesExpanded] = useState(false);
  const blocks = run?.blocks ?? message.content_blocks; const text = blocks.map((block) => block.value).join("");
  const status = run?.status ?? message.status; const sources = run?.sources ?? message.sources;
  const streaming = status === "streaming" || status === "running";
  const sourceLinks = sources.map(toSourceLink).filter((source): source is SourceLink => source !== null);
  return <article className={`chat-message ${message.role}`}><div className="message-bubble">{message.role === "assistant" ? <>{!text ? <span className="subtle">{run?.phase?.label ?? "正在检索录音资料…"}</span> : null}<MarkdownContent className="chat-markdown" markdown={text} streaming={streaming} /></> : text}</div>{message.role === "assistant" && status === "failed" ? <p className="chat-error">{run?.error?.message ?? message.error_message ?? "回答生成失败"}</p> : null}{message.role === "assistant" && sourceLinks.length > 0 ? <div className="message-sources"><button aria-expanded={sourcesExpanded} className="message-sources-toggle" onClick={() => setSourcesExpanded((expanded) => !expanded)} type="button">引用了 {sourceLinks.length} 条录音资料</button>{sourcesExpanded ? <ul className="message-source-list">{sourceLinks.map((source) => <li key={`${source.recordingId}-${source.href}`}><a href={source.href} rel="noreferrer" target="_blank">{source.title}{source.timeRange ? <span>{source.timeRange}</span> : null}</a></li>)}</ul> : null}</div> : null}</article>;
}

type SourceLink = { recordingId: string; title: string; href: string; timeRange: string | null };

function toSourceLink(source: Record<string, unknown>): SourceLink | null {
  const recording = source.recording;
  if (typeof recording !== "object" || recording === null || Array.isArray(recording)) return null;
  const data = recording as Record<string, unknown>;
  const recordingId = data.id;
  const title = data.title;
  if (typeof recordingId !== "string" || typeof title !== "string") return null;
  const url = typeof source.url === "string" && source.url.startsWith("/recordings/") ? source.url : `/recordings/${recordingId}`;
  const chunk = source.chunk;
  const chunkData = typeof chunk === "object" && chunk !== null && !Array.isArray(chunk) ? chunk as Record<string, unknown> : null;
  const startMs = typeof chunkData?.startMs === "number" ? chunkData.startMs : null;
  const endMs = typeof chunkData?.endMs === "number" ? chunkData.endMs : null;
  return { recordingId, title, href: url, timeRange: startMs === null || endMs === null ? null : `${formatTime(startMs)}–${formatTime(endMs)}` };
}

function formatTime(milliseconds: number): string {
  const totalSeconds = Math.max(0, Math.floor(milliseconds / 1_000));
  return `${Math.floor(totalSeconds / 60)}:${String(totalSeconds % 60).padStart(2, "0")}`;
}
