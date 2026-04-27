"use client";

import { useState } from "react";
import { EvidenceList } from "./evidence-list";
import type { RagAnswer, SearchEvidence } from "@/lib/types/models";

type StreamPayload =
  | { event: "evidence"; data: { evidence: SearchEvidence[]; message?: string } }
  | { event: "thinking_start"; data: { text?: string } }
  | { event: "thinking_done"; data: { text?: string } }
  | { event: "answer_delta"; data: { text: string } }
  | { event: "answer_done"; data: { answer: RagAnswer | null } }
  | { event: "error"; data: { message: string } };

function parseSseEvents(buffer: string): { events: StreamPayload[]; rest: string } {
  const events: StreamPayload[] = [];
  const parts = buffer.split("\n\n");
  const rest = parts.pop() ?? "";
  for (const part of parts) {
    const eventLine = part.split(/\r?\n/).find((line) => line.startsWith("event: "));
    const dataLine = part.split(/\r?\n/).find((line) => line.startsWith("data: "));
    if (!eventLine || !dataLine) continue;
    events.push({
      event: eventLine.slice("event: ".length) as StreamPayload["event"],
      data: JSON.parse(dataLine.slice("data: ".length))
    } as StreamPayload);
  }
  return { events, rest };
}

export function RagChat() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [askedQuery, setAskedQuery] = useState("");
  const [answer, setAnswer] = useState<RagAnswer | null>(null);
  const [thinking, setThinking] = useState<{ active: boolean; text: string }>({ active: false, text: "" });
  const [evidence, setEvidence] = useState<SearchEvidence[]>([]);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    const text = query.trim();
    if (!text) return;
    setLoading(true);
    setError(null);
    setAskedQuery(text);
    setAnswer(null);
    setThinking({ active: false, text: "" });
    setEvidence([]);
    try {
      const response = await fetch("/api/rag/query/stream", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ query: text, mode: "answer", limit: 10 })
      });
      if (!response.ok || !response.body) throw new Error("查询失败");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let streamedText = "";
      let done = false;
      while (!done) {
        const next = await reader.read();
        done = next.done;
        buffer += decoder.decode(next.value ?? new Uint8Array(), { stream: !done });
        const parsed = parseSseEvents(buffer);
        buffer = parsed.rest;
        for (const item of parsed.events) {
          if (item.event === "evidence") {
            setEvidence(item.data.evidence);
            setAnswer({ text: "", citations: [], notEnoughEvidence: false });
          }
          if (item.event === "answer_delta") {
            streamedText += item.data.text;
            setAnswer((current) => ({
              text: streamedText,
              citations: current?.citations ?? [],
              notEnoughEvidence: current?.notEnoughEvidence ?? false
            }));
          }
          if (item.event === "thinking_start") {
            setThinking({ active: true, text: "" });
          }
          if (item.event === "thinking_done") {
            setThinking({ active: false, text: item.data.text ?? "" });
          }
          if (item.event === "answer_done") {
            setAnswer(item.data.answer);
            setThinking((current) => ({ ...current, active: false }));
          }
          if (item.event === "error") {
            throw new Error(item.data.message);
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="rag-page">
      <form className="rag-search" onSubmit={submit}>
        <textarea value={query} onChange={(event) => setQuery(event.target.value)} placeholder="输入一个关于录音的问题..." rows={4} />
        <div className="rag-actions">
          <button type="submit" disabled={loading}>
            {loading ? "查询中..." : "提问"}
          </button>
        </div>
      </form>

      {error ? <div className="panel error-panel">{error}</div> : null}

      {askedQuery ? (
        <section className="rag-result">
          <div className="user-question">{askedQuery}</div>
          {answer ? (
            <article className={`assistant-answer ${answer.notEnoughEvidence ? "weak" : ""}`}>
              {thinking.active ? <div className="thinking-indicator">思考中</div> : null}
              {!thinking.active && thinking.text ? (
                <details className="thinking-details">
                  <summary>思考过程</summary>
                  <div>{thinking.text}</div>
                </details>
              ) : null}
              <div className="full-text">{answer.text || "正在生成回答..."}</div>
            </article>
          ) : loading ? (
            <article className="assistant-answer">正在检索录音片段...</article>
          ) : null}
          <EvidenceList evidence={evidence} />
        </section>
      ) : null}
    </div>
  );
}
