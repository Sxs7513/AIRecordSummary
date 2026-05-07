"use client";

import { useMemo, useState } from "react";
import { EvidenceList } from "./evidence-list";
import type { RagAnswer, RagQueryResponse, SearchEvidence } from "@/lib/types/models";
import type { ReactNode } from "react";

function AnswerText({
  text,
  evidence,
  onCitationClick
}: {
  text: string;
  evidence: SearchEvidence[];
  onCitationClick: (index: number) => void;
}) {
  const evidenceByIndex = useMemo(() => new Map(evidence.map((item) => [item.index, item])), [evidence]);
  const parts: ReactNode[] = [];
  const citationPattern = /(?:\[|【|［)([\d\s,，、]+)(?:\]|】|］)/g;
  let cursor = 0;
  let match: RegExpExecArray | null;

  while ((match = citationPattern.exec(text)) !== null) {
    if (match.index > cursor) parts.push(text.slice(cursor, match.index));
    const indexes = match[1]
      .split(/[\s,，、]+/)
      .map((value) => Number(value))
      .filter((value) => Number.isInteger(value));
    if (indexes.length === 0) {
      parts.push(match[0]);
    }
    for (const index of indexes) {
      const item = evidenceByIndex.get(index);
      if (item) {
        parts.push(
          <a
            className="citation-link"
            href={item.url}
            key={`${match.index}-${index}`}
            target="_blank"
            rel="noreferrer"
            onClick={() => onCitationClick(index)}
            title={`打开录音：${item.recording.title}`}
          >
            [{index}]
          </a>
        );
      } else {
        parts.push(<span key={`${match.index}-${index}`}>[{index}]</span>);
      }
    }
    cursor = match.index + match[0].length;
  }

  if (cursor < text.length) parts.push(text.slice(cursor));
  return <>{parts}</>;
}

export function RagChat() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [askedQuery, setAskedQuery] = useState("");
  const [answer, setAnswer] = useState<RagAnswer | null>(null);
  const [thinking, setThinking] = useState<{ active: boolean; text: string }>({ active: false, text: "" });
  const [evidence, setEvidence] = useState<SearchEvidence[]>([]);
  const [activeEvidenceIndex, setActiveEvidenceIndex] = useState<number | null>(null);

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
    setActiveEvidenceIndex(null);
    try {
      const response = await fetch("/api/rag/query", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ query: text, mode: "answer", limit: 10 })
      });
      const payload = await response.json() as RagQueryResponse & { error?: string };
      if (!response.ok) throw new Error(payload.error || "查询失败");
      setEvidence(payload.evidence);
      setAnswer(payload.answer);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  function selectEvidence(index: number) {
    setActiveEvidenceIndex(index);
    window.setTimeout(() => {
      document.getElementById(`evidence-${index}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }, 0);
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
              <div className="full-text">
                <AnswerText text={answer.text || "正在生成回答..."} evidence={evidence} onCitationClick={selectEvidence} />
              </div>
            </article>
          ) : loading ? (
            <article className="assistant-answer">正在检索录音片段...</article>
          ) : null}
          <EvidenceList evidence={evidence} activeIndex={activeEvidenceIndex} onSelect={selectEvidence} />
        </section>
      ) : null}
    </div>
  );
}
