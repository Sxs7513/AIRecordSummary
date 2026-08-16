"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { EvidenceList } from "./evidence-list";
import type { SearchEvidence } from "@/app/shared/models";
import { GenerationStreamClient } from "@/app/sdk/generation/client";
import { useGenerationStore } from "@/app/sdk/generation/store";
import { selectGenerationText } from "@/app/sdk/generation/selectors";

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
    const indexes = match[1].split(/[\s,，、]+/).map(Number).filter(Number.isInteger);
    if (indexes.length === 0) parts.push(match[0]);
    for (const index of indexes) {
      const item = evidenceByIndex.get(index);
      parts.push(item ? (
        <a className="citation-link" href={item.url} key={`${match.index}-${index}`} target="_blank" rel="noreferrer" onClick={() => onCitationClick(index)} title={`打开录音：${item.recording.title}`}>
          [{index}]
        </a>
      ) : <span key={`${match.index}-${index}`}>[{index}]</span>);
    }
    cursor = match.index + match[0].length;
  }
  if (cursor < text.length) parts.push(text.slice(cursor));
  return <>{parts}</>;
}

export function RagChat() {
  const client = useRef<GenerationStreamClient | null>(null);
  const [query, setQuery] = useState("");
  const [askedQuery, setAskedQuery] = useState("");
  const [runId, setRunId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeEvidenceIndex, setActiveEvidenceIndex] = useState<number | null>(null);
  const run = useGenerationStore((state) => runId ? state.runs[runId] : undefined);

  if (client.current === null) client.current = new GenerationStreamClient();

  useEffect(() => () => {
    if (runId) client.current?.close(runId);
  }, [runId]);

  const answerText = selectGenerationText(run);
  const evidence = asEvidence(run?.sources);
  const loading = run?.status === "queued" || run?.status === "running" || (runId !== null && run === undefined);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    const text = query.trim();
    if (!text || loading) return;
    setError(null);
    setAskedQuery(text);
    setActiveEvidenceIndex(null);
    try {
      const nextRunId = await client.current!.start("/api/rag/queries", { query: text, limit: 10 });
      setRunId(nextRunId);
    } catch (reason) {
      setRunId(null);
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  function selectEvidence(index: number) {
    setActiveEvidenceIndex(index);
    window.setTimeout(() => document.getElementById(`evidence-${index}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" }), 0);
  }

  const runError = run?.status === "failed" ? run.error?.message ?? "录音问答失败" : null;

  return (
    <div className="rag-page">
      <form className="rag-search" onSubmit={submit}>
        <textarea value={query} onChange={(event) => setQuery(event.target.value)} placeholder="输入一个关于录音的问题..." rows={4} />
        <div className="rag-actions"><button type="submit" disabled={loading}>{loading ? "生成中..." : "提问"}</button></div>
      </form>

      {error || runError ? <div className="panel error-panel">{error ?? runError}</div> : null}
      {askedQuery ? (
        <section className="rag-result">
          <div className="user-question">{askedQuery}</div>
          {runId ? (
            <article className={`assistant-answer ${run?.output?.notEnoughEvidence === true ? "weak" : ""}`}>
              <div className="full-text"><AnswerText text={answerText || "正在生成回答..."} evidence={evidence} onCitationClick={selectEvidence} /></div>
            </article>
          ) : null}
          <EvidenceList evidence={evidence} activeIndex={activeEvidenceIndex} onSelect={selectEvidence} />
        </section>
      ) : null}
    </div>
  );
}

function asEvidence(value: unknown): SearchEvidence[] {
  return Array.isArray(value) ? value as SearchEvidence[] : [];
}
