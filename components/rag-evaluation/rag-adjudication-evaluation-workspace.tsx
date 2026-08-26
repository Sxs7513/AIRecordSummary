"use client";

import Link from "next/link";
import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ragAdjudicationEvaluationRequest } from "@/app/sdk/rag-adjudication-evaluation/client";
import type {
  AdjudicationCase,
  AdjudicationDataset,
  AdjudicationDatasetDetail,
  AdjudicationEvidence,
  AdjudicationRun,
  AdjudicationRunDetail,
  AdjudicationVersion,
  RecordingOption,
  SearchChunk,
  VersionPreview
} from "@/app/sdk/rag-adjudication-evaluation/types";

export function RagAdjudicationEvaluationWorkspace() {
  const [datasets, setDatasets] = useState<AdjudicationDataset[]>([]);
  const [datasetId, setDatasetId] = useState("");
  const [detail, setDetail] = useState<AdjudicationDatasetDetail | null>(null);
  const [selectedCaseId, setSelectedCaseId] = useState("");
  const [chunks, setChunks] = useState<SearchChunk[]>([]);
  const [recordings, setRecordings] = useState<RecordingOption[]>([]);
  const [recordingId, setRecordingId] = useState("");
  const [runs, setRuns] = useState<AdjudicationRun[]>([]);
  const [runDetail, setRunDetail] = useState<AdjudicationRunDetail | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectedCase = detail?.cases.find((item) => item.id === selectedCaseId) ?? null;
  const selectedChunkIds = useMemo(
    () => new Set(selectedCase?.evidence.map((item) => item.source_chunk_id).filter(Boolean) ?? []),
    [selectedCase]
  );
  const targets = selectedCase?.evidence.filter((item) => item.role === "target").sort(byPosition) ?? [];
  const references = selectedCase?.evidence.filter((item) => item.role === "reference").sort(byPosition) ?? [];

  const loadDatasets = useCallback(async () => {
    const result = await ragAdjudicationEvaluationRequest<AdjudicationDataset[]>("/datasets");
    setDatasets(result);
    setDatasetId((current) => current || result[0]?.id || "");
  }, []);

  const loadDetail = useCallback(async () => {
    if (!datasetId) {
      setDetail(null);
      return;
    }
    const result = await ragAdjudicationEvaluationRequest<AdjudicationDatasetDetail>(`/datasets/${datasetId}`);
    setDetail(result);
    setSelectedCaseId((current) => result.cases.some((item) => item.id === current) ? current : result.cases[0]?.id || "");
  }, [datasetId]);

  const loadRuns = useCallback(async () => {
    setRuns(await ragAdjudicationEvaluationRequest<AdjudicationRun[]>("/runs"));
  }, []);

  const loadRun = useCallback(async (runId: string) => {
    setRunDetail(await ragAdjudicationEvaluationRequest<AdjudicationRunDetail>(`/runs/${runId}`));
  }, []);

  useEffect(() => {
    void Promise.all([
      loadDatasets(),
      loadRuns(),
      ragAdjudicationEvaluationRequest<RecordingOption[]>("/recordings").then(setRecordings)
    ]).catch((caught) => setError(message(caught)));
  }, [loadDatasets, loadRuns]);

  useEffect(() => {
    void loadDetail().catch((caught) => setError(message(caught)));
  }, [loadDetail]);

  useEffect(() => {
    const run = runDetail?.run;
    if (!run || !["queued", "running"].includes(run.status)) return;
    const timer = window.setInterval(() => {
      void Promise.all([loadRun(run.id), loadRuns()]).catch((caught) => setError(message(caught)));
    }, 2000);
    return () => window.clearInterval(timer);
  }, [loadRun, loadRuns, runDetail?.run]);

  async function perform(action: () => Promise<void>) {
    setBusy(true);
    setError(null);
    try {
      await action();
    } catch (caught) {
      setError(message(caught));
    } finally {
      setBusy(false);
    }
  }

  async function createDataset(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    const data = new FormData(form);
    await perform(async () => {
      const created = await ragAdjudicationEvaluationRequest<AdjudicationDataset>("/datasets", {
        method: "POST",
        body: JSON.stringify({ name: data.get("name"), description: data.get("description") || null })
      });
      form.reset();
      await loadDatasets();
      setDatasetId(created.id);
    });
  }

  async function createCase(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!datasetId) return;
    const form = event.currentTarget;
    const data = new FormData(form);
    const tags = String(data.get("tags") || "").split(",").map((item) => item.trim()).filter(Boolean);
    await perform(async () => {
      const created = await ragAdjudicationEvaluationRequest<AdjudicationCase>(`/datasets/${datasetId}/cases`, {
        method: "POST",
        body: JSON.stringify({ query: data.get("query"), tags })
      });
      form.reset();
      await loadDetail();
      setSelectedCaseId(created.id);
    });
  }

  async function searchChunks(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    const form = event?.currentTarget;
    const query = form ? String(new FormData(form).get("query") || "") : "";
    const params = new URLSearchParams({ query, limit: "50" });
    if (recordingId) params.set("recording_id", recordingId);
    await perform(async () => {
      setChunks(await ragAdjudicationEvaluationRequest<SearchChunk[]>(`/chunks?${params}`));
    });
  }

  async function addEvidence(chunk: SearchChunk, role: "target" | "reference") {
    if (!selectedCase) return;
    const evidenceForRole = role === "target" ? targets : references;
    const position = evidenceForRole.reduce((max, item) => Math.max(max, item.position), -1) + 1;
    await perform(async () => {
      await ragAdjudicationEvaluationRequest(`/cases/${selectedCase.id}/evidence`, {
        method: "POST",
        body: JSON.stringify({ chunk_id: chunk.id, role, position })
      });
      await loadDetail();
    });
  }

  async function removeEvidence(id: string) {
    await perform(async () => {
      await ragAdjudicationEvaluationRequest(`/evidence/${id}`, { method: "DELETE" });
      await loadDetail();
    });
  }

  async function changeRole(item: AdjudicationEvidence, role: "target" | "reference") {
    const evidenceForRole = role === "target" ? targets : references;
    const position = evidenceForRole.reduce((max, entry) => Math.max(max, entry.position), -1) + 1;
    await perform(async () => {
      await ragAdjudicationEvaluationRequest(`/evidence/${item.id}`, {
        method: "PATCH",
        body: JSON.stringify({ role, position })
      });
      await loadDetail();
    });
  }

  async function moveTarget(index: number, direction: -1 | 1) {
    const otherIndex = index + direction;
    if (otherIndex < 0 || otherIndex >= targets.length) return;
    const current = targets[index];
    const other = targets[otherIndex];
    await perform(async () => {
      await ragAdjudicationEvaluationRequest(`/evidence/${current.id}`, {
        method: "PATCH",
        body: JSON.stringify({ role: "target", position: 10000 })
      });
      await ragAdjudicationEvaluationRequest(`/evidence/${other.id}`, {
        method: "PATCH",
        body: JSON.stringify({ role: "target", position: current.position })
      });
      await ragAdjudicationEvaluationRequest(`/evidence/${current.id}`, {
        method: "PATCH",
        body: JSON.stringify({ role: "target", position: other.position })
      });
      await loadDetail();
    });
  }

  async function transition(item: AdjudicationCase, action: "review" | "approve") {
    await perform(async () => {
      await ragAdjudicationEvaluationRequest(`/cases/${item.id}:${action}`, {
        method: "POST",
        body: JSON.stringify({ revision: item.revision })
      });
      await loadDetail();
    });
  }

  async function deleteCase(item: AdjudicationCase) {
    if (!window.confirm(`确定删除“${item.query}”吗？`)) return;
    await perform(async () => {
      await ragAdjudicationEvaluationRequest(`/cases/${item.id}`, { method: "DELETE" });
      await loadDetail();
    });
  }

  async function freezeVersion() {
    if (!detail) return;
    await perform(async () => {
      const preview = await ragAdjudicationEvaluationRequest<VersionPreview>(
        `/datasets/${detail.dataset.id}/versions:preview`,
        { method: "POST" }
      );
      if (!window.confirm(`冻结 ${preview.case_count} 个 Case、${preview.target_count} 个 Target 和 ${preview.correction_count} 条 Gold？`)) return;
      await ragAdjudicationEvaluationRequest(`/datasets/${detail.dataset.id}/versions:freeze`, {
        method: "POST",
        body: JSON.stringify({ expected_checksum: preview.checksum })
      });
      await loadDetail();
    });
  }

  async function createRun(version: AdjudicationVersion) {
    await perform(async () => {
      const run = await ragAdjudicationEvaluationRequest<AdjudicationRun>("/runs", {
        method: "POST",
        body: JSON.stringify({ dataset_version_id: version.id, idempotency_key: crypto.randomUUID() })
      });
      await loadRuns();
      await loadRun(run.id);
    });
  }

  async function deleteRun(run: AdjudicationRun) {
    if (!window.confirm("确定删除该评测 Run 和全部结果吗？")) return;
    await perform(async () => {
      await ragAdjudicationEvaluationRequest(`/runs/${run.id}`, { method: "DELETE" });
      if (runDetail?.run.id === run.id) setRunDetail(null);
      await loadRuns();
    });
  }

  const caseReady = targets.length > 0 && targets.every((item) => item.corrections.length > 0);

  return (
    <div className="rag-eval-page grid">
      <header className="topbar">
        <div>
          <h1>RAG ASR 文本裁决评测</h1>
          <p className="subtle">固定 Target 与 Reference，直接评测关键错误是否被 Adjudication Agent 修改正确。</p>
          <nav className="rag-eval-tabs">
            <Link href="/rag-evaluation">检索评测</Link>
            <Link className="active" href="/rag-evaluation/adjudication">文本裁决评测</Link>
          </nav>
        </div>
      </header>

      {error && <div className="error">{error}</div>}

      <section className="rag-eval-layout">
        <aside className="panel rag-eval-sidebar">
          <h2>评测集</h2>
          <form className="form-grid" onSubmit={createDataset}>
            <label>名称<input name="name" required placeholder="关键术语修正集" /></label>
            <label>说明<textarea name="description" rows={2} /></label>
            <button className="button" disabled={busy}>新建评测集</button>
          </form>
          <div className="rag-eval-dataset-list">
            {datasets.map((item) => (
              <button
                className={`button secondary rag-eval-dataset ${item.id === datasetId ? "active" : ""}`}
                key={item.id}
                onClick={() => setDatasetId(item.id)}
              >
                <strong>{item.name}</strong>
                <span>{item.case_count} 个 Case · {item.version_count} 个版本</span>
              </button>
            ))}
          </div>
        </aside>

        <main className="grid rag-eval-main">
          {!detail ? <section className="panel subtle">创建或选择一个评测集。</section> : (
            <>
              <section className="panel">
                <div className="rag-eval-section-title">
                  <div><h2>{detail.dataset.name}</h2><p className="subtle">{detail.dataset.description || "尚无说明"}</p></div>
                  <button className="button" disabled={busy || !detail.cases.some((item) => item.status === "approved")} onClick={freezeVersion}>
                    冻结新版本
                  </button>
                </div>
                <form className="rag-eval-create-case" onSubmit={createCase}>
                  <label>Query<input name="query" required placeholder="这里提到的接口协议是什么？" /></label>
                  <label>标签（逗号分隔）<input name="tags" placeholder="专名,关键事实" /></label>
                  <button className="button" disabled={busy}>添加 Case</button>
                </form>
              </section>

              <section className="rag-adj-grid">
                <section className="panel">
                  <h2>Case</h2>
                  <div className="rag-eval-case-list">
                    {detail.cases.map((item) => (
                      <article className={`rag-eval-case ${selectedCaseId === item.id ? "active" : ""}`} key={item.id}>
                        <button className="rag-eval-case-select" onClick={() => setSelectedCaseId(item.id)}>
                          <span><span className={`badge ${item.status}`}>{statusLabel(item.status)}</span> {item.query}</span>
                          <small>{item.evidence.filter((entry) => entry.role === "target").length}T / {item.evidence.filter((entry) => entry.role === "reference").length}R</small>
                        </button>
                        <div className="actions">
                          {item.status === "draft" && (
                            <button className="button secondary" disabled={busy || item.id !== selectedCaseId || !caseReady} onClick={() => transition(item, "review")}>提交审核</button>
                          )}
                          {item.status === "reviewed" && <button className="button secondary" disabled={busy} onClick={() => transition(item, "approve")}>批准</button>}
                          <button className="button danger" disabled={busy} onClick={() => deleteCase(item)}>删除</button>
                        </div>
                      </article>
                    ))}
                  </div>
                </section>

                <section className="panel">
                  <h2>Evidence 搜索</h2>
                  {!selectedCase ? <p className="subtle">先选择一个 Case。</p> : (
                    <>
                      <label>录音筛选
                        <select value={recordingId} onChange={(event) => setRecordingId(event.target.value)}>
                          <option value="">全部录音</option>
                          {recordings.map((item) => <option key={item.id} value={item.id}>{item.title}（{item.chunk_count}）</option>)}
                        </select>
                      </label>
                      <form className="rag-eval-chunk-search" onSubmit={searchChunks}>
                        <input name="query" placeholder="输入关键词；留空可浏览所选录音" />
                        <button className="button" disabled={busy}>搜索</button>
                      </form>
                      <div className="rag-eval-chunks">
                        {chunks.map((chunk) => {
                          const selected = selectedChunkIds.has(chunk.id);
                          return (
                            <article className="rag-eval-chunk" key={chunk.id}>
                              <div className="rag-eval-chunk-meta"><strong>{chunk.recording_title}</strong><span>{formatTime(chunk.start_ms)}–{formatTime(chunk.end_ms)}</span></div>
                              <p>{chunk.text}</p>
                              <div className="actions">
                                <button className="button" disabled={busy || selected || targets.length >= 2} onClick={() => addEvidence(chunk, "target")}>加入 Target</button>
                                <button className="button secondary" disabled={busy || selected} onClick={() => addEvidence(chunk, "reference")}>加入 Reference</button>
                              </div>
                            </article>
                          );
                        })}
                        {!chunks.length && <p className="subtle">搜索所有 SearchChunk，或选择录音后留空浏览。</p>}
                      </div>
                    </>
                  )}
                </section>
              </section>

              {selectedCase && (
                <section className="panel">
                  <h2>Evidence 编排与 Gold 标注</h2>
                  <p className="subtle">先固定 Target/Reference，再在 Target 原文中框选关键错误并录入可接受表达。</p>

                  <h3>Target Evidence（最多 2 条）</h3>
                  <div className="rag-adj-evidence-list">
                    {targets.map((item, index) => (
                      <TargetEditor
                        busy={busy}
                        evidence={item}
                        index={index}
                        key={item.id}
                        onChanged={loadDetail}
                        onMove={(direction) => moveTarget(index, direction)}
                        onRemove={() => removeEvidence(item.id)}
                        onRoleChange={() => changeRole(item, "reference")}
                        targetCount={targets.length}
                      />
                    ))}
                    {!targets.length && <p className="subtle">从搜索结果中加入至少一个 Target。</p>}
                  </div>

                  <h3>Reference Evidence</h3>
                  <div className="rag-adj-reference-list">
                    {references.map((item) => (
                      <article className="rag-adj-reference" key={item.id}>
                        <div><strong>{item.recording_title}</strong><span>{formatTime(item.start_ms)}–{formatTime(item.end_ms)}</span></div>
                        <p>{item.text}</p>
                        <div className="actions">
                          <button className="button secondary" disabled={busy || targets.length >= 2} onClick={() => changeRole(item, "target")}>设为 Target</button>
                          <button className="button danger" disabled={busy} onClick={() => removeEvidence(item.id)}>移除</button>
                        </div>
                      </article>
                    ))}
                    {!references.length && <p className="subtle">Reference 可为空且可以来自不同录音；运行时每个 Target 最多使用 5 条 Reference Evidence。</p>}
                  </div>
                </section>
              )}

              <section className="panel">
                <h2>冻结版本</h2>
                <div className="rag-eval-version-list">
                  {detail.versions.slice(0, 1).map((version) => (
                    <div className="rag-eval-version" key={version.id}>
                      <div><strong>v{version.version_number} <span className="badge">最新</span></strong><span>{version.case_count} 个 Case · {version.status}</span></div>
                      <button className="button" disabled={busy || version.status !== "frozen"} onClick={() => createRun(version)}>运行评测</button>
                    </div>
                  ))}
                  {detail.versions.length > 1 && (
                    <details className="rag-eval-version-history">
                      <summary>其他冻结版本（{detail.versions.length - 1}）</summary>
                      <div className="rag-eval-version-list">
                        {detail.versions.slice(1).map((version) => (
                          <div className="rag-eval-version" key={version.id}>
                            <div><strong>v{version.version_number}</strong><span>{version.case_count} 个 Case · {version.status}</span></div>
                            <button className="button secondary" disabled={busy || version.status !== "frozen"} onClick={() => createRun(version)}>运行评测</button>
                          </div>
                        ))}
                      </div>
                    </details>
                  )}
                  {!detail.versions.length && <p className="subtle">批准 Case 后冻结不可变版本。</p>}
                </div>
              </section>
            </>
          )}
        </main>
      </section>

      <RunResults busy={busy} detail={runDetail} runs={runs} onDelete={deleteRun} onOpen={(id) => perform(() => loadRun(id))} />
    </div>
  );
}

function TargetEditor({
  evidence,
  index,
  targetCount,
  busy,
  onChanged,
  onMove,
  onRemove,
  onRoleChange
}: {
  evidence: AdjudicationEvidence;
  index: number;
  targetCount: number;
  busy: boolean;
  onChanged: () => Promise<void>;
  onMove: (direction: -1 | 1) => Promise<void>;
  onRemove: () => Promise<void>;
  onRoleChange: () => Promise<void>;
}) {
  const textRef = useRef<HTMLDivElement>(null);
  const [selection, setSelection] = useState<{ start: number; end: number; text: string } | null>(null);
  const [accepted, setAccepted] = useState("");
  const [importance, setImportance] = useState<"important" | "minor">("important");
  const [localError, setLocalError] = useState<string | null>(null);

  function captureSelection() {
    const root = textRef.current;
    const selected = window.getSelection();
    if (!root || !selected || selected.rangeCount === 0 || selected.isCollapsed) return;
    const range = selected.getRangeAt(0);
    if (!root.contains(range.commonAncestorContainer)) return;
    const prefix = range.cloneRange();
    prefix.selectNodeContents(root);
    prefix.setEnd(range.startContainer, range.startOffset);
    const start = Array.from(prefix.toString()).length;
    const selectedText = range.toString();
    const end = start + Array.from(selectedText).length;
    if (selectedText && Array.from(evidence.text).slice(start, end).join("") === selectedText) {
      setSelection({ start, end, text: selectedText });
      setLocalError(null);
    }
  }

  async function addCorrection(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selection) {
      setLocalError("请先在 Target 文本中框选错误表达。");
      return;
    }
    const values = accepted.split(/[,，\n]/).map((item) => item.trim()).filter(Boolean);
    if (!values.length) {
      setLocalError("至少录入一个正确表达。");
      return;
    }
    try {
      await ragAdjudicationEvaluationRequest(`/evidence/${evidence.id}/corrections`, {
        method: "POST",
        body: JSON.stringify({
          start_char: selection.start,
          end_char: selection.end,
          original_expression: selection.text,
          accepted_expressions: values,
          importance
        })
      });
      setSelection(null);
      setAccepted("");
      setImportance("important");
      setLocalError(null);
      await onChanged();
    } catch (caught) {
      setLocalError(message(caught));
    }
  }

  async function deleteCorrection(id: string) {
    try {
      await ragAdjudicationEvaluationRequest(`/corrections/${id}`, { method: "DELETE" });
      await onChanged();
    } catch (caught) {
      setLocalError(message(caught));
    }
  }

  async function updateImportance(
    item: AdjudicationEvidence["corrections"][number],
    value: "important" | "minor"
  ) {
    try {
      await ragAdjudicationEvaluationRequest(`/corrections/${item.id}`, {
        method: "PATCH",
        body: JSON.stringify({
          start_char: item.start_char,
          end_char: item.end_char,
          original_expression: item.original_expression,
          accepted_expressions: item.accepted_expressions,
          importance: value
        })
      });
      await onChanged();
    } catch (caught) {
      setLocalError(message(caught));
    }
  }

  return (
    <article className="rag-adj-target">
      <div className="rag-adj-target-heading">
        <div><strong>Target {index + 1} · {evidence.recording_title}</strong><span>{formatTime(evidence.start_ms)}–{formatTime(evidence.end_ms)}</span></div>
        <div className="actions">
          <button className="button secondary" disabled={busy || index === 0} onClick={() => onMove(-1)}>上移</button>
          <button className="button secondary" disabled={busy || index === targetCount - 1} onClick={() => onMove(1)}>下移</button>
          <button className="button secondary" disabled={busy || evidence.corrections.length > 0} onClick={onRoleChange}>设为 Reference</button>
          <button className="button danger" disabled={busy} onClick={onRemove}>移除</button>
        </div>
      </div>
      <div className="rag-adj-selectable-text" onMouseUp={captureSelection} ref={textRef}>
        <HighlightedText corrections={evidence.corrections} text={evidence.text} />
      </div>
      <form className="rag-adj-correction-form" onSubmit={addCorrection}>
        <label>已选错误文本<input readOnly value={selection?.text ?? ""} placeholder="请在上方文本中框选" /></label>
        <label>正确表达（逗号或换行分隔）<textarea onChange={(event) => setAccepted(event.target.value)} rows={2} value={accepted} /></label>
        <label>重要性<select onChange={(event) => setImportance(event.target.value as "important" | "minor")} value={importance}><option value="important">重要 · 1.0</option><option value="minor">次要 · 0.5</option></select></label>
        <button className="button" disabled={busy || !selection}>添加 Gold</button>
      </form>
      {localError && <p className="error">{localError}</p>}
      <div className="rag-adj-gold-list">
        {evidence.corrections.map((item) => (
          <div className="rag-adj-gold" key={item.id}>
            <span><del>{item.original_expression}</del> → <AcceptedExpressionTags values={item.accepted_expressions} /></span>
            <select aria-label="Gold 重要性" disabled={busy} onChange={(event) => void updateImportance(item, event.target.value as "important" | "minor")} value={item.importance}><option value="important">重要 · 1.0</option><option value="minor">次要 · 0.5</option></select>
            <small>[{item.start_char}, {item.end_char})</small>
            <button className="button danger" disabled={busy} onClick={() => deleteCorrection(item.id)}>删除</button>
          </div>
        ))}
      </div>
    </article>
  );
}

function HighlightedText({ text, corrections }: { text: string; corrections: AdjudicationEvidence["corrections"] }) {
  const chars = Array.from(text);
  const ordered = [...corrections].sort((left, right) => left.start_char - right.start_char);
  const nodes: React.ReactNode[] = [];
  let cursor = 0;
  ordered.forEach((item) => {
    nodes.push(chars.slice(cursor, item.start_char).join(""));
    nodes.push(<mark key={item.id}>{chars.slice(item.start_char, item.end_char).join("")}</mark>);
    cursor = item.end_char;
  });
  nodes.push(chars.slice(cursor).join(""));
  return <>{nodes}</>;
}

function AcceptedExpressionTags({ values }: { values: string[] }) {
  return (
    <span className="rag-adj-expression-tags">
      {values.map((value, index) => (
        <span className="rag-adj-expression-tag" key={`${value}-${index}`}>{value}</span>
      ))}
    </span>
  );
}

function RunResults({
  runs,
  detail,
  busy,
  onOpen,
  onDelete
}: {
  runs: AdjudicationRun[];
  detail: AdjudicationRunDetail | null;
  busy: boolean;
  onOpen: (id: string) => void;
  onDelete: (run: AdjudicationRun) => Promise<void>;
}) {
  const metric = (name: string) => detail?.metrics.find((item) => item.metric_name === name);
  const strictPrecision = metric("correction_precision_strict");
  const strictRecall = metric("correction_recall_strict");
  const strictF1 = metric("correction_f1_strict");
  const relaxedPrecision = metric("correction_precision_relaxed");
  const relaxedRecall = metric("correction_recall_relaxed");
  const relaxedF1 = metric("correction_f1_relaxed");
  const counts = relaxedRecall?.details;
  const percentage = (value: number | string | undefined) => value === undefined ? "—" : `${(Number(value) * 100).toFixed(1)}%`;

  return (
    <section className="panel">
      <h2>评测 Run</h2>
      <div className="rag-eval-run-list">
        {runs.map((run) => (
          <div className="rag-eval-run-item" key={run.id}>
            <button className={`rag-eval-run ${detail?.run.id === run.id ? "active" : ""}`} disabled={busy} onClick={() => onOpen(run.id)}>
              <span><strong>{run.dataset_name} v{run.version_number}</strong><small>{run.completed_case_count}/{run.total_case_count} Case</small></span>
              <span className={`badge ${run.status}`}>{run.status}</span>
            </button>
            {["succeeded", "failed", "cancelled"].includes(run.status) && (
              <button className="button danger rag-eval-run-delete" disabled={busy} onClick={() => onDelete(run)}>删除</button>
            )}
          </div>
        ))}
      </div>
      {detail && (
        <div className="rag-adj-run-detail">
          <div className="rag-adj-metric-groups">
            <section className="rag-adj-metric-group primary">
              <div className="rag-adj-metric-heading"><strong>语义匹配</strong><span>主指标 · 重要 1.0 / 次要 0.5</span></div>
              <div className="stats rag-adj-metrics">
                <div className="stat card"><span className="subtle">Precision</span><strong>{percentage(relaxedPrecision?.value)}</strong></div>
                <div className="stat card"><span className="subtle">Recall</span><strong>{percentage(relaxedRecall?.value)}</strong></div>
                <div className="stat card"><span className="subtle">F1</span><strong>{percentage(relaxedF1?.value)}</strong></div>
              </div>
            </section>
            <div className="stats rag-adj-metric-summary">
              <div className="stat card"><span className="subtle">漏改 / 误改（互斥计数）</span><strong>{counts?.missed_gold_count ?? counts?.false_negative ?? 0}/{counts?.incorrect_prediction_count ?? counts?.false_positive ?? 0}</strong></div>
            </div>
            <details className="rag-adj-metric-advanced">
              <summary>高级指标 · 精确匹配</summary>
              <section className="rag-adj-metric-group">
                <div className="rag-adj-metric-heading"><strong>精确匹配</strong><span>仅统计 Exact</span></div>
                <div className="stats rag-adj-metrics">
                  <div className="stat card"><span className="subtle">Precision</span><strong>{percentage(strictPrecision?.value)}</strong></div>
                  <div className="stat card"><span className="subtle">Recall</span><strong>{percentage(strictRecall?.value)}</strong></div>
                  <div className="stat card"><span className="subtle">F1</span><strong>{percentage(strictF1?.value)}</strong></div>
                </div>
              </section>
            </details>
          </div>
          <div className="rag-eval-result-cases">
            {detail.cases.map((item) => (
              <details className="rag-eval-result-case" key={item.id}>
                <summary><strong>{item.query}</strong><span className={`badge ${item.status}`}>{item.status}</span><span>{item.latency_ms ?? 0} ms</span></summary>
                {item.error_message && <p className="error">{item.error_type}: {item.error_message}</p>}
                <div className="rag-adj-result-gold">
                  {item.corrections.map((correction) => (
                    <div className={correction.passed ? "passed" : "failed"} key={correction.gold_correction_id}>
                      <span>{correction.original_expression} → <AcceptedExpressionTags values={correction.accepted_expressions} /> <span className={`rag-adj-importance ${correction.importance}`}>{correction.importance === "important" ? "重要 · 1.0" : "次要 · 0.5"}</span></span>
                      <strong>
                        {correction.passed
                          ? `✓ ${correction.actual_expression} · ${correction.details.match_kind}${correction.details.match_kind === "fuzzy" ? ` ${Number(correction.details.similarity).toFixed(1)} · ${correction.details.match_basis === "expression" ? "表达匹配" : "局部文本匹配"}` : ""}`
                          : "漏改"}
                      </strong>
                    </div>
                  ))}
                  {item.predictions.filter((prediction) => prediction.match_kind === "unmatched").map((prediction) => (
                    <div className="failed" key={prediction.id}>
                      <span>{prediction.original_expression} → {prediction.resolved_expression}</span>
                      <strong>误改</strong>
                    </div>
                  ))}
                </div>
                {(item.agent_state || item.trace_events.length > 0) && (
                  <details className="rag-adj-trace">
                    <summary>查看 Agent Trace · {item.trace_events.length} 条模型日志</summary>
                    <div className="rag-adj-trace-events">
                      {item.trace_events.map((event) => (
                        <section className="rag-adj-trace-event" key={event.sequence}>
                          <div>
                            <strong>#{event.sequence} · {event.operation}</strong>
                            <span>Case {event.case_index + 1} · Iteration {event.iteration} · {event.provider}/{event.model}</span>
                          </div>
                          <small>
                            Tokens {event.prompt_tokens ?? "—"} + {event.completion_tokens ?? "—"}
                            {event.finish_reason ? ` · ${event.finish_reason}` : ""}
                          </small>
                          <pre>{JSON.stringify(event.payload, null, 2)}</pre>
                        </section>
                      ))}
                    </div>
                    {item.agent_state && (
                      <details className="rag-adj-final-state">
                        <summary>查看最终 Agent State</summary>
                        <pre>{JSON.stringify(item.agent_state, null, 2)}</pre>
                      </details>
                    )}
                  </details>
                )}
              </details>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function byPosition(left: AdjudicationEvidence, right: AdjudicationEvidence) {
  return left.position - right.position;
}

function statusLabel(value: AdjudicationCase["status"]) {
  return value === "draft" ? "草稿" : value === "reviewed" ? "已审核" : "已批准";
}

function formatTime(value: number) {
  const total = Math.floor(value / 1000);
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

function message(error: unknown) {
  return error instanceof Error ? error.message : "请求失败";
}
