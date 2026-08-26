"use client";

import Link from "next/link";
import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";

import { ragEvaluationRequest } from "@/app/sdk/rag-evaluation/client";
import type {
  RagEvalCase,
  RagEvalDataset,
  RagEvalDatasetDetail,
  RagEvalDatasetVersion,
  RagEvalMetric,
  RagEvalRecording,
  RagEvalRun,
  RagEvalRunDetail,
  SearchChunk,
  VersionPreview
} from "@/app/sdk/rag-evaluation/types";

const operationLabels: Record<string, string> = {
  "retrieve.vector": "Vector",
  "retrieve.vector.original": "Vector / 原问题",
  "retrieve.vector.expanded": "Vector / 扩展问题",
  "retrieve.lexical": "Lexical",
  "retrieve.lexical.term": "Lexical / 专业术语（逐词）",
  "retrieve.rrf": "RRF",
  "retrieve.scope": "Scope Summary",
  "retrieve.expand": "Expand",
  "retrieve.rerank": "Rerank",
  "retrieve.empty": "Empty Retrieval",
  "route.unresolved": "Route Unresolved"
};

const retrievalOperationOrder: Record<string, number> = {
  "retrieve.vector": 0,
  "retrieve.vector.original": 0,
  "retrieve.vector.expanded": 1,
  "retrieve.lexical": 2,
  "retrieve.lexical.term": 2,
  "retrieve.rrf": 3,
  "retrieve.expand": 4,
  "retrieve.rerank": 5
};

function compareRetrievalOperations(left: string, right: string) {
  const fallbackRank = Object.keys(retrievalOperationOrder).length;
  return (retrievalOperationOrder[left] ?? fallbackRank) - (retrievalOperationOrder[right] ?? fallbackRank);
}

const CHUNK_PAGE_SIZE = 50;

export function RagEvaluationWorkspace() {
  const [datasets, setDatasets] = useState<RagEvalDataset[]>([]);
  const [datasetId, setDatasetId] = useState("");
  const [detail, setDetail] = useState<RagEvalDatasetDetail | null>(null);
  const [selectedCaseId, setSelectedCaseId] = useState("");
  const [chunks, setChunks] = useState<SearchChunk[]>([]);
  const [recordings, setRecordings] = useState<RagEvalRecording[]>([]);
  const [recordingId, setRecordingId] = useState("");
  const [recordingChunks, setRecordingChunks] = useState<SearchChunk[]>([]);
  const [browseOffset, setBrowseOffset] = useState(0);
  const [browseOpen, setBrowseOpen] = useState(false);
  const [runs, setRuns] = useState<RagEvalRun[]>([]);
  const [runDetail, setRunDetail] = useState<RagEvalRunDetail | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectedCase = detail?.cases.find((item) => item.id === selectedCaseId) ?? null;

  const loadDatasets = useCallback(async () => {
    const result = await ragEvaluationRequest<RagEvalDataset[]>("/datasets");
    setDatasets(result);
    setDatasetId((current) => current || result[0]?.id || "");
  }, []);

  const loadRuns = useCallback(async () => {
    setRuns(await ragEvaluationRequest<RagEvalRun[]>("/runs"));
  }, []);

  const loadRecordings = useCallback(async () => {
    const result = await ragEvaluationRequest<RagEvalRecording[]>("/recordings");
    setRecordings(result);
    setRecordingId((current) => current || result[0]?.id || "");
  }, []);

  const loadDetail = useCallback(async () => {
    if (!datasetId) {
      setDetail(null);
      return;
    }
    const result = await ragEvaluationRequest<RagEvalDatasetDetail>(`/datasets/${datasetId}`);
    setDetail(result);
    setSelectedCaseId((current) => result.cases.some((item) => item.id === current) ? current : result.cases[0]?.id || "");
  }, [datasetId]);

  const loadRun = useCallback(async (runId: string) => {
    setRunDetail(await ragEvaluationRequest<RagEvalRunDetail>(`/runs/${runId}`));
  }, []);

  useEffect(() => {
    void Promise.all([loadDatasets(), loadRuns(), loadRecordings()]).catch((caught) => setError(message(caught)));
  }, [loadDatasets, loadRecordings, loadRuns]);

  useEffect(() => {
    void loadDetail().catch((caught) => setError(message(caught)));
  }, [loadDetail]);

  useEffect(() => {
    const run = runDetail?.run;
    if (!run || (run.status !== "queued" && run.status !== "running")) return;
    const timer = window.setInterval(() => {
      void Promise.all([loadRun(run.id), loadRuns()]).catch((caught) => setError(message(caught)));
    }, 2000);
    return () => window.clearInterval(timer);
  }, [loadRun, loadRuns, runDetail?.run]);

  async function refreshDataset() {
    await Promise.all([loadDatasets(), loadDetail()]);
  }

  async function createDataset(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    const data = new FormData(form);
    await perform(async () => {
      const created = await ragEvaluationRequest<RagEvalDataset>("/datasets", {
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
      const created = await ragEvaluationRequest<RagEvalCase>(`/datasets/${datasetId}/cases`, {
        method: "POST",
        body: JSON.stringify({ query: data.get("query"), tags, recording_ids: [] })
      });
      form.reset();
      await refreshDataset();
      setSelectedCaseId(created.id);
      setChunks([]);
    });
  }

  async function searchChunks(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const query = String(new FormData(event.currentTarget).get("query") || "");
    await perform(async () => {
      setChunks(await ragEvaluationRequest<SearchChunk[]>(`/chunks?query=${encodeURIComponent(query)}&limit=40`));
    });
  }

  async function browseRecordingChunks(offset: number, selectedRecordingId = recordingId) {
    if (!selectedRecordingId) return;
    await perform(async () => {
      const result = await ragEvaluationRequest<SearchChunk[]>(
        `/chunks?recording_id=${encodeURIComponent(selectedRecordingId)}&limit=${CHUNK_PAGE_SIZE}&offset=${offset}`
      );
      setRecordingChunks(result);
      setBrowseOffset(offset);
    });
  }

  async function openRecordingBrowser() {
    if (!recordingId) return;
    setRecordingChunks([]);
    setBrowseOffset(0);
    setBrowseOpen(true);
    await browseRecordingChunks(0);
  }

  async function addEvidence(chunkId: string, relevance: number) {
    if (!selectedCase) return;
    await perform(async () => {
      await ragEvaluationRequest(`/cases/${selectedCase.id}/evidence`, {
        method: "POST",
        body: JSON.stringify({ chunk_id: chunkId, relevance })
      });
      await loadDetail();
    });
  }

  async function removeEvidence(evidenceId: string) {
    await perform(async () => {
      await ragEvaluationRequest(`/evidence/${evidenceId}`, { method: "DELETE" });
      await loadDetail();
    });
  }

  async function transition(item: RagEvalCase, action: "review" | "approve") {
    await perform(async () => {
      await ragEvaluationRequest(`/cases/${item.id}:${action}`, {
        method: "POST",
        body: JSON.stringify({ revision: item.revision })
      });
      await loadDetail();
    });
  }

  async function deleteCase(item: RagEvalCase) {
    if (!window.confirm(`确定删除问题“${item.query}”吗？`)) return;
    await perform(async () => {
      await ragEvaluationRequest(`/cases/${item.id}`, { method: "DELETE" });
      await refreshDataset();
      setChunks([]);
    });
  }

  async function archiveCase(item: RagEvalCase) {
    if (!window.confirm(`归档问题“${item.query}”？它不会进入后续冻结版本，已有版本不受影响。`)) return;
    await perform(async () => {
      await ragEvaluationRequest(`/cases/${item.id}:archive`, { method: "POST" });
      await refreshDataset();
      if (selectedCaseId === item.id) setSelectedCaseId("");
    });
  }

  async function freezeVersion() {
    if (!detail) return;
    await perform(async () => {
      const preview = await ragEvaluationRequest<VersionPreview>(`/datasets/${detail.dataset.id}/versions:preview`, { method: "POST" });
      if (!window.confirm(`冻结 ${preview.case_count} 个问题、${preview.evidence_count} 条正确证据为新版本？冻结后不可修改。`)) return;
      await ragEvaluationRequest(`/datasets/${detail.dataset.id}/versions:freeze`, {
        method: "POST",
        body: JSON.stringify({ expected_checksum: preview.checksum })
      });
      await refreshDataset();
    });
  }

  async function createRun(version: RagEvalDatasetVersion) {
    await perform(async () => {
      const run = await ragEvaluationRequest<RagEvalRun>("/runs", {
        method: "POST",
        body: JSON.stringify({ dataset_version_id: version.id, idempotency_key: crypto.randomUUID() })
      });
      await loadRuns();
      await loadRun(run.id);
    });
  }

  async function deleteRun(run: RagEvalRun) {
    if (!window.confirm(`确定删除 ${run.dataset_name} v${run.version_number} 的评测 Run 吗？此操作会删除所有评测结果和指标。`)) return;
    await perform(async () => {
      await ragEvaluationRequest(`/runs/${run.id}`, { method: "DELETE" });
      if (runDetail?.run.id === run.id) setRunDetail(null);
      await loadRuns();
    });
  }

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

  return (
    <div className="rag-eval-page grid">
      <header className="topbar">
        <div>
          <h1>RAG 检索评测</h1>
          <p className="subtle">用人工标注的正确 Chunk，对比 Vector、Lexical、RRF、Expand 和 Rerank 的真实效果。</p>
          <nav className="rag-eval-tabs">
            <Link className="active" href="/rag-evaluation">检索评测</Link>
            <Link href="/rag-evaluation/adjudication">文本裁决评测</Link>
          </nav>
        </div>
      </header>

      {error && <div className="error">{error}</div>}

      <section className="rag-eval-layout">
        <aside className="panel rag-eval-sidebar">
          <h2>评测集</h2>
          <form className="form-grid" onSubmit={createDataset}>
            <label>名称<input name="name" required placeholder="核心检索集" /></label>
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
                <span>{item.case_count} 个问题 · {item.version_count} 个版本</span>
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
                  <button className="button" disabled={busy || !detail.cases.some((item) => item.status === "approved" && !item.archived_at)} onClick={freezeVersion}>
                    冻结新版本
                  </button>
                </div>
                <form className="rag-eval-create-case" onSubmit={createCase}>
                  <label>问题<input name="query" required placeholder="公司目前营收多少？" /></label>
                  <label>标签（逗号分隔）<input name="tags" placeholder="口语省略,数字事实" /></label>
                  <button className="button" disabled={busy}>添加问题</button>
                </form>
              </section>

              <section className="rag-eval-annotation-grid">
                <div className="panel">
                  <h2>问题与正确证据</h2>
                  <div className="rag-eval-case-list">
                    {detail.cases.map((item) => (
                      <article className={`rag-eval-case ${item.id === selectedCaseId ? "active" : ""}`} key={item.id}>
                        <button className="rag-eval-case-select" onClick={() => { setSelectedCaseId(item.id); setChunks([]); setBrowseOpen(false); setRecordingChunks([]); setBrowseOffset(0); }}>
                          <span><span className={`badge ${item.archived_at ? "archived" : item.status}`}>{item.archived_at ? "已归档" : statusLabel(item.status)}</span> {item.query}</span>
                          <small>{item.evidence.length} 条正确证据</small>
                        </button>
                        <div className="actions">
                          {!item.archived_at && item.status === "draft" && <button className="button secondary" disabled={busy || !item.evidence.length} onClick={() => transition(item, "review")}>提交审核</button>}
                          {!item.archived_at && item.status === "reviewed" && <button className="button secondary" disabled={busy} onClick={() => transition(item, "approve")}>批准</button>}
                          {!item.archived_at && <button className="button secondary" disabled={busy} onClick={() => archiveCase(item)}>归档</button>}
                          {!item.archived_at && <button className="button danger" disabled={busy} onClick={() => deleteCase(item)}>删除</button>}
                        </div>
                        {item.evidence.map((evidence) => (
                          <details className="rag-eval-evidence" key={evidence.id}>
                            <summary>
                              <strong>R{evidence.relevance} · {evidence.recording_title}</strong>
                              <span>{formatTime(evidence.start_ms)}–{formatTime(evidence.end_ms)}</span>
                            </summary>
                            <div className="rag-eval-evidence-body">
                              <p>{evidence.quote}</p>
                              <button className="button danger" disabled={busy} onClick={() => removeEvidence(evidence.id)}>移除</button>
                            </div>
                          </details>
                        ))}
                      </article>
                    ))}
                    {!detail.cases.length && <p className="subtle">先添加一个来自真实录音的问题。</p>}
                  </div>
                </div>

                <div className="panel">
                  <h2>标注正确 Chunk</h2>
                  {!selectedCase ? <p className="subtle">选择左侧问题后搜索录音片段。</p> : (
                    <>
                      <p className="subtle">当前问题：{selectedCase.query}</p>
                      <button className="button secondary" disabled={busy || !recordingId} onClick={() => void openRecordingBrowser()}>
                        浏览录音全部 Chunk
                      </button>
                      <div className="rag-eval-search-divider"><span>关键词辅助查找</span></div>
                      <form className="rag-eval-chunk-search" onSubmit={searchChunks}>
                        <input name="query" placeholder="搜索录音原词或口语表达" />
                        <button className="button" disabled={busy}>搜索</button>
                      </form>
                      <div className="rag-eval-chunks">
                        {chunks.map((chunk) => (
                          <ChunkCandidate key={chunk.id} chunk={chunk} busy={busy} onAdd={addEvidence} />
                        ))}
                        {!chunks.length && <p className="subtle">尚未加载 Chunk，选择录音浏览或输入关键词搜索。</p>}
                      </div>
                      {browseOpen && (
                        <div className="rag-eval-modal-backdrop" onClick={() => setBrowseOpen(false)}>
                          <section aria-modal="true" className="rag-eval-modal" onClick={(event) => event.stopPropagation()} role="dialog">
                            <div className="rag-eval-section-title">
                              <div><h2>浏览录音全部 Chunk</h2><p className="subtle">按录音时间顺序浏览，不依赖检索召回。</p></div>
                              <button aria-label="关闭录音 Chunk 浏览器" className="button secondary" onClick={() => setBrowseOpen(false)}>关闭</button>
                            </div>
                            <label>录音
                              <select
                                disabled={busy}
                                value={recordingId}
                                onChange={(event) => {
                                  const selectedId = event.target.value;
                                  setRecordingId(selectedId);
                                  setRecordingChunks([]);
                                  void browseRecordingChunks(0, selectedId);
                                }}
                              >
                                {!recordings.length && <option value="">暂无可浏览录音</option>}
                                {recordings.map((recording) => (
                                  <option key={recording.id} value={recording.id}>{recording.title}（{recording.chunk_count} chunks）</option>
                                ))}
                              </select>
                            </label>
                            <div className="rag-eval-modal-chunks">
                              {recordingChunks.map((chunk) => (
                                <ChunkCandidate key={chunk.id} chunk={chunk} busy={busy} onAdd={addEvidence} />
                              ))}
                              {!recordingChunks.length && <p className="subtle">该录音暂无可标注 Chunk。</p>}
                            </div>
                            <div className="actions rag-eval-chunk-pagination">
                              <button className="button secondary" disabled={busy || browseOffset === 0} onClick={() => void browseRecordingChunks(Math.max(0, browseOffset - CHUNK_PAGE_SIZE))}>上一页</button>
                              <span>第 {Math.floor(browseOffset / CHUNK_PAGE_SIZE) + 1} 页</span>
                              <button className="button secondary" disabled={busy || recordingChunks.length < CHUNK_PAGE_SIZE} onClick={() => void browseRecordingChunks(browseOffset + CHUNK_PAGE_SIZE)}>下一页</button>
                            </div>
                          </section>
                        </div>
                      )}
                    </>
                  )}
                </div>
              </section>

              <section className="panel">
                <h2>冻结版本</h2>
                <div className="rag-eval-version-list">
                  {detail.versions.map((version) => (
                    <div className="rag-eval-version" key={version.id}>
                      <div><strong>v{version.version_number}</strong><span>{version.case_count} 个问题 · {version.status}</span></div>
                      <button className="button" disabled={busy || version.status !== "frozen"} onClick={() => createRun(version)}>运行评测</button>
                    </div>
                  ))}
                  {!detail.versions.length && <p className="subtle">批准问题后冻结为不可变版本。</p>}
                </div>
              </section>
            </>
          )}
        </main>
      </section>

      <RunWorkspace runs={runs} detail={runDetail} busy={busy} onOpen={(id) => perform(() => loadRun(id))} onDelete={deleteRun} />
    </div>
  );
}

function ChunkCandidate({ chunk, busy, onAdd }: { chunk: SearchChunk; busy: boolean; onAdd: (id: string, relevance: number) => Promise<void> }) {
  const [relevance, setRelevance] = useState(3);
  return (
    <article className="rag-eval-chunk">
      <div className="rag-eval-chunk-meta"><strong>{chunk.recording_title}</strong><span>{formatTime(chunk.start_ms)}–{formatTime(chunk.end_ms)}</span></div>
      <p>{chunk.text}</p>
      <div className="actions">
        <select value={relevance} onChange={(event) => setRelevance(Number(event.target.value))}>
          <option value={3}>3 · 直接回答</option>
          <option value={2}>2 · 部分回答</option>
          <option value={1}>1 · 相关背景</option>
        </select>
        <button className="button secondary" disabled={busy} onClick={() => void onAdd(chunk.id, relevance)}>标为正确证据</button>
      </div>
    </article>
  );
}

function RunWorkspace({ runs, detail, busy, onOpen, onDelete }: { runs: RagEvalRun[]; detail: RagEvalRunDetail | null; busy: boolean; onOpen: (id: string) => void; onDelete: (run: RagEvalRun) => Promise<void> }) {
  const finalMetrics = useMemo(() => detail?.metrics.filter((item) => item.scope === "run") ?? [], [detail]);
  const operations = useMemo(() => {
    const rows = detail?.metrics.filter((item) => item.scope === "operation") ?? [];
    return Array.from(new Set(rows.map((item) => item.operation).filter((item): item is string => Boolean(item))))
      .sort(compareRetrievalOperations);
  }, [detail]);
  return (
    <section className="panel grid">
      <h2>评测 Run</h2>
      <div className="rag-eval-run-list">
        {runs.map((run) => (
          <div className="rag-eval-run-item" key={run.id}>
            <button className={`rag-eval-run ${detail?.run.id === run.id ? "active" : ""}`} disabled={busy} onClick={() => onOpen(run.id)}>
              <span><strong>{run.dataset_name} v{run.version_number}</strong><small>{new Date(run.created_at).toLocaleString()}</small></span>
              <span className={`badge ${run.status}`}>{run.status}</span>
              <span>{run.completed_case_count}/{run.total_case_count}</span>
            </button>
            {run.status !== "queued" && run.status !== "running" ? (
              <button className="button danger rag-eval-run-delete" disabled={busy} onClick={() => void onDelete(run)}>删除</button>
            ) : null}
          </div>
        ))}
      </div>
      {detail && (
        <>
          <div className="stats rag-eval-stats">
            <MetricCard label="Hit@5" metric={metric(finalMetrics, "hit_at_5")} percent />
            <MetricCard label="Recall@10" metric={metric(finalMetrics, "recall_at_10")} percent />
            <MetricCard label="MRR" metric={metric(finalMetrics, "reciprocal_rank")} />
            <MetricCard label="nDCG@10" metric={metric(finalMetrics, "ndcg_at_10")} />
          </div>
          <div className="table-wrap">
            <table>
              <thead><tr><th>阶段</th><th>Hit@5</th><th>Recall@10</th><th>MRR</th><th>nDCG@10</th></tr></thead>
              <tbody>{operations.map((operation) => (
                <tr key={operation}>
                  <td>{operationLabels[operation] || operation}</td>
                  <td>{formatMetric(metricForOperation(detail.metrics, operation, "hit_at_5"), true)}</td>
                  <td>{formatMetric(metricForOperation(detail.metrics, operation, "recall_at_10"), true)}</td>
                  <td>{formatMetric(metricForOperation(detail.metrics, operation, "reciprocal_rank"))}</td>
                  <td>{formatMetric(metricForOperation(detail.metrics, operation, "ndcg_at_10"))}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
          <div className="rag-eval-result-cases">
            {detail.cases.map((item) => (
              <details className="rag-eval-result-case" key={item.id}>
                <summary><span>{item.query}</span><span className={`badge ${item.status}`}>{item.status}</span><span>{item.latency_ms ?? 0} ms</span></summary>
                {item.error_message && <p className="error">{item.error_message}</p>}
                {[...item.steps].sort((left, right) => compareRetrievalOperations(left.operation, right.operation)).map((step) => (
                  <div className="rag-eval-step" key={step.id}>
                    <h3>
                      {operationLabels[step.operation] || step.operation}
                      {typeof step.details.query === "string" ? <small> · {step.details.query}</small> : null}{" "}
                      <small>{step.latency_ms ?? 0} ms · {step.output.candidate_count ?? 0} candidates</small>
                    </h3>
                    <ol>{step.ranked_results.slice(0, 10).map((ranked) => (
                      <li className={ranked.matched_relevance > 0 ? "matched" : ""} key={ranked.rank}>
                        <span>#{ranked.rank}</span>
                        <div><strong>{ranked.recording_title}</strong><p>{ranked.details?.text || ranked.text || ""}</p></div>
                        <span>{ranked.matched_relevance > 0 ? `命中 R${ranked.matched_relevance}` : "未命中"}</span>
                      </li>
                    ))}</ol>
                  </div>
                ))}
              </details>
            ))}
          </div>
        </>
      )}
    </section>
  );
}

function MetricCard({ label, metric: value, percent = false, suffix = "" }: { label: string; metric: number | null; percent?: boolean; suffix?: string }) {
  return <div className="stat card"><span className="subtle">{label}</span><strong>{formatMetric(value, percent)}{value === null ? "" : suffix}</strong></div>;
}

function metric(items: RagEvalMetric[], name: string): number | null {
  const item = items.find((value) => value.metric_name === name);
  return item ? Number(item.value) : null;
}

function metricForOperation(items: RagEvalMetric[], operation: string, name: string): number | null {
  const item = items.find((value) => value.scope === "operation" && value.operation === operation && value.metric_name === name);
  return item ? Number(item.value) : null;
}

function formatMetric(value: number | null, percent = false): string {
  if (value === null || Number.isNaN(value)) return "—";
  return percent ? `${(value * 100).toFixed(1)}%` : value.toFixed(3);
}

function formatTime(milliseconds: number): string {
  const totalSeconds = Math.floor(milliseconds / 1000);
  return `${Math.floor(totalSeconds / 60)}:${String(totalSeconds % 60).padStart(2, "0")}`;
}

function statusLabel(status: RagEvalCase["status"]): string {
  return { draft: "草稿", reviewed: "已审核", approved: "已批准" }[status];
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
