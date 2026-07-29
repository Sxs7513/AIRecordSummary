"use client";

import { useCallback, useEffect, useMemo, useState, type FormEvent } from "react";
import { BarChart3, Play, Trash2 } from "lucide-react";

import { asrLabRequest, assetAudioUrl } from "@/app/sdk/asr-lab/client";
import type {
  CaseResult,
  Dataset,
  DatasetDetail,
  DatasetVersion,
  EditOperation,
  EvaluationRun,
  EvaluationRunDetail,
  MetricValue,
  ModelVersion
} from "@/app/sdk/asr-lab/types";

export function AsrEvaluationWorkspace() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [versions, setVersions] = useState<DatasetVersion[]>([]);
  const [models, setModels] = useState<ModelVersion[]>([]);
  const [runs, setRuns] = useState<EvaluationRun[]>([]);
  const [detail, setDetail] = useState<EvaluationRunDetail | null>(null);
  const [selectedDataset, setSelectedDataset] = useState("");
  const [selectedRun, setSelectedRun] = useState("");
  const [filter, setFilter] = useState<"all" | "better" | "worse" | "failed">("all");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    const [datasetItems, modelItems, runItems] = await Promise.all([
      asrLabRequest<Dataset[]>("/api/evaluation/datasets"),
      asrLabRequest<ModelVersion[]>("/api/model-versions"),
      asrLabRequest<EvaluationRun[]>("/api/evaluation/runs")
    ]);
    setDatasets(datasetItems);
    setModels(modelItems);
    setRuns(runItems);
    setSelectedDataset((current) => current || datasetItems[0]?.id || "");
    setSelectedRun((current) => current || runItems[0]?.id || "");
  }, []);

  useEffect(() => { void load().catch((caught) => setError(message(caught))); }, [load]);
  useEffect(() => {
    if (!selectedDataset) return;
    void asrLabRequest<DatasetDetail>(`/api/evaluation/datasets/${selectedDataset}`)
      .then((value) => setVersions(value.versions.filter((item) => item.status === "frozen")))
      .catch((caught) => setError(message(caught)));
  }, [selectedDataset]);
  useEffect(() => {
    if (!selectedRun) {
      setDetail(null);
      return;
    }
    void asrLabRequest<EvaluationRunDetail>(`/api/evaluation/runs/${selectedRun}`)
      .then(setDetail)
      .catch((caught) => setError(message(caught)));
  }, [selectedRun, runs]);

  async function createRun(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const values = new FormData(event.currentTarget);
    const baseline = String(values.get("baseline") || "");
    const candidate = String(values.get("candidate") || "");
    if (baseline === candidate) {
      setError("基准模型和对比模型不能相同");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const run = await asrLabRequest<EvaluationRun>("/api/evaluation/runs", {
        method: "POST",
        body: JSON.stringify({
          dataset_version_id: values.get("version"),
          split: values.get("split"),
          model_version_ids: [baseline, candidate],
          normalization_name: "zh_asr",
          normalization_version: "v1",
          idempotency_key: crypto.randomUUID()
        })
      });
      await load();
      setSelectedRun(run.id);
    } catch (caught) {
      setError(message(caught));
    } finally {
      setBusy(false);
    }
  }

  async function cancelRun(run: EvaluationRun) {
    setBusy(true);
    setError(null);
    try {
      await asrLabRequest(`/api/evaluation/runs/${run.id}:cancel`, { method: "POST" });
      await load();
    } catch (caught) {
      setError(message(caught));
    } finally {
      setBusy(false);
    }
  }

  async function deleteRun(run: EvaluationRun) {
    const confirmed = window.confirm(
      `确定删除“${run.dataset_name} v${run.dataset_version_number} · ${run.split}”评测任务吗？`
      + "逐切片识别结果和评测指标会一并删除；冻结数据版本和模型不会被删除。"
    );
    if (!confirmed) return;
    setBusy(true);
    setError(null);
    try {
      await asrLabRequest(`/api/evaluation/runs/${run.id}`, { method: "DELETE" });
      setDetail(null);
      setSelectedRun("");
      await load();
    } catch (caught) {
      setError(message(caught));
    } finally {
      setBusy(false);
    }
  }

  const modelMetrics = useMemo(() => metricMap(detail?.metrics ?? []), [detail]);
  const groupedCases = useMemo(() => groupCaseResults(detail?.case_results ?? []), [detail]);
  const filteredCases = groupedCases.filter((group) => {
    if (filter === "all") return true;
    if (filter === "failed") return group.results.some((item) => item.status === "failed");
    const values = group.results.map(caseCer);
    if (values.length < 2 || values.some((item) => item === null)) return false;
    return filter === "better" ? values[1]! < values[0]! : values[1]! > values[0]!;
  });

  return (
    <>
      <div className="topbar">
        <div><h1>ASR 模型评测</h1><p className="subtle">在相同冻结测试集和标准化规则下比较训练前后的文本错误率。</p></div>
      </div>
      {error ? <div className="panel error-panel">{error}</div> : null}

      <form className="panel asr-evaluation-form" onSubmit={createRun}>
        <label>数据集
          <select value={selectedDataset} onChange={(event) => setSelectedDataset(event.target.value)}>
            {datasets.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
          </select>
        </label>
        <label>冻结版本<select name="version" required>{versions.map((item) => <option key={item.id} value={item.id}>v{item.version_number} · {item.case_count} 段</option>)}</select></label>
        <label>评测集<select name="split" defaultValue="test"><option value="test">test</option><option value="validation">validation</option></select></label>
        <label>基准模型<select name="baseline">{models.map(modelOption)}</select></label>
        <label>对比模型<select name="candidate">{models.map(modelOption)}</select></label>
        <button disabled={busy || versions.length === 0 || models.length < 2}><BarChart3 size={16} />{busy ? "创建中…" : "开始评测"}</button>
      </form>

      <section className="panel" style={{ marginTop: 16 }}>
        <div className="toolbar" style={{ alignItems: "end" }}>
          <label style={{ flex: 1 }}>历史评测
            <select value={selectedRun} onChange={(event) => setSelectedRun(event.target.value)}>
              <option value="">选择评测任务</option>
              {runs.map((item) => <option key={item.id} value={item.id}>{item.dataset_name} v{item.dataset_version_number} · {item.status} · {new Date(item.created_at).toLocaleString()}</option>)}
            </select>
          </label>
          {detail?.run.status === "running" ? (
            <button className="secondary" disabled={busy} onClick={() => void cancelRun(detail.run)}>取消评测</button>
          ) : detail ? (
            <button className="secondary" disabled={busy} onClick={() => void deleteRun(detail.run)}>
              <Trash2 size={15} />删除评测任务
            </button>
          ) : null}
        </div>
        {detail ? <RunProgress run={detail.run} /> : null}
      </section>

      {detail && detail.models.length ? (
        <>
          <section className="panel" style={{ marginTop: 16 }}>
            <h2>总体指标</h2>
            <MetricTable models={detail.models} values={modelMetrics} />
          </section>
          <section className="panel" style={{ marginTop: 16 }}>
            <div className="segment-head">
              <h2>逐区间结果</h2>
              <div className="segmented">
                {(["all", "better", "worse", "failed"] as const).map((item) => (
                  <button key={item} className={filter === item ? "active" : ""} onClick={() => setFilter(item)}>{filterLabel(item)}</button>
                ))}
              </div>
            </div>
            {filteredCases.length ? <div className="asr-case-list">{filteredCases.map((group) => <CaseComparison key={group.caseId} group={group} models={detail.models} />)}</div> : <div className="empty">当前筛选没有结果</div>}
          </section>
        </>
      ) : null}
    </>
  );
}

function RunProgress({ run }: { run: EvaluationRun }) {
  const completed = run.completed_case_count + run.failed_case_count;
  const percent = run.total_case_count ? Math.round(completed / run.total_case_count * 100) : 0;
  return (
    <div className="asr-run-progress">
      <span className={`badge ${run.status}`}>{run.status}</span>
      <span>{completed}/{run.total_case_count} 个模型样本</span>
      <div className="job-progress-track"><div className="progress-bar" style={{ width: `${percent}%` }} /></div>
      {run.error_message ? <span className="subtle">{run.error_message}</span> : null}
    </div>
  );
}

function MetricTable({ models, values }: { models: ModelVersion[]; values: Map<string, Map<string, number>> }) {
  const names = ["cer", "wer", "blank_output_rate", "average_inference_duration_ms", "p95_inference_duration_ms"];
  return (
    <table><thead><tr><th>指标</th>{models.map((item) => <th key={item.id}>{item.name}<br /><span className="subtle">{item.version}</span></th>)}{models.length === 2 ? <th>变化</th> : null}</tr></thead>
      <tbody>{names.map((name) => {
        const row = models.map((model) => values.get(model.id)?.get(name));
        const delta = row.length === 2 && row[0] !== undefined && row[1] !== undefined ? row[1] - row[0] : null;
        return <tr key={name}><td>{metricLabel(name)}</td>{row.map((value, index) => <td key={models[index].id}>{formatMetric(name, value)}</td>)}{models.length === 2 ? <td className={delta !== null && delta <= 0 ? "metric-good" : "metric-bad"}>{delta === null ? "—" : `${delta > 0 ? "↑" : "↓"} ${formatMetric(name, Math.abs(delta))}`}</td> : null}</tr>;
      })}</tbody>
    </table>
  );
}

type CaseGroup = { caseId: string; first: CaseResult; results: CaseResult[] };

function CaseComparison({ group, models }: { group: CaseGroup; models: ModelVersion[] }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <article className="asr-case-card">
      <div className="segment-head">
        <div><strong>{group.first.file_name}</strong><span className="subtle"> · {formatTimestamp(group.first.start_ms)}–{formatTimestamp(group.first.end_ms)}</span></div>
        <div className="toolbar">
          <audio id={`audio-${group.caseId}`} src={assetAudioUrl(group.first.source_asset_id)} preload="none" />
          <button className="secondary" onClick={() => playCase(group.first)}><Play size={14} />播放区间</button>
          <button className="secondary" onClick={() => setExpanded((value) => !value)}>{expanded ? "收起差异" : "查看差异"}</button>
        </div>
      </div>
      <p><span className="subtle">人工文本：</span>{group.first.reference_text_raw}</p>
      {group.results.map((result) => (
        <div key={result.model_version_id} className="asr-hypothesis">
          <strong>{models.find((item) => item.id === result.model_version_id)?.name ?? result.model_version_id}</strong>
          <span>{result.status === "failed" ? result.error_message : result.hypothesis_text_raw || "（空白输出）"}</span>
          <span className="badge">CER {formatMetric("cer", caseCer(result) ?? undefined)}</span>
          {expanded && result.details.cer?.operations ? <TranscriptDiff operations={result.details.cer.operations} /> : null}
        </div>
      ))}
    </article>
  );
}

function TranscriptDiff({ operations }: { operations: EditOperation[] }) {
  return <div className="asr-diff">{operations.map((item, index) => <span className={`diff-${item.kind}`} key={index}>{item.hypothesis ?? `−${item.reference}`}</span>)}</div>;
}

function playCase(result: CaseResult) {
  const audio = document.getElementById(`audio-${result.evaluation_case_id}`) as HTMLAudioElement | null;
  if (!audio) return;
  audio.currentTime = result.start_ms / 1000;
  void audio.play();
  const stop = () => {
    if (audio.currentTime * 1000 >= result.end_ms) {
      audio.pause();
      audio.removeEventListener("timeupdate", stop);
    }
  };
  audio.addEventListener("timeupdate", stop);
}

function groupCaseResults(results: CaseResult[]): CaseGroup[] {
  const groups = new Map<string, CaseResult[]>();
  for (const result of results) groups.set(result.evaluation_case_id, [...(groups.get(result.evaluation_case_id) ?? []), result]);
  return [...groups.entries()].map(([caseId, items]) => ({ caseId, first: items[0], results: items }));
}

function metricMap(metrics: MetricValue[]): Map<string, Map<string, number>> {
  const output = new Map<string, Map<string, number>>();
  for (const metric of metrics) {
    if (!metric.model_version_id) continue;
    const values = output.get(metric.model_version_id) ?? new Map<string, number>();
    values.set(metric.metric_name, Number(metric.value));
    output.set(metric.model_version_id, values);
  }
  return output;
}

function caseCer(result: CaseResult): number | null {
  const value = result.details.cer?.value;
  return typeof value === "number" ? value : null;
}

function modelOption(item: ModelVersion) {
  return <option key={item.id} value={item.id}>{item.name} · {item.version}</option>;
}

function metricLabel(value: string) {
  return { cer: "CER", wer: "WER", blank_output_rate: "空白输出率", average_inference_duration_ms: "平均推理耗时", p95_inference_duration_ms: "P95 推理耗时" }[value] ?? value;
}

function formatMetric(name: string, value: number | undefined): string {
  if (value === undefined) return "—";
  return name.endsWith("_ms") ? `${Math.round(value)} ms` : `${(value * 100).toFixed(2)}%`;
}

function filterLabel(value: "all" | "better" | "worse" | "failed") {
  return { all: "全部", better: "对比模型变好", worse: "对比模型变差", failed: "推理失败" }[value];
}

function formatTimestamp(ms: number) {
  const minutes = Math.floor(ms / 60_000);
  const seconds = ((ms % 60_000) / 1000).toFixed(3).padStart(6, "0");
  return `${String(Math.floor(minutes / 60)).padStart(2, "0")}:${String(minutes % 60).padStart(2, "0")}:${seconds}`;
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
