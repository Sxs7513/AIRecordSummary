"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { Check, Database, Play, Plus, Upload } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";

import { asrLabRequest, assetAudioUrl } from "@/app/sdk/asr-lab/client";
import type {
  Annotation,
  Dataset,
  DatasetDetail,
  DatasetPreview,
  DatasetVersion,
  ModelVersion,
  SourceAsset
} from "@/app/sdk/asr-lab/types";

const VERSION_OPTIONS = { normalization_name: "zh_asr", normalization_version: "v1", seed: "asr-lab-v1" };

export function AsrDatasetWorkspace() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [detail, setDetail] = useState<DatasetDetail | null>(null);
  const [selectedAssetId, setSelectedAssetId] = useState<string>("");
  const [editing, setEditing] = useState<Annotation | null>(null);
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [models, setModels] = useState<ModelVersion[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const selectedDatasetId = searchParams.get("dataset") || datasets[0]?.id || "";

  const loadDatasets = useCallback(async () => {
    const items = await asrLabRequest<Dataset[]>("/api/evaluation/datasets");
    setDatasets(items);
    if (!searchParams.get("dataset") && items[0]) router.replace(`/asr-lab/datasets?dataset=${items[0].id}`);
  }, [router, searchParams]);

  const loadDetail = useCallback(async () => {
    if (!selectedDatasetId) {
      setDetail(null);
      return;
    }
    const result = await asrLabRequest<DatasetDetail>(`/api/evaluation/datasets/${selectedDatasetId}`);
    setDetail(result);
    setSelectedAssetId((current) => current && result.assets.some((item) => item.id === current) ? current : result.assets[0]?.id || "");
  }, [selectedDatasetId]);

  useEffect(() => { void loadDatasets().catch(showError(setError)); }, [loadDatasets]);
  useEffect(() => { void loadDetail().catch(showError(setError)); }, [loadDetail]);

  async function refresh() {
    await Promise.all([loadDatasets(), loadDetail()]);
  }

  async function createDataset(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    const formData = new FormData(form);
    setBusy(true);
    setError(null);
    try {
      const created = await asrLabRequest<Dataset>("/api/evaluation/datasets", {
        method: "POST",
        body: JSON.stringify({ name: formData.get("name"), description: formData.get("description") || null })
      });
      form.reset();
      await loadDatasets();
      router.push(`/asr-lab/datasets?dataset=${created.id}`);
    } catch (caught) {
      setError(message(caught));
    } finally {
      setBusy(false);
    }
  }

  async function uploadAsset(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedDatasetId) return;
    const form = event.currentTarget;
    setBusy(true);
    setError(null);
    try {
      const asset = await asrLabRequest<SourceAsset>(`/api/evaluation/datasets/${selectedDatasetId}/assets`, {
        method: "POST",
        body: new FormData(form)
      });
      form.reset();
      await loadDetail();
      setSelectedAssetId(asset.id);
    } catch (caught) {
      setError(message(caught));
    } finally {
      setBusy(false);
    }
  }

  async function transition(annotation: Annotation, action: "review" | "approve") {
    setBusy(true);
    setError(null);
    try {
      await asrLabRequest(`/api/evaluation/annotations/${annotation.id}:${action}`, {
        method: "POST",
        body: JSON.stringify({ revision: annotation.revision })
      });
      await refresh();
    } catch (caught) {
      setError(message(caught));
    } finally {
      setBusy(false);
    }
  }

  async function previewVersion() {
    if (!selectedDatasetId) return;
    setBusy(true);
    setError(null);
    try {
      setPreview(await asrLabRequest<DatasetPreview>(`/api/evaluation/datasets/${selectedDatasetId}/versions:preview`, {
        method: "POST",
        body: JSON.stringify(VERSION_OPTIONS)
      }));
    } catch (caught) {
      setError(message(caught));
    } finally {
      setBusy(false);
    }
  }

  async function freezeVersion() {
    if (!selectedDatasetId) return;
    setBusy(true);
    setError(null);
    try {
      await asrLabRequest(`/api/evaluation/datasets/${selectedDatasetId}/versions:freeze`, {
        method: "POST",
        body: JSON.stringify(VERSION_OPTIONS)
      });
      setPreview(null);
      await refresh();
    } catch (caught) {
      setError(message(caught));
    } finally {
      setBusy(false);
    }
  }

  async function openTraining() {
    setModels(await asrLabRequest<ModelVersion[]>("/api/model-versions"));
  }

  const asset = detail?.assets.find((item) => item.id === selectedAssetId) ?? null;
  const annotations = useMemo(
    () => (detail?.annotations ?? []).filter((item) => item.source_asset_id === selectedAssetId),
    [detail, selectedAssetId]
  );
  const latestVersion = detail?.versions.find((item) => item.status === "frozen") ?? null;

  return (
    <>
      <div className="topbar">
        <div>
          <h1>ASR 数据标注</h1>
          <p className="subtle">上传完整录音，确认区间参考文本，并冻结为可复现的训练与评测数据。</p>
        </div>
        <div className="toolbar">
          <button className="secondary" disabled={!detail || busy} onClick={() => void previewVersion()}>
            <Database size={16} />预览数据集
          </button>
          <button disabled={!latestVersion || busy} onClick={() => void openTraining()}>创建训练任务</button>
        </div>
      </div>

      {error ? <div className="panel error-panel">{error}</div> : null}

      <section className="panel">
        <form className="toolbar" onSubmit={createDataset}>
          <label>新数据集<input name="name" placeholder="例如：客服录音第一批" required /></label>
          <label>说明<input name="description" placeholder="可选" /></label>
          <button disabled={busy}><Plus size={16} />创建</button>
        </form>
        {datasets.length ? (
          <label style={{ marginTop: 14 }}>
            当前数据集
            <select
              value={selectedDatasetId}
              onChange={(event) => router.push(`/asr-lab/datasets?dataset=${event.target.value}`)}
            >
              {datasets.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
            </select>
          </label>
        ) : <div className="empty" style={{ marginTop: 14 }}>先创建一个 ASR 数据集</div>}
      </section>

      {detail ? (
        <>
          <section className="grid stats asr-stats">
            <Stat label="草稿" value={detail.annotations.filter((item) => item.status === "draft").length} />
            <Stat label="已复核" value={detail.annotations.filter((item) => item.status === "reviewed").length} />
            <Stat label="已确认" value={detail.annotations.filter((item) => item.status === "approved").length} />
            <Stat label="冻结版本" value={detail.versions.filter((item) => item.status === "frozen").length} />
          </section>

          <section className="panel" style={{ marginTop: 16 }}>
            <form className="toolbar" onSubmit={uploadAsset}>
              <label>完整录音<input name="audio" type="file" accept="audio/*" required /></label>
              <button disabled={busy}><Upload size={16} />上传到当前数据集</button>
            </form>
          </section>

          <section className="asr-annotation-layout">
            <aside className="panel asr-asset-list">
              <h2>录音列表</h2>
              {detail.assets.length ? detail.assets.map((item) => (
                <button
                  className={`asr-asset-button ${item.id === selectedAssetId ? "active" : ""}`}
                  key={item.id}
                  onClick={() => { setSelectedAssetId(item.id); setEditing(null); }}
                >
                  <strong>{item.file_name}</strong>
                  <span>{formatDuration(item.duration_ms)} · {item.annotation_count ?? 0} 段</span>
                </button>
              )) : <div className="empty">还没有上传录音</div>}
            </aside>

            <main className="panel">
              {asset ? (
                <SegmentEditor
                  key={`${asset.id}:${editing?.id ?? "new"}`}
                  asset={asset}
                  editing={editing}
                  busy={busy}
                  onSaved={async () => { setEditing(null); await refresh(); }}
                  onError={setError}
                />
              ) : <div className="empty">选择或上传一条完整录音开始标注</div>}
            </main>
          </section>

          <section className="panel" style={{ marginTop: 16 }}>
            <h2>当前录音的标注区间</h2>
            {annotations.length ? (
              <div className="asr-annotation-list">
                {annotations.map((item) => (
                  <div className="asr-annotation-row" key={item.id}>
                    <div>
                      <strong>{formatTimestamp(item.start_ms)} – {formatTimestamp(item.end_ms)}</strong>
                      <p>{item.reference_text}</p>
                      <span className={`badge ${item.status}`}>{statusLabel(item.status)}</span>
                      <span className="subtle"> · 训练 {item.train_allowed ? "✓" : "—"} · 评测 {item.evaluation_allowed ? "✓" : "—"}</span>
                    </div>
                    <div className="toolbar">
                      <button className="secondary" onClick={() => setEditing(item)}>编辑</button>
                      {item.status === "draft" ? <button disabled={busy} onClick={() => void transition(item, "review")}>复核</button> : null}
                      {item.status === "reviewed" ? <button disabled={busy} onClick={() => void transition(item, "approve")}><Check size={15} />确认</button> : null}
                    </div>
                  </div>
                ))}
              </div>
            ) : <div className="empty">当前录音还没有标注区间</div>}
          </section>
        </>
      ) : null}

      {preview ? (
        <div className="asr-modal-backdrop">
          <section className="asr-modal">
            <h2>数据集版本预览</h2>
            <SplitTable preview={preview} />
            <p className="subtle">排除 {preview.excluded_count} 条未确认或用途不允许的数据</p>
            <p className="subtle">版本校验：{preview.checksum.slice(0, 16)}…</p>
            <div className="toolbar asr-modal-actions">
              <button className="secondary" onClick={() => setPreview(null)}>取消</button>
              <button disabled={busy} onClick={() => void freezeVersion()}>冻结新版本</button>
            </div>
          </section>
        </div>
      ) : null}

      {models.length && latestVersion ? (
        <TrainingDrawer
          models={models}
          version={latestVersion}
          onClose={() => setModels([])}
          onCreated={() => { setModels([]); router.push("/asr-lab/training-runs"); }}
          onError={setError}
        />
      ) : null}
    </>
  );
}

function SegmentEditor({
  asset,
  editing,
  busy,
  onSaved,
  onError
}: {
  asset: SourceAsset;
  editing: Annotation | null;
  busy: boolean;
  onSaved: () => Promise<void>;
  onError: (value: string | null) => void;
}) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [start, setStart] = useState(formatTimestamp(editing?.start_ms ?? 0));
  const [end, setEnd] = useState(formatTimestamp(editing?.end_ms ?? Math.min(asset.duration_ms, 10_000)));
  const [text, setText] = useState(editing?.reference_text ?? "");
  const [trainAllowed, setTrainAllowed] = useState(editing?.train_allowed ?? true);
  const [evaluationAllowed, setEvaluationAllowed] = useState(editing?.evaluation_allowed ?? true);
  const [looping, setLooping] = useState(false);

  function useCurrent(target: "start" | "end") {
    const value = formatTimestamp(Math.round((audioRef.current?.currentTime ?? 0) * 1000));
    if (target === "start") setStart(value); else setEnd(value);
  }

  function playInterval() {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = parseTimestamp(start) / 1000;
    setLooping(true);
    void audio.play();
  }

  async function save(event: FormEvent) {
    event.preventDefault();
    const payload = {
      source_asset_id: asset.id,
      start_ms: parseTimestamp(start),
      end_ms: parseTimestamp(end),
      reference_text: text,
      language: "zh",
      train_allowed: trainAllowed,
      evaluation_allowed: evaluationAllowed,
      contains_sensitive_data: editing?.contains_sensitive_data ?? false,
      ...(editing ? { revision: editing.revision } : {})
    };
    try {
      await asrLabRequest(
        editing ? `/api/evaluation/annotations/${editing.id}` : `/api/evaluation/datasets/${asset.dataset_id}/annotations`,
        { method: editing ? "PATCH" : "POST", body: JSON.stringify(payload) }
      );
      if (!editing) {
        setText("");
        setStart(end);
        setEnd(formatTimestamp(Math.min(asset.duration_ms, parseTimestamp(end) + 10_000)));
      }
      await onSaved();
    } catch (caught) {
      onError(message(caught));
    }
  }

  return (
    <form className="asr-segment-editor" onSubmit={save}>
      <div className="segment-head">
        <div><h2>{asset.file_name}</h2><span className="subtle">{formatDuration(asset.duration_ms)}</span></div>
        <button type="button" className="secondary" onClick={playInterval}><Play size={15} />循环当前区间</button>
      </div>
      <audio
        ref={audioRef}
        controls
        preload="metadata"
        src={assetAudioUrl(asset.id)}
        onPause={() => setLooping(false)}
        onTimeUpdate={(event) => {
          const audio = event.currentTarget;
          if (looping && audio.currentTime * 1000 >= parseTimestamp(end)) {
            audio.currentTime = parseTimestamp(start) / 1000;
            void audio.play();
          }
        }}
      />
      <div className="asr-time-grid">
        <label>开始时间<input value={start} onChange={(event) => setStart(event.target.value)} /></label>
        <button type="button" className="secondary" onClick={() => useCurrent("start")}>当前播放位置</button>
        <label>结束时间<input value={end} onChange={(event) => setEnd(event.target.value)} /></label>
        <button type="button" className="secondary" onClick={() => useCurrent("end")}>当前播放位置</button>
      </div>
      <label>手工校验文本<textarea rows={5} value={text} onChange={(event) => setText(event.target.value)} required /></label>
      <div className="asr-checkboxes">
        <label><input type="checkbox" checked={trainAllowed} onChange={(event) => setTrainAllowed(event.target.checked)} />允许训练</label>
        <label><input type="checkbox" checked={evaluationAllowed} onChange={(event) => setEvaluationAllowed(event.target.checked)} />允许评测</label>
      </div>
      <button disabled={busy || !text.trim()}>{editing ? "保存修改（重新进入草稿）" : "保存区间"}</button>
    </form>
  );
}

function TrainingDrawer({
  models,
  version,
  onClose,
  onCreated,
  onError
}: {
  models: ModelVersion[];
  version: DatasetVersion;
  onClose: () => void;
  onCreated: () => void;
  onError: (value: string) => void;
}) {
  const [busy, setBusy] = useState(false);
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const values = new FormData(event.currentTarget);
    setBusy(true);
    try {
      await asrLabRequest("/api/training-runs", {
        method: "POST",
        body: JSON.stringify({
          dataset_version_id: version.id,
          base_model_version_id: values.get("model"),
          preset_name: "lora_safe_v1",
          candidate_model_name: values.get("name"),
          idempotency_key: crypto.randomUUID()
        })
      });
      onCreated();
    } catch (caught) {
      onError(message(caught));
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="asr-modal-backdrop">
      <form className="asr-modal" onSubmit={submit}>
        <h2>创建 LoRA 训练任务</h2>
        <p>数据集版本：v{version.version_number} · {version.case_count} 个片段</p>
        <label>基础模型<select name="model">{models.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.version}</option>)}</select></label>
        <label>训练预设<input value="lora_safe_v1" readOnly /></label>
        <label>候选模型名称<input name="name" placeholder="qwen3-asr-domain-v1" required /></label>
        <div className="toolbar asr-modal-actions">
          <button type="button" className="secondary" onClick={onClose}>取消</button>
          <button disabled={busy}>{busy ? "创建中…" : "开始训练"}</button>
        </div>
      </form>
    </div>
  );
}

function SplitTable({ preview }: { preview: DatasetPreview }) {
  return (
    <table><thead><tr><th>Split</th><th>录音组</th><th>片段</th><th>总时长</th></tr></thead>
      <tbody>{(["train", "validation", "test"] as const).map((key) => (
        <tr key={key}><td>{key}</td><td>{preview[key].group_count}</td><td>{preview[key].case_count}</td><td>{formatDuration(preview[key].duration_ms)}</td></tr>
      ))}</tbody>
    </table>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return <div className="card stat"><span className="subtle">{label}</span><strong>{value}</strong></div>;
}

function statusLabel(value: Annotation["status"]) {
  return { draft: "草稿", reviewed: "已复核", approved: "已确认" }[value];
}

function formatTimestamp(ms: number): string {
  const hours = Math.floor(ms / 3_600_000);
  const minutes = Math.floor((ms % 3_600_000) / 60_000);
  const seconds = Math.floor((ms % 60_000) / 1000);
  const millis = ms % 1000;
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}.${String(millis).padStart(3, "0")}`;
}

function parseTimestamp(value: string): number {
  const match = /^(\d+):([0-5]\d):([0-5]\d)(?:\.(\d{1,3}))?$/.exec(value.trim());
  if (!match) throw new Error("时间格式应为 HH:MM:SS.mmm");
  return Number(match[1]) * 3_600_000 + Number(match[2]) * 60_000 + Number(match[3]) * 1000 + Number((match[4] ?? "").padEnd(3, "0"));
}

function formatDuration(ms: number): string {
  return formatTimestamp(ms).replace(/\.000$/, "");
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function showError(setter: (value: string | null) => void) {
  return (error: unknown) => setter(message(error));
}
