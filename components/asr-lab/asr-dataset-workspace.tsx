"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { Check, FolderInput, LockKeyhole, Play, Plus, Trash2, Upload } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";

import { asrLabRequest, assetAudioUrl } from "@/app/sdk/asr-lab/client";
import type {
  Annotation,
  Dataset,
  DatasetDetail,
  DatasetPreview,
  DatasetVersion,
  EncryptedProjectDataset,
  ModelVersion,
  SourceAsset
} from "@/app/sdk/asr-lab/types";

type LocalAudioSource = {
  file: File;
  objectUrl: string;
  durationMs: number;
};

export function AsrDatasetWorkspace() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [detail, setDetail] = useState<DatasetDetail | null>(null);
  const [localAudioSource, setLocalAudioSource] = useState<LocalAudioSource | null>(null);
  const [editing, setEditing] = useState<Annotation | null>(null);
  const [models, setModels] = useState<ModelVersion[]>([]);
  const [trainingOpen, setTrainingOpen] = useState(false);
  const [freezePreview, setFreezePreview] = useState<DatasetPreview | null>(null);
  const [projectDatasets, setProjectDatasets] = useState<EncryptedProjectDataset[] | null>(null);
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
  }, [selectedDatasetId]);

  useEffect(() => { void loadDatasets().catch(showError(setError)); }, [loadDatasets]);
  useEffect(() => { void loadDetail().catch(showError(setError)); }, [loadDetail]);
  useEffect(() => () => {
    if (localAudioSource) URL.revokeObjectURL(localAudioSource.objectUrl);
  }, [localAudioSource]);
  useEffect(() => { setLocalAudioSource(null); }, [selectedDatasetId]);

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

  async function deleteDataset() {
    if (!detail) return;
    const hasVersions = detail.versions.length > 0;
    const prompt = hasVersions
      ? `确定删除数据集“${detail.dataset.name}”吗？它已有历史训练数据，删除后会从列表隐藏，但不会破坏已有训练记录。`
      : `确定永久删除数据集“${detail.dataset.name}”吗？其中的切片和标注也会一并删除。`;
    if (!window.confirm(prompt)) return;
    setBusy(true);
    setError(null);
    try {
      const result = await asrLabRequest<{
        mode: "deleted" | "archived";
        retained_version_count: number;
      }>(`/api/evaluation/datasets/${detail.dataset.id}`, { method: "DELETE" });
      setDetail(null);
      setLocalAudioSource(null);
      const remaining = await asrLabRequest<Dataset[]>("/api/evaluation/datasets");
      setDatasets(remaining);
      router.replace(remaining[0] ? `/asr-lab/datasets?dataset=${remaining[0].id}` : "/asr-lab/datasets");
      if (result.mode === "archived") {
        window.alert(`数据集已从列表隐藏；保留了 ${result.retained_version_count} 个历史数据版本，以免破坏训练或评测记录。`);
      }
    } catch (caught) {
      setError(message(caught));
    } finally {
      setBusy(false);
    }
  }

  async function selectLocalAudioSource(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    const selectedFile = new FormData(form).get("audio");
    if (!(selectedFile instanceof File) || selectedFile.size === 0) return;
    setBusy(true);
    setError(null);
    const objectUrl = URL.createObjectURL(selectedFile);
    try {
      const durationMs = await probeBrowserAudioDuration(objectUrl);
      setLocalAudioSource({ file: selectedFile, objectUrl, durationMs });
      form.reset();
    } catch (caught) {
      URL.revokeObjectURL(objectUrl);
      setError(message(caught));
    } finally {
      setBusy(false);
    }
  }

  function clearLocalAudioSource() {
    setLocalAudioSource(null);
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

  async function deleteSourceAsset(sourceAsset: SourceAsset) {
    const annotationCount = sourceAsset.annotation_count ?? 0;
    const prompt = annotationCount
      ? `这条切片包含 ${annotationCount} 条标注。确定删除切片和这些标注吗？`
      : "确定删除这条音频切片吗？";
    if (!window.confirm(prompt)) return;
    setBusy(true);
    setError(null);
    try {
      const result = await asrLabRequest<{
        mode: "deleted" | "archived";
        retained_snapshot_reference_count: number;
      }>(`/api/evaluation/assets/${sourceAsset.id}?delete_annotations=${annotationCount > 0}`, {
        method: "DELETE"
      });
      await refresh();
      if (result.mode === "archived") {
        window.alert(`切片已从当前数据集隐藏；它仍被 ${result.retained_snapshot_reference_count} 条冻结数据引用，因此保留了音频和历史标注。`);
      }
    } catch (caught) {
      setError(message(caught));
    } finally {
      setBusy(false);
    }
  }

  async function openTraining() {
    setBusy(true);
    setError(null);
    try {
      setModels(await asrLabRequest<ModelVersion[]>("/api/model-versions"));
      setTrainingOpen(true);
    } catch (caught) {
      setError(message(caught));
    } finally {
      setBusy(false);
    }
  }

  async function openFreezePreview() {
    if (!detail) return;
    setBusy(true);
    setError(null);
    try {
      const preview = await asrLabRequest<DatasetPreview>(
        `/api/evaluation/datasets/${detail.dataset.id}/versions:preview`,
        {
          method: "POST",
          body: JSON.stringify(datasetVersionParameters())
        }
      );
      setFreezePreview(preview);
    } catch (caught) {
      setError(message(caught));
    } finally {
      setBusy(false);
    }
  }

  async function freezeDataset(preview: DatasetPreview) {
    if (!detail) return;
    setBusy(true);
    setError(null);
    try {
      const version = await asrLabRequest<DatasetVersion>(
        `/api/evaluation/datasets/${detail.dataset.id}/versions:freeze`,
        {
          method: "POST",
          body: JSON.stringify({
            ...datasetVersionParameters(),
            expected_checksum: preview.checksum
          })
        }
      );
      setFreezePreview(null);
      await refresh();
      window.alert(`数据集版本 v${version.version_number} 已冻结，共 ${version.case_count} 个切片。`);
    } catch (caught) {
      setError(message(caught));
    } finally {
      setBusy(false);
    }
  }

  async function openProjectImport() {
    setBusy(true);
    setError(null);
    try {
      setProjectDatasets(await asrLabRequest<EncryptedProjectDataset[]>("/api/evaluation/project-datasets"));
    } catch (caught) {
      setError(message(caught));
    } finally {
      setBusy(false);
    }
  }

  async function importProjectDataset(packageId: string, password: string) {
    setBusy(true);
    setError(null);
    try {
      const imported = await asrLabRequest<Dataset>(`/api/evaluation/project-datasets/${packageId}:import`, {
        method: "POST",
        body: JSON.stringify({ password })
      });
      setProjectDatasets(null);
      await loadDatasets();
      router.push(`/asr-lab/datasets?dataset=${imported.id}`);
    } catch (caught) {
      setError(message(caught));
      throw caught;
    } finally {
      setBusy(false);
    }
  }

  const assetById = useMemo(
    () => new Map((detail?.assets ?? []).map((item) => [item.id, item])),
    [detail]
  );
  return (
    <>
      <div className="topbar">
        <div>
          <h1>ASR 数据标注</h1>
          <p className="subtle">在页面中打开本地录音，只保存独立音频切片和校验文本；创建训练任务时自动保存可复现的数据版本。</p>
        </div>
        <div className="toolbar">
          <button className="secondary" disabled={busy} onClick={() => void openProjectImport()}>
            <FolderInput size={16} />导入项目数据集
          </button>
          <button
            className="secondary"
            disabled={
              !detail
              || busy
              || !detail.annotations.some(
                (item) => item.status === "approved" && (item.train_allowed || item.evaluation_allowed)
              )
            }
            onClick={() => void openFreezePreview()}
          >
            <LockKeyhole size={16} />冻结数据集
          </button>
          <button
            disabled={!detail || busy || !detail.annotations.some((item) => item.status === "approved" && item.train_allowed)}
            onClick={() => void openTraining()}
          >
            创建训练任务
          </button>
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
          <div className="toolbar" style={{ marginTop: 14, alignItems: "end" }}>
            <label style={{ flex: 1 }}>
              当前数据集
              <select
                value={selectedDatasetId}
                onChange={(event) => router.push(`/asr-lab/datasets?dataset=${event.target.value}`)}
              >
                {datasets.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
              </select>
            </label>
            <button className="secondary" disabled={!detail || busy} onClick={() => void deleteDataset()}>
              <Trash2 size={16} />删除数据集
            </button>
          </div>
        ) : <div className="empty" style={{ marginTop: 14 }}>先创建一个 ASR 数据集</div>}
      </section>

      {detail ? (
        <>
          <section className="grid stats asr-stats">
            <Stat label="草稿" value={detail.annotations.filter((item) => item.status === "draft").length} />
            <Stat label="已复核" value={detail.annotations.filter((item) => item.status === "reviewed").length} />
            <Stat label="已确认" value={detail.annotations.filter((item) => item.status === "approved").length} />
            <Stat label="历史数据版本" value={detail.versions.filter((item) => item.status === "frozen").length} />
          </section>

          <section className="panel" style={{ marginTop: 16 }}>
            <form className="toolbar" onSubmit={selectLocalAudioSource}>
              <label>本地原始录音<input name="audio" type="file" accept="audio/*" required /></label>
              <button disabled={busy}><Upload size={16} />在页面中打开</button>
            </form>
            <p className="subtle" style={{ marginTop: 10 }}>原始录音仅保存在当前浏览器页面；只有保存区间时才发送给后端即时切片，后端不会保存完整录音。</p>
          </section>

          <section className="asr-annotation-layout">
            <aside className="panel asr-asset-list">
              <h2>当前本地录音</h2>
              {localAudioSource ? (
                <div className="asr-asset-row">
                  <button
                    className="asr-asset-button active"
                    type="button"
                  >
                    <strong>{localAudioSource.file.name}</strong>
                    <span>{formatDuration(localAudioSource.durationMs)} · 仅当前页面</span>
                  </button>
                  <button
                    className="secondary asr-delete-asset"
                    disabled={busy}
                    aria-label={`关闭本地录音 ${localAudioSource.file.name}`}
                    title="关闭本地原始录音"
                    onClick={clearLocalAudioSource}
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              ) : <div className="empty">还没有打开本地录音</div>}
            </aside>

            <main className="panel">
              {localAudioSource ? (
                <SegmentEditor
                  key={localAudioSource.objectUrl}
                  source={localAudioSource}
                  datasetId={selectedDatasetId}
                  busy={busy}
                  onSaved={refresh}
                  onError={setError}
                />
              ) : <div className="empty">在页面中打开一条本地录音开始切片标注</div>}
            </main>
          </section>

          <section className="panel" style={{ marginTop: 16 }}>
            <h2>已保存切片</h2>
            {detail.annotations.length ? (
              <div className="asr-annotation-list">
                {detail.annotations.map((item) => {
                  const sampleAsset = assetById.get(item.source_asset_id);
                  return (
                  <div className="asr-annotation-row" key={item.id}>
                    <div>
                      {sampleAsset ? <audio controls preload="none" src={assetAudioUrl(sampleAsset.id)} /> : null}
                      <strong>{formatDuration(item.end_ms - item.start_ms)}</strong>
                      <p>{item.reference_text}</p>
                      <span className={`badge ${item.status}`}>{statusLabel(item.status)}</span>
                      <span className="subtle"> · 训练 {item.train_allowed ? "✓" : "—"} · 评测 {item.evaluation_allowed ? "✓" : "—"}</span>
                    </div>
                    <div className="toolbar">
                      <button className="secondary" onClick={() => setEditing(item)}>编辑</button>
                      {sampleAsset ? (
                        <button className="secondary" disabled={busy} onClick={() => void deleteSourceAsset(sampleAsset)}>
                          <Trash2 size={15} />删除切片
                        </button>
                      ) : null}
                      {item.status === "draft" ? <button disabled={busy} onClick={() => void transition(item, "review")}>复核</button> : null}
                      {item.status === "reviewed" ? <button disabled={busy} onClick={() => void transition(item, "approve")}><Check size={15} />确认</button> : null}
                    </div>
                  </div>
                  );
                })}
              </div>
            ) : <div className="empty">还没有保存任何切片</div>}
          </section>
        </>
      ) : null}

      {trainingOpen && detail ? (
        <TrainingDrawer
          models={models}
          dataset={detail.dataset}
          approvedTrainingCount={detail.annotations.filter((item) => item.status === "approved" && item.train_allowed).length}
          onClose={() => {
            setTrainingOpen(false);
            setModels([]);
          }}
          onCreated={() => {
            setTrainingOpen(false);
            setModels([]);
            router.push("/asr-lab/training-runs");
          }}
          onError={setError}
        />
      ) : null}

      {freezePreview && detail ? (
        <FreezeDatasetModal
          preview={freezePreview}
          assets={assetById}
          busy={busy}
          onClose={() => setFreezePreview(null)}
          onConfirm={freezeDataset}
        />
      ) : null}

      {projectDatasets ? (
        <ProjectDatasetImportModal
          datasets={projectDatasets}
          busy={busy}
          onClose={() => setProjectDatasets(null)}
          onImport={importProjectDataset}
        />
      ) : null}

      {editing && assetById.get(editing.source_asset_id) ? (
        <SampleTextEditorModal
          annotation={editing}
          asset={assetById.get(editing.source_asset_id)!}
          busy={busy}
          onClose={() => setEditing(null)}
          onSaved={async () => { setEditing(null); await refresh(); }}
          onError={setError}
        />
      ) : null}
    </>
  );
}

function FreezeDatasetModal({
  preview,
  assets,
  busy,
  onClose,
  onConfirm
}: {
  preview: DatasetPreview;
  assets: Map<string, SourceAsset>;
  busy: boolean;
  onClose: () => void;
  onConfirm: (preview: DatasetPreview) => Promise<void>;
}) {
  const trainingCases = preview.cases.filter((item) => item.split === "train");
  const validationCases = preview.cases.filter((item) => item.split === "validation");
  const testCases = preview.cases.filter((item) => item.split === "test");

  return (
    <div className="asr-modal-backdrop">
      <div className="asr-modal asr-freeze-modal" role="dialog" aria-modal="true" aria-labelledby="freeze-dataset-title">
        <div>
          <h2 id="freeze-dataset-title">确认冻结数据集</h2>
          <p className="subtle">
            冻结后会生成不可变版本。请确认下面的训练与评测切片划分；验证集和测试集都属于评测切片。
          </p>
        </div>
        <FreezeCaseGroup title="训练切片" summary={preview.train} cases={trainingCases} assets={assets} />
        <FreezeCaseGroup title="评测切片 · 验证集" summary={preview.validation} cases={validationCases} assets={assets} />
        <FreezeCaseGroup title="评测切片 · 测试集" summary={preview.test} cases={testCases} assets={assets} />
        {preview.excluded_count > 0 ? (
          <p className="subtle">另有 {preview.excluded_count} 个未确认或未允许用于训练/评测的切片，不会进入本次冻结版本。</p>
        ) : null}
        <div className="toolbar asr-modal-actions">
          <button type="button" className="secondary" disabled={busy} onClick={onClose}>取消</button>
          <button type="button" disabled={busy} onClick={() => void onConfirm(preview)}>
            {busy ? "正在冻结…" : `确认冻结 ${preview.cases.length} 个切片`}
          </button>
        </div>
      </div>
    </div>
  );
}

function FreezeCaseGroup({
  title,
  summary,
  cases,
  assets
}: {
  title: string;
  summary: DatasetPreview["train"];
  cases: DatasetPreview["cases"];
  assets: Map<string, SourceAsset>;
}) {
  return (
    <section className="asr-freeze-group">
      <div className="asr-freeze-group-title">
        <h3>{title}</h3>
        <span className="subtle">{summary.case_count} 个切片 · {formatDuration(summary.duration_ms)}</span>
      </div>
      {cases.length ? (
        <div className="asr-freeze-case-list">
          {cases.map((item) => {
            const annotation = item.annotation;
            const asset = assets.get(annotation.source_asset_id);
            return (
              <div className="asr-freeze-case" key={annotation.id}>
                <audio controls preload="none" src={assetAudioUrl(annotation.source_asset_id)} />
                <div>
                  <strong>{asset?.file_name ?? "音频切片"} · {formatDuration(annotation.end_ms - annotation.start_ms)}</strong>
                  <p>{annotation.reference_text}</p>
                </div>
              </div>
            );
          })}
        </div>
      ) : <div className="empty">本次没有分配到该集合的切片</div>}
    </section>
  );
}

function datasetVersionParameters() {
  return {
    normalization_name: "zh_asr",
    normalization_version: "v1",
    seed: "asr-lab-v1"
  };
}

function SegmentEditor({
  source,
  datasetId,
  busy,
  onSaved,
  onError
}: {
  source: LocalAudioSource;
  datasetId: string;
  busy: boolean;
  onSaved: () => Promise<void>;
  onError: (value: string | null) => void;
}) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [start, setStart] = useState(formatTimestamp(0));
  const [end, setEnd] = useState(formatTimestamp(Math.min(source.durationMs, 10_000)));
  const [text, setText] = useState("");
  const [trainAllowed, setTrainAllowed] = useState(true);
  const [evaluationAllowed, setEvaluationAllowed] = useState(true);
  const [looping, setLooping] = useState(false);
  const [persistEncrypted, setPersistEncrypted] = useState(false);
  const [passwordDialogOpen, setPasswordDialogOpen] = useState(false);
  const [persistencePassword, setPersistencePassword] = useState("");
  const [persistencePasswordConfirmation, setPersistencePasswordConfirmation] = useState("");
  const [saving, setSaving] = useState(false);
  const persistencePasswordsMatch = persistencePassword === persistencePasswordConfirmation;

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

  async function persist(password?: string) {
    const payload = new FormData();
    payload.set("audio", source.file);
    payload.set("start_ms", String(parseTimestamp(start)));
    payload.set("end_ms", String(parseTimestamp(end)));
    payload.set("reference_text", text);
    payload.set("language", "zh");
    payload.set("train_allowed", String(trainAllowed));
    payload.set("evaluation_allowed", String(evaluationAllowed));
    payload.set("contains_sensitive_data", "false");
    if (password) payload.set("project_persistence_password", password);
    setSaving(true);
    try {
      await asrLabRequest(
        `/api/evaluation/datasets/${datasetId}/samples`,
        { method: "POST", body: payload }
      );
      setText("");
      setStart(end);
      setEnd(formatTimestamp(Math.min(source.durationMs, parseTimestamp(end) + 10_000)));
      setPasswordDialogOpen(false);
      setPersistencePassword("");
      setPersistencePasswordConfirmation("");
      await onSaved();
    } catch (caught) {
      onError(message(caught));
    } finally {
      setSaving(false);
    }
  }

  function save(event: FormEvent) {
    event.preventDefault();
    if (persistEncrypted) {
      setPasswordDialogOpen(true);
      return;
    }
    void persist();
  }

  function saveEncrypted(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!persistencePasswordsMatch) return;
    void persist(persistencePassword);
  }

  return (
    <>
      <form className="asr-segment-editor" onSubmit={save}>
        <div className="segment-head">
          <div><h2>{source.file.name}</h2><span className="subtle">{formatDuration(source.durationMs)}</span></div>
          <button type="button" className="secondary" onClick={playInterval}><Play size={15} />循环当前区间</button>
        </div>
        <audio
          ref={audioRef}
          controls
          preload="metadata"
          src={source.objectUrl}
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
        <div className="toolbar asr-save-actions">
          <button disabled={busy || saving || !text.trim()}>{saving ? "正在切片保存…" : "保存区间"}</button>
          <label className="asr-inline-checkbox">
            <input
              type="checkbox"
              checked={persistEncrypted}
              onChange={(event) => setPersistEncrypted(event.target.checked)}
            />
            是否加密持久化
          </label>
        </div>
      </form>
      {passwordDialogOpen ? (
        <div className="asr-modal-backdrop">
          <form className="asr-modal" onSubmit={saveEncrypted}>
            <h2>加密持久化到项目</h2>
            <p className="subtle">只保存当前区间的独立音频切片和手工校验文本。请记住密码，项目不会保存密码。</p>
            <label>
              加密密码
              <input
                type="password"
                autoFocus
                minLength={8}
                maxLength={256}
                autoComplete="new-password"
                value={persistencePassword}
                onChange={(event) => setPersistencePassword(event.target.value)}
                required
              />
            </label>
            <label>
              再次输入密码
              <input
                type="password"
                minLength={8}
                maxLength={256}
                autoComplete="new-password"
                value={persistencePasswordConfirmation}
                onChange={(event) => setPersistencePasswordConfirmation(event.target.value)}
                aria-invalid={persistencePasswordConfirmation.length > 0 && !persistencePasswordsMatch}
                required
              />
              {persistencePasswordConfirmation.length > 0 && !persistencePasswordsMatch ? (
                <span className="field-error">两次输入的密码不一致</span>
              ) : null}
            </label>
            <div className="toolbar asr-modal-actions">
              <button
                type="button"
                className="secondary"
                onClick={() => {
                  setPasswordDialogOpen(false);
                  setPersistencePassword("");
                  setPersistencePasswordConfirmation("");
                }}
              >
                取消
              </button>
              <button
                disabled={
                  busy
                  || saving
                  || persistencePassword.length < 8
                  || persistencePasswordConfirmation.length < 8
                  || !persistencePasswordsMatch
                }
              >
                {saving ? "正在切片保存…" : "保存并加密持久化"}
              </button>
            </div>
          </form>
        </div>
      ) : null}
    </>
  );
}

function ProjectDatasetImportModal({
  datasets,
  busy,
  onClose,
  onImport
}: {
  datasets: EncryptedProjectDataset[];
  busy: boolean;
  onClose: () => void;
  onImport: (packageId: string, password: string) => Promise<void>;
}) {
  const [selectedId, setSelectedId] = useState(datasets[0]?.id ?? "");
  const [password, setPassword] = useState("");

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await onImport(selectedId, password).catch(() => undefined);
    setPassword("");
  }

  return (
    <div className="asr-modal-backdrop">
      <form className="asr-modal" onSubmit={submit}>
        <h2>导入项目加密数据集</h2>
        {datasets.length ? (
          <>
            <label>
              项目数据集
              <select value={selectedId} onChange={(event) => setSelectedId(event.target.value)}>
                {datasets.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.file_name} · {formatFileSize(item.file_size_bytes)}
                  </option>
                ))}
              </select>
            </label>
            <label>
              解密密码
              <input
                type="password"
                autoFocus
                minLength={8}
                maxLength={256}
                autoComplete="current-password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                required
              />
            </label>
          </>
        ) : <div className="empty">项目里还没有加密持久化的数据集</div>}
        <div className="toolbar asr-modal-actions">
          <button type="button" className="secondary" onClick={onClose}>取消</button>
          {datasets.length ? <button disabled={busy || password.length < 8}>解密并导入</button> : null}
        </div>
      </form>
    </div>
  );
}

function SampleTextEditorModal({
  annotation,
  asset,
  busy,
  onClose,
  onSaved,
  onError
}: {
  annotation: Annotation;
  asset: SourceAsset;
  busy: boolean;
  onClose: () => void;
  onSaved: () => Promise<void>;
  onError: (value: string | null) => void;
}) {
  const [text, setText] = useState(annotation.reference_text);
  const [trainAllowed, setTrainAllowed] = useState(annotation.train_allowed);
  const [evaluationAllowed, setEvaluationAllowed] = useState(annotation.evaluation_allowed);
  const [syncEncrypted, setSyncEncrypted] = useState(false);
  const [password, setPassword] = useState("");
  const [passwordConfirmation, setPasswordConfirmation] = useState("");
  const passwordsMatch = password === passwordConfirmation;

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (syncEncrypted && !passwordsMatch) return;
    try {
      await asrLabRequest(`/api/evaluation/annotations/${annotation.id}`, {
        method: "PATCH",
        body: JSON.stringify({
          source_asset_id: asset.id,
          start_ms: annotation.start_ms,
          end_ms: annotation.end_ms,
          reference_text: text,
          language: annotation.language,
          train_allowed: trainAllowed,
          evaluation_allowed: evaluationAllowed,
          contains_sensitive_data: annotation.contains_sensitive_data,
          revision: annotation.revision,
          ...(syncEncrypted ? { project_persistence_password: password } : {})
        })
      });
      setPassword("");
      setPasswordConfirmation("");
      await onSaved();
    } catch (caught) {
      onError(message(caught));
    }
  }

  return (
    <div className="asr-modal-backdrop">
      <form className="asr-modal" onSubmit={submit}>
        <h2>编辑切片文本</h2>
        <audio controls preload="none" src={assetAudioUrl(asset.id)} />
        <p className="subtle">切片音频已经独立保存；编辑时只修改文字和用途，不再依赖临时原始录音。</p>
        <label>手工校验文本<textarea rows={5} value={text} onChange={(event) => setText(event.target.value)} required /></label>
        <div className="asr-checkboxes">
          <label><input type="checkbox" checked={trainAllowed} onChange={(event) => setTrainAllowed(event.target.checked)} />允许训练</label>
          <label><input type="checkbox" checked={evaluationAllowed} onChange={(event) => setEvaluationAllowed(event.target.checked)} />允许评测</label>
          <label>
            <input
              type="checkbox"
              checked={syncEncrypted}
              onChange={(event) => {
                setSyncEncrypted(event.target.checked);
                if (!event.target.checked) {
                  setPassword("");
                  setPasswordConfirmation("");
                }
              }}
            />
            同步项目加密包
          </label>
        </div>
        {syncEncrypted ? (
          <>
            <label>
              加密密码
              <input
                type="password"
                minLength={8}
                maxLength={256}
                autoComplete="current-password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                required
              />
            </label>
            <label>
              再次输入密码
              <input
                type="password"
                minLength={8}
                maxLength={256}
                autoComplete="current-password"
                value={passwordConfirmation}
                onChange={(event) => setPasswordConfirmation(event.target.value)}
                aria-invalid={passwordConfirmation.length > 0 && !passwordsMatch}
                required
              />
              {passwordConfirmation.length > 0 && !passwordsMatch ? (
                <span className="field-error">两次输入的密码不一致</span>
              ) : null}
            </label>
          </>
        ) : null}
        <div className="toolbar asr-modal-actions">
          <button type="button" className="secondary" onClick={onClose}>取消</button>
          <button
            disabled={
              busy
              || !text.trim()
              || (
                syncEncrypted
                && (password.length < 8 || passwordConfirmation.length < 8 || !passwordsMatch)
              )
            }
          >
            保存修改（重新进入草稿）
          </button>
        </div>
      </form>
    </div>
  );
}

function TrainingDrawer({
  models,
  dataset,
  approvedTrainingCount,
  onClose,
  onCreated,
  onError
}: {
  models: ModelVersion[];
  dataset: Dataset;
  approvedTrainingCount: number;
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
          dataset_id: dataset.id,
          base_model_version_id: values.get("model"),
          preset_name: "lora_safe_v1",
          candidate_model_name: values.get("name"),
          run_validation: false,
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
        <p>数据集：{dataset.name} · {approvedTrainingCount} 个已确认训练片段</p>
        <p className="subtle">开始训练时会自动生成或复用数据版本，无需手动冻结。</p>
        <p className="subtle">当前为链路验证模式：只训练，跳过 validation 和训练后自动评测。</p>
        {models.length ? (
          <label>
            基础模型
            <select name="model" required>
              {models.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.version}</option>)}
            </select>
          </label>
        ) : (
          <div className="empty">没有可用的 Qwen3-ASR 基础模型，请确认 training-api 和评测数据库已初始化。</div>
        )}
        <label>训练预设<input value="lora_safe_v1" readOnly /></label>
        <label>候选模型名称<input name="name" placeholder="qwen3-asr-domain-v1" required /></label>
        <div className="toolbar asr-modal-actions">
          <button type="button" className="secondary" onClick={onClose}>取消</button>
          <button disabled={busy || models.length === 0}>{busy ? "创建中…" : "开始训练"}</button>
        </div>
      </form>
    </div>
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

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function probeBrowserAudioDuration(objectUrl: string): Promise<number> {
  return new Promise((resolve, reject) => {
    const audio = new Audio();
    audio.preload = "metadata";
    audio.onloadedmetadata = () => {
      if (!Number.isFinite(audio.duration) || audio.duration <= 0) {
        reject(new Error("无法读取录音时长"));
        return;
      }
      resolve(Math.round(audio.duration * 1000));
    };
    audio.onerror = () => reject(new Error("浏览器无法读取这个音频文件"));
    audio.src = objectUrl;
  });
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function showError(setter: (value: string | null) => void) {
  return (error: unknown) => setter(message(error));
}
