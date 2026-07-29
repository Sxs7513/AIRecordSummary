"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { Trash2 } from "lucide-react";

import { asrLabRequest } from "@/app/sdk/asr-lab/client";
import type { TrainingRun } from "@/app/sdk/asr-lab/types";

export function AsrTrainingRuns() {
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [error, setError] = useState<string | null>(null);
  const load = useCallback(async () => setRuns(await asrLabRequest<TrainingRun[]>("/api/training-runs")), []);
  useEffect(() => {
    void load().catch((caught) => setError(caught instanceof Error ? caught.message : String(caught)));
    const timer = window.setInterval(() => { void load(); }, 5000);
    return () => window.clearInterval(timer);
  }, [load]);

  async function cancel(runId: string) {
    setError(null);
    try {
      await asrLabRequest(`/api/training-runs/${runId}:cancel`, { method: "POST" });
      await load();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }

  async function deleteRun(run: TrainingRun) {
    if (!window.confirm(`确定删除训练任务“${run.candidate_model_name}”吗？对应的失败日志和临时训练文件也会删除。`)) return;
    setError(null);
    try {
      await asrLabRequest(`/api/training-runs/${run.id}`, { method: "DELETE" });
      await load();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }

  return (
    <>
      <div className="topbar"><div><h1>ASR 训练记录</h1><p className="subtle">跟踪离线 LoRA 任务、验证阶段和候选模型产出。</p></div><Link className="button" href="/asr-lab/datasets">创建训练任务</Link></div>
      {error ? <div className="panel error-panel">{error}</div> : null}
      <section className="panel">
        {runs.length ? <table><thead><tr><th>候选模型</th><th>数据集</th><th>基础模型</th><th>状态</th><th>进度</th><th>创建时间</th><th>操作</th></tr></thead>
          <tbody>{runs.map((run) => <tr key={run.id}>
            <td><strong>{run.candidate_model_name}</strong><br /><span className="subtle">{run.preset_name}</span></td>
            <td>{run.dataset_name} v{run.dataset_version_number}</td>
            <td>{run.base_model_name}</td>
            <td><span className={`badge ${run.status}`}>{run.status}</span>{run.error_message ? <p className="subtle">{run.error_message}</p> : null}</td>
            <td><div className="job-progress"><span>{run.progress_percent ?? 0}%</span><div className="job-progress-track"><div className="progress-bar" style={{ width: `${run.progress_percent ?? 0}%` }} /></div><span className="subtle">{run.progress_message}</span></div></td>
            <td>{new Date(run.created_at).toLocaleString()}</td>
            <td>
              {["preparing", "training", "validating"].includes(run.status) ? (
                <button className="secondary" onClick={() => void cancel(run.id)}>取消</button>
              ) : ["queued", "failed", "cancelled"].includes(run.status) ? (
                <button className="secondary" onClick={() => void deleteRun(run)}><Trash2 size={15} />删除</button>
              ) : run.status === "succeeded" ? (
                <Link className="button secondary" href="/asr-lab/evaluations">去评测</Link>
              ) : "—"}
            </td>
          </tr>)}</tbody>
        </table> : <div className="empty">还没有训练任务</div>}
      </section>
    </>
  );
}
