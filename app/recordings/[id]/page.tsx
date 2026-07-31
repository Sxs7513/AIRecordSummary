import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeft } from "lucide-react";
import { AutoRefresh } from "@/components/auto-refresh";
import { DeleteRecordingButton } from "@/components/delete-recording-button";
import { RecordingLocationForm } from "@/components/recording-location-form";
import { RecordingPlayer } from "@/components/recording-player";
import { RecordingTitleForm } from "@/components/recording-title-form";
import { RetryEmbeddingIndexButton } from "@/components/retry-embedding-index-button";
import { RetryRecordingButton } from "@/components/retry-recording-button";
import { SpeakerLabelForm } from "@/components/speaker-label-form";
import { StatusBadge } from "@/components/status-badge";
import { RecordingSummaryPanel } from "@/components/recording-summary-panel";
import { UtteranceList } from "@/components/utterance-list";
import { getPythonRecordingDetail } from "@/app/data/recordings";
import { formatBytes, formatMs } from "@/app/shared/format";

export const dynamic = "force-dynamic";

function shortError(message: string) {
  return message.length > 42 ? `${message.slice(0, 42)}...` : message;
}

const stageLabels: Record<string, string> = {
  normalize_audio: "音频标准化",
  diarize_pyannote: "说话人分离",
  preprocess_asr_audio: "ASR 音频预处理",
  transcribe_qwen_asr: "Qwen ASR 转写",
  transcribe_funasr_nano: "Fun-ASR-Nano 转写",
  correct_asr_windows: "文本校正与润色",
  correct_text: "文本校正与润色（旧版）",
  align_transcript: "文字与录音时间轴对齐",
  build_utterances: "生成最终转写段落",
  build_search_chunks: "构建检索文本分段",
  embedding_indexing: "生成向量索引",
  generate_summary: "生成录音总结"
};

function stageAttempts(attemptCount: number, maxAttempts: number | null): string {
  const limit = maxAttempts === null ? "不限次数" : `最多 ${maxAttempts} 次`;
  return attemptCount === 0 ? `尚未执行（${limit}）` : `已执行 ${attemptCount} 次（${limit}）`;
}

export default async function RecordingDetailPage({
  params,
  searchParams
}: {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ t?: string; end?: string }>;
}) {
  const { id } = await params;
  const search = await searchParams;
  const detail = await getPythonRecordingDetail(id);
  if (!detail) notFound();

  const pipelineRun = detail.pipelineRun;
  const shouldRefresh =
    detail.recording.status === "uploaded" ||
    detail.recording.status === "processing" ||
    pipelineRun?.status === "queued" ||
    pipelineRun?.status === "running";
  const seekMs = search.t && Number.isFinite(Number(search.t)) ? Number(search.t) : null;
  const requestedEndMs = search.end && Number.isFinite(Number(search.end)) ? Number(search.end) : null;
  const highlightRange = seekMs === null
    ? null
    : { startMs: seekMs, endMs: Math.max(seekMs, requestedEndMs ?? seekMs) };
  const speakers = Array.from(
    [...detail.speakerDiarizationSegments, ...detail.utteranceSegments, ...detail.transcriptionSegments]
      .reduce((byCluster, segment) => {
        if (segment.speakerClusterId && !byCluster.has(segment.speakerClusterId)) {
          byCluster.set(segment.speakerClusterId, {
            speakerClusterId: segment.speakerClusterId,
            displayName: segment.speakerLabel || segment.speakerClusterId
          });
        }
        return byCluster;
      }, new Map<string, { speakerClusterId: string; displayName: string }>())
      .values()
  ).sort((left, right) => left.speakerClusterId.localeCompare(right.speakerClusterId, "zh-CN"));
  const summaryGenerationRunId = pipelineRun?.stages.find((stage) => stage.stageName === "generate_summary")?.generationRunId ?? null;

  return (
    <>
      <AutoRefresh enabled={shouldRefresh} />
      <div className="topbar">
        <div>
          <h1>{detail.recording.title}</h1>
          <p className="subtle">
            {detail.recording.fileName} · {formatBytes(detail.recording.fileSizeBytes)} · {detail.recording.location ? `${detail.recording.location} · ` : ""}
            <StatusBadge status={detail.recording.status} />
          </p>
        </div>
        <div className="toolbar">
          <DeleteRecordingButton recordingId={detail.recording.id} />
          <Link className="button secondary" href="/recordings">
            <ArrowLeft size={16} />
            返回
          </Link>
        </div>
      </div>

      <details className="panel collapsible-panel" style={{ marginTop: 16 }} open>
        <summary>流水线状态</summary>
        <section style={{ marginTop: 14 }}>
          <div>
            <h2>流水线节点</h2>
            {pipelineRun ? (
              <p className="subtle">
                本次运行：<StatusBadge status={pipelineRun.status} />
                {pipelineRun.status === "failed" ? <RetryRecordingButton recordingId={detail.recording.id} /> : null}
              </p>
            ) : (
              <p className="subtle">尚未创建流水线运行</p>
            )}
            <table>
              <thead>
                <tr>
                  <th>节点</th>
                  <th>状态</th>
                  <th>进度</th>
                  <th>执行次数</th>
                  <th>错误</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {pipelineRun?.stages.map((stage) => (
                  <tr key={stage.id}>
                    <td>{stageLabels[stage.nodeName] ?? stage.nodeName}</td>
                    <td><StatusBadge status={stage.status} /></td>
                    <td>{stage.progressPercent === null ? "-" : `${stage.progressPercent}%${stage.progressMessage ? ` · ${stage.progressMessage}` : ""}`}</td>
                    <td>{stageAttempts(stage.attemptCount, stage.maxAttempts)}</td>
                    <td>
                      {stage.errorMessage ? (
                        <span className="error-tooltip" data-tooltip={stage.errorMessage} title={stage.errorMessage}>
                          <span className="error-chip">{shortError(stage.errorMessage)}</span>
                        </span>
                      ) : ""}
                    </td>
                    <td>
                      {stage.nodeName === "embedding_indexing" ? (
                        <RetryEmbeddingIndexButton recordingId={detail.recording.id} stageStatus={stage.status} />
                      ) : null}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </details>

      <details className="panel collapsible-panel" style={{ marginTop: 16 }}>
        <summary>录音信息</summary>
        <div className="collapsible-body">
          <RecordingTitleForm recordingId={detail.recording.id} title={detail.recording.title} />
          <RecordingLocationForm recordingId={detail.recording.id} location={detail.recording.location} />
        </div>
      </details>

      <details className="panel collapsible-panel" style={{ marginTop: 16 }}>
        <summary>说话人配置</summary>
        <div className="collapsible-body">
          {detail.recording.status === "completed" ? (
            <SpeakerLabelForm recordingId={detail.recording.id} speakers={speakers} />
          ) : (
            <div className="empty">录音解析完成后可配置说话人名称</div>
          )}
        </div>
      </details>

      <details className="panel collapsible-panel" style={{ marginTop: 16 }} open>
        <summary>录音总结</summary>
        <div className="collapsible-body">
          <RecordingSummaryPanel
            recordingId={detail.recording.id}
            initialGenerationRunId={summaryGenerationRunId}
            persistedMarkdown={detail.summary?.summaryText ?? null}
          />
        </div>
      </details>

      <details className="panel collapsible-panel" style={{ marginTop: 16 }} open>
        <summary>文字记录</summary>
        <div className="collapsible-body">
          <UtteranceList segments={detail.utteranceSegments} tokens={detail.transcriptionTokens} speakerProfiles={detail.speakerProfiles} highlightRange={highlightRange} />
        </div>
      </details>

      <details className="panel collapsible-panel" style={{ marginTop: 16 }}>
        <summary>完整转写</summary>
        <div className="collapsible-body">
          {detail.transcription ? (
            <div className="full-text">
              {detail.transcriptionSegments.length > 0
                ? detail.transcriptionSegments.map((segment) => <p key={segment.id}>{segment.text}</p>)
                : detail.transcription.fullText.split(/(?<=[。！？!?])/).filter(Boolean).map((sentence, index) => <p key={index}>{sentence.trim()}</p>)}
            </div>
          ) : <div className="empty">转写结果尚未生成</div>}
        </div>
      </details>

      <details className="panel collapsible-panel" style={{ marginTop: 16 }}>
        <summary>转写分段</summary>
        <div className="collapsible-body">
          {detail.transcriptionSegments.length === 0 ? <div className="empty">暂无分段</div> : (
            <div className="segments">
              {detail.transcriptionSegments.map((segment) => (
                <article className={`segment ${segment.isTargetPerson ? "target" : ""}`} key={segment.id}>
                  <div className="segment-head">
                    <div className="meta">
                      <span>{formatMs(segment.startMs)} - {formatMs(segment.endMs)}</span>
                      <strong>{segment.speakerLabel || "Unknown Speaker"}</strong>
                      {segment.speakerConfidence ? <span>speaker {(segment.speakerConfidence * 100).toFixed(0)}%</span> : null}
                    </div>
                  </div>
                  <div>{segment.text}</div>
                </article>
              ))}
            </div>
          )}
        </div>
      </details>

      <details className="panel collapsible-panel" style={{ marginTop: 16 }}>
        <summary>Speaker Diarization</summary>
        <div className="collapsible-body">
          {detail.speakerDiarizationSegments.length === 0 ? <div className="empty">暂无说话人片段</div> : (
            <table>
              <thead><tr><th>时间</th><th>标签</th><th>Cluster</th><th>置信度</th></tr></thead>
              <tbody>
                {detail.speakerDiarizationSegments.map((segment) => (
                  <tr key={segment.id}>
                    <td>{formatMs(segment.startMs)} - {formatMs(segment.endMs)}</td>
                    <td>{segment.speakerLabel}</td>
                    <td>{segment.speakerClusterId}</td>
                    <td>{segment.confidence ? `${(segment.confidence * 100).toFixed(0)}%` : "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </details>

      <div className="fixed-audio-player">
        <div className="fixed-audio-player-inner">
          <div className="fixed-audio-meta"><strong>{detail.recording.title}</strong><span>{detail.recording.fileName}</span></div>
          <RecordingPlayer src={`/api/recordings/${encodeURIComponent(detail.recording.id)}/audio`} seekMs={seekMs} />
        </div>
      </div>
    </>
  );
}
