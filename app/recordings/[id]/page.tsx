import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeft } from "lucide-react";
import { AutoRefresh } from "@/components/auto-refresh";
import { MarkdownSummary } from "@/components/markdown-summary";
import { RecordingPlayer } from "@/components/recording-player";
import { RecordingLocationForm } from "@/components/recording-location-form";
import { RecordingProgress } from "@/components/recording-progress";
import { RecordingTitleForm } from "@/components/recording-title-form";
import { SpeakerLabelForm } from "@/components/speaker-label-form";
import { StatusBadge } from "@/components/status-badge";
import { UtteranceList } from "@/components/utterance-list";
import { getRecordingDetail } from "@/lib/db/recordings";
import { publicFileUrl } from "@/lib/storage/local-storage";
import { formatBytes, formatMs } from "@/lib/types/format";

export const dynamic = "force-dynamic";

function shortError(message: string) {
  return message.length > 42 ? `${message.slice(0, 42)}...` : message;
}

export default async function RecordingDetailPage({ params, searchParams }: { params: Promise<{ id: string }>; searchParams: Promise<{ t?: string; chunk?: string }> }) {
  const { id } = await params;
  const search = await searchParams;
  const detail = await getRecordingDetail(id);
  if (!detail) notFound();

  const profileById = new Map(detail.speakerProfiles.map((profile) => [profile.id, profile]));
  const shouldRefresh = detail.recording.status === "uploaded" || detail.recording.status === "processing";
  const seekMs = search.t && Number.isFinite(Number(search.t)) ? Number(search.t) : null;
  const speakerLabels = Array.from(
    new Set(
      [...detail.utteranceSegments, ...detail.transcriptionSegments, ...detail.speakerDiarizationSegments]
        .map((segment) => segment.speakerLabel)
        .filter((label): label is string => Boolean(label))
    )
  ).sort((a, b) => a.localeCompare(b, "zh-CN"));

  return (
    <>
      <AutoRefresh enabled={shouldRefresh} />
      <div className="topbar">
        <div>
          <h1>{detail.recording.title}</h1>
          <p className="subtle">
            {detail.recording.fileName} · {formatBytes(detail.recording.fileSizeBytes)} ·{" "}
            {detail.recording.location ? `${detail.recording.location} · ` : ""}
            <StatusBadge status={detail.recording.status} />
          </p>
        </div>
        <Link className="button secondary" href="/recordings">
          <ArrowLeft size={16} />
          返回
        </Link>
      </div>

      <details className="panel collapsible-panel" style={{ marginTop: 16 }} open>
        <summary>音频与任务状态</summary>
        <section className="grid" style={{ gridTemplateColumns: "minmax(280px,1fr) minmax(280px,1fr)", marginTop: 14 }}>
          <div>
            <h2>音频</h2>
            <p className="subtle">播放器已固定在页面底部，滚动查看文字记录时也可以操作。</p>
            {detail.recording.errorMessage ? (
	            <p className="subtle">
	              转码异常，错误信息：
	              <span className="error-tooltip" data-tooltip={detail.recording.errorMessage} title={detail.recording.errorMessage}>
	                <span className="error-chip">{shortError(detail.recording.errorMessage)}</span>
	              </span>
	            </p>
          ) : null}
          </div>

          <div>
            <h2>任务状态</h2>
          <table>
            <thead>
              <tr>
                <th>任务</th>
                <th>状态</th>
                <th>进度</th>
                <th>次数</th>
                <th>错误</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {detail.jobs.map((job) => (
                <tr key={job.id}>
                  <td>{job.jobType}</td>
                  <td>
                    <StatusBadge status={job.status} />
                  </td>
                  <td>
                    <RecordingProgress recordingId={detail.recording.id} status={job.status} />
                  </td>
                  <td>{job.attemptCount}</td>
	                  <td>
	                    {job.errorMessage ? (
	                      <span className="error-tooltip" data-tooltip={job.errorMessage} title={job.errorMessage}>
	                        <span className="error-chip">{shortError(job.errorMessage)}</span>
	                      </span>
	                    ) : (
                      ""
                    )}
                  </td>
                  <td>
                    {job.status === "failed" || (job.status === "completed" && (job.jobType === "text_correction" || job.jobType === "embedding_indexing" || job.jobType === "summary")) ? (
                      <form action={`/api/jobs/${job.id}/retry`} method="post">
                        <button className="secondary" type="submit">
                          {job.jobType === "text_correction" ? "重新校正" : job.jobType === "embedding_indexing" ? "重新索引" : job.jobType === "summary" ? "重新总结" : "重试"}
                        </button>
                      </form>
                    ) : (
                      ""
                    )}
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
          <SpeakerLabelForm recordingId={detail.recording.id} speakerLabels={speakerLabels} />
        ) : (
          <div className="empty">录音解析完成后可配置说话人名称</div>
        )}
        </div>
      </details>

      <details className="panel collapsible-panel" style={{ marginTop: 16 }} open>
        <summary>录音总结</summary>
        <div className="collapsible-body">
        {detail.summary ? (
          <MarkdownSummary markdown={detail.summary.summaryText} />
        ) : (
          <div className="empty">总结尚未生成</div>
        )}
        </div>
      </details>

      <details className="panel collapsible-panel" style={{ marginTop: 16 }} open>
        <summary>文字记录</summary>
        <div className="collapsible-body">
        <UtteranceList segments={detail.utteranceSegments} speakerProfiles={detail.speakerProfiles} highlightMs={seekMs} />
        </div>
      </details>

      <details className="panel collapsible-panel" style={{ marginTop: 16 }}>
        <summary>完整转写</summary>
        <div className="collapsible-body">
        {detail.transcription ? (
          <div className="full-text">
            {detail.transcriptionSegments.length > 0
              ? detail.transcriptionSegments.map((segment) => <p key={segment.id}>{segment.text}</p>)
              : detail.transcription.fullText
                  .split(/(?<=[。！？!?])/)
                  .filter(Boolean)
                  .map((sentence, index) => <p key={index}>{sentence.trim()}</p>)}
          </div>
        ) : (
          <div className="empty">转写结果尚未生成</div>
        )}
        </div>
      </details>

      <details className="panel collapsible-panel" style={{ marginTop: 16 }}>
        <summary>转写分段</summary>
        <div className="collapsible-body">
        {detail.transcriptionSegments.length === 0 ? (
          <div className="empty">暂无分段</div>
        ) : (
          <div className="segments">
            {detail.transcriptionSegments.map((segment) => {
              const profile = segment.matchedSpeakerProfileId ? profileById.get(segment.matchedSpeakerProfileId) : null;
              return (
                <article className={`segment ${segment.isTargetPerson ? "target" : ""}`} key={segment.id}>
                  <div className="segment-head">
                    <div className="meta">
                      <span>
                        {formatMs(segment.startMs)} - {formatMs(segment.endMs)}
                      </span>
                      <strong>{segment.speakerLabel || "Unknown Speaker"}</strong>
                      {segment.speakerConfidence ? <span>speaker {(segment.speakerConfidence * 100).toFixed(0)}%</span> : null}
                    </div>
                    {segment.isTargetPerson ? (
                      <span className="badge matched">
                        目标人物 {profile?.displayName || ""} {segment.targetPersonConfidence ? `${(segment.targetPersonConfidence * 100).toFixed(0)}%` : ""}
                      </span>
                    ) : null}
                  </div>
                  <div>{segment.text}</div>
                </article>
              );
            })}
          </div>
        )}
        </div>
      </details>

      <details className="panel collapsible-panel" style={{ marginTop: 16 }}>
        <summary>Speaker Diarization</summary>
        <div className="collapsible-body">
        {detail.speakerDiarizationSegments.length === 0 ? (
          <div className="empty">暂无说话人片段</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>时间</th>
                <th>标签</th>
                <th>Cluster</th>
                <th>置信度</th>
                <th>目标人物</th>
              </tr>
            </thead>
            <tbody>
              {detail.speakerDiarizationSegments.map((segment) => (
                <tr key={segment.id}>
                  <td>
                    {formatMs(segment.startMs)} - {formatMs(segment.endMs)}
                  </td>
                  <td>{segment.speakerLabel}</td>
                  <td>{segment.speakerClusterId}</td>
                  <td>{segment.confidence ? `${(segment.confidence * 100).toFixed(0)}%` : "-"}</td>
                  <td>{segment.isTargetPerson ? <StatusBadge status="matched" /> : <span className="badge">not_matched</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        </div>
      </details>

      <div className="fixed-audio-player">
        <div className="fixed-audio-player-inner">
          <div className="fixed-audio-meta">
            <strong>{detail.recording.title}</strong>
            <span>{detail.recording.fileName}</span>
          </div>
          <RecordingPlayer src={publicFileUrl(detail.recording.storagePath)} seekMs={seekMs} />
        </div>
      </div>
    </>
  );
}
