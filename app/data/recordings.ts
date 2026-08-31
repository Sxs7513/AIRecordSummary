import { cookies } from "next/headers";
import type { Recording, RecordingSummary, SpeakerDiarizationSegment, Transcription, TranscriptionSegment, TranscriptionToken, UtteranceSegment } from "@/app/shared/models";

export type PipelineStage = { id: string; nodeName: string; stageName: string; stageVersion: string; required: boolean; status: string; attemptCount: number; maxAttempts: number | null; progressPercent: number | null; progressMessage: string | null; progressUpdatedAt: string | null; generationRunId: string | null; errorMessage: string | null; startedAt: string | null; finishedAt: string | null };
export type PipelineRun = { id: string; status: string; errorMessage: string | null; startedAt: string | null; finishedAt: string | null; stages: PipelineStage[] };
export type PipelineRecordingDetail = { recording: Recording; summary: RecordingSummary | null; transcription: Transcription | null; transcriptionSegments: TranscriptionSegment[]; transcriptionTokens: TranscriptionToken[]; speakerDiarizationSegments: SpeakerDiarizationSegment[]; utteranceSegments: UtteranceSegment[]; speakerProfiles: []; pipelineRun: PipelineRun | null };

const pythonApiOrigin = process.env.PYTHON_API_ORIGIN ?? "http://localhost:8000";
async function get<T>(path: string): Promise<T | null> {
  try {
    const cookie = (await cookies()).toString();
    const response = await fetch(`${pythonApiOrigin}${path}`, { cache: "no-store", headers: cookie ? { cookie } : undefined });
    if (!response.ok) return null;
    return await response.json() as T;
  } catch {
    return null;
  }
}

type ApiRecording = Record<string, unknown>;
function recording(value: ApiRecording): Recording {
  return { id: String(value.id), title: String(value.title), fileName: String(value.file_name), storagePath: "", location: stringOrNull(value.location), mimeType: String(value.mime_type), fileSizeBytes: Number(value.file_size_bytes), durationSeconds: numberOrNull(value.duration_seconds), processingDurationMs: null, status: value.status as Recording["status"], errorMessage: stringOrNull(value.error_message), uploadedAt: String(value.uploaded_at), createdAt: String(value.created_at), updatedAt: String(value.updated_at) };
}
function summary(value: ApiRecording): RecordingSummary { return { id: String(value.id), recordingId: String(value.recording_id), provider: value.provider === "deepseek_api" ? "deepseek_api" : "local_llm", modelName: String(value.model_name), summaryText: String(value.summary_text), createdAt: String(value.created_at), updatedAt: String(value.updated_at) }; }
function transcription(value: ApiRecording): Transcription { return { id: String(value.id), recordingId: String(value.recording_id), language: stringOrNull(value.language), modelName: String(value.model_name), fullText: String(value.full_text), originalFullText: stringOrNull(value.original_full_text), segmentCount: Number(value.segment_count), createdAt: String(value.created_at), updatedAt: String(value.updated_at) }; }
function transcriptionSegment(value: ApiRecording): TranscriptionSegment { return { id: String(value.id), recordingId: String(value.recording_id), transcriptionId: String(value.transcription_id), segmentIndex: Number(value.segment_index), startMs: Number(value.start_ms), endMs: Number(value.end_ms), text: String(value.text), originalText: stringOrNull(value.original_text), speakerLabel: stringOrNull(value.speaker_label), speakerClusterId: stringOrNull(value.speaker_cluster_id), speakerConfidence: numberOrNull(value.speaker_confidence), isTargetPerson: Boolean(value.is_target_person), targetPersonConfidence: numberOrNull(value.target_person_confidence), diarizationSegmentId: stringOrNull(value.diarization_segment_id), matchedSpeakerProfileId: stringOrNull(value.matched_speaker_profile_id), createdAt: String(value.created_at) }; }
function transcriptionToken(value: ApiRecording): TranscriptionToken { return { id: String(value.id), recordingId: String(value.recording_id), transcriptionId: String(value.transcription_id), tokenIndex: Number(value.token_index), sourceWindowIndex: Number(value.source_window_index), text: String(value.text), startMs: Number(value.start_ms), endMs: Number(value.end_ms), speakerClusterId: stringOrNull(value.speaker_cluster_id), speakerLabel: stringOrNull(value.speaker_label), attributionStatus: String(value.attribution_status) }; }
function diarizationSegment(value: ApiRecording): SpeakerDiarizationSegment { return { id: String(value.id), recordingId: String(value.recording_id), speakerClusterId: String(value.speaker_cluster_id), speakerLabel: String(value.speaker_label), startMs: Number(value.start_ms), endMs: Number(value.end_ms), confidence: numberOrNull(value.confidence), isTargetPerson: Boolean(value.is_target_person), targetPersonConfidence: numberOrNull(value.target_person_confidence), matchedSpeakerProfileId: stringOrNull(value.matched_speaker_profile_id), createdAt: String(value.created_at) }; }
function utteranceSegment(value: ApiRecording): UtteranceSegment { return { id: String(value.id), recordingId: String(value.recording_id), utteranceIndex: Number(value.utterance_index), startMs: Number(value.start_ms), endMs: Number(value.end_ms), text: String(value.text), speakerLabel: stringOrNull(value.speaker_label), speakerClusterId: stringOrNull(value.speaker_cluster_id), sourceTranscriptionSegmentIds: Array.isArray(value.source_transcription_segment_ids) ? value.source_transcription_segment_ids.map(String) : [], isTargetPerson: Boolean(value.is_target_person), targetPersonConfidence: numberOrNull(value.target_person_confidence), matchedSpeakerProfileId: stringOrNull(value.matched_speaker_profile_id), mergeReason: String(value.merge_reason), createdAt: String(value.created_at) }; }
function stage(value: ApiRecording): PipelineStage { return { id: String(value.id), nodeName: String(value.node_name), stageName: String(value.stage_name), stageVersion: String(value.stage_version), required: Boolean(value.required), status: String(value.status), attemptCount: Number(value.attempt_count), maxAttempts: numberOrNull(value.max_attempts), progressPercent: numberOrNull(value.progress_percent), progressMessage: stringOrNull(value.progress_message), progressUpdatedAt: stringOrNull(value.progress_updated_at), generationRunId: stringOrNull(value.generation_run_id), errorMessage: stringOrNull(value.error_message), startedAt: stringOrNull(value.started_at), finishedAt: stringOrNull(value.finished_at) }; }
function stringOrNull(value: unknown): string | null { return typeof value === "string" ? value : null; }
function numberOrNull(value: unknown): number | null { return typeof value === "number" ? value : null; }

export async function listPythonRecordings(params: { status?: string; page?: number; pageSize?: number }): Promise<{ items: Recording[]; total: number; stats: Record<string, number>; page: number; pageSize: number }> {
  const query = new URLSearchParams({ page: String(params.page ?? 1), page_size: String(params.pageSize ?? 20) });
  if (params.status && params.status !== "all") query.set("status", params.status);
  const response = await get<{ items: ApiRecording[]; total: number; stats: Record<string, number>; page: number; page_size: number }>(`/api/recordings?${query}`);
  if (!response) return { items: [], total: 0, stats: {}, page: params.page ?? 1, pageSize: params.pageSize ?? 20 };
  return { items: response.items.map(recording), total: response.total, stats: response.stats, page: response.page, pageSize: response.page_size };
}

export async function getPythonRecordingDetail(recordingId: string): Promise<PipelineRecordingDetail | null> {
  const response = await get<{ recording: ApiRecording; summary: ApiRecording | null; transcription: ApiRecording | null; transcription_segments: ApiRecording[]; transcription_tokens: ApiRecording[]; speaker_diarization_segments: ApiRecording[]; utterance_segments: ApiRecording[]; pipeline_runs: ApiRecording[] }>(`/api/recordings/${recordingId}`);
  if (!response) return null;
  const latest = response.pipeline_runs[0];
  const runDetail = latest ? await get<{ run: ApiRecording; stages: ApiRecording[] }>(`/api/recordings/pipeline-runs/${latest.id}`) : null;
  return { recording: recording(response.recording), summary: response.summary ? summary(response.summary) : null, transcription: response.transcription ? transcription(response.transcription) : null, transcriptionSegments: response.transcription_segments.map(transcriptionSegment), transcriptionTokens: response.transcription_tokens.map(transcriptionToken), speakerDiarizationSegments: response.speaker_diarization_segments.map(diarizationSegment), utteranceSegments: response.utterance_segments.map(utteranceSegment), speakerProfiles: [], pipelineRun: latest ? { id: String(latest.id), status: String(latest.status), errorMessage: stringOrNull(latest.error_message), startedAt: stringOrNull(latest.started_at), finishedAt: stringOrNull(latest.finished_at), stages: (runDetail?.stages ?? []).map(stage) } : null };
}
