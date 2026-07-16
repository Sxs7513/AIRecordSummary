import type {
  Recording,
  RecordingDetail,
  RecordingSummary,
  SpeakerDiarizationSegment,
  Transcription,
  TranscriptionSegment,
  UtteranceSegment
} from "@/lib/types/models";

export interface PipelineStage {
  id: string;
  nodeName: string;
  stageName: string;
  stageVersion: string;
  required: boolean;
  resourceQueue: "cpu" | "gpu_normal" | "gpu_high";
  status: string;
  attemptCount: number;
  maxAttempts: number | null;
  progressPercent: number | null;
  progressMessage: string | null;
  progressUpdatedAt: string | null;
  errorMessage: string | null;
  startedAt: string | null;
  finishedAt: string | null;
}

export interface PipelineRun {
  id: string;
  status: string;
  errorMessage: string | null;
  startedAt: string | null;
  finishedAt: string | null;
  stages: PipelineStage[];
}

export interface PipelineRecordingDetail extends Omit<RecordingDetail, "jobs"> {
  pipelineRun: PipelineRun | null;
}

interface ApiRecording {
  id: string;
  title: string;
  file_name: string;
  storage_path: string;
  location: string | null;
  mime_type: string;
  file_size_bytes: number;
  duration_seconds: number | null;
  status: Recording["status"];
  error_message: string | null;
  uploaded_at: string;
  created_at: string;
  updated_at: string;
}

interface ApiTranscription {
  id: string;
  recording_id: string;
  language: string | null;
  model_name: string;
  full_text: string;
  segment_count: number;
  created_at: string;
  updated_at: string;
}

interface ApiRecordingSummary {
  id: string;
  recording_id: string;
  provider: string;
  model_name: string;
  summary_text: string;
  created_at: string;
  updated_at: string;
}

interface ApiTranscriptionSegment {
  id: string;
  recording_id: string;
  transcription_id: string;
  segment_index: number;
  start_ms: number;
  end_ms: number;
  text: string;
  speaker_label: string | null;
  speaker_cluster_id: string | null;
  speaker_confidence: number | null;
  is_target_person: boolean;
  target_person_confidence: number | null;
  diarization_segment_id: string | null;
  matched_speaker_profile_id: string | null;
  created_at: string;
}

interface ApiDiarizationSegment {
  id: string;
  recording_id: string;
  speaker_cluster_id: string;
  speaker_label: string;
  start_ms: number;
  end_ms: number;
  confidence: number | null;
  is_target_person: boolean;
  target_person_confidence: number | null;
  matched_speaker_profile_id: string | null;
  created_at: string;
}

interface ApiUtteranceSegment {
  id: string;
  recording_id: string;
  utterance_index: number;
  start_ms: number;
  end_ms: number;
  text: string;
  speaker_label: string | null;
  speaker_cluster_id: string | null;
  source_transcription_segment_ids: string[];
  is_target_person: boolean;
  target_person_confidence: number | null;
  matched_speaker_profile_id: string | null;
  merge_reason: string;
  created_at: string;
}

interface ApiPipelineRun {
  id: string;
  status: string;
  error_message: string | null;
  started_at: string | null;
  finished_at: string | null;
}

interface ApiStageRun {
  id: string;
  node_name: string;
  stage_name: string;
  stage_version: string;
  required: boolean;
  resource_queue: PipelineStage["resourceQueue"];
  status: string;
  attempt_count: number;
  max_attempts: number | null;
  progress_percent: number | null;
  progress_message: string | null;
  progress_updated_at: string | null;
  error_message: string | null;
  started_at: string | null;
  finished_at: string | null;
}

interface ApiRecordingDetail {
  recording: ApiRecording;
  summary: ApiRecordingSummary | null;
  transcription: ApiTranscription | null;
  transcription_segments: ApiTranscriptionSegment[];
  speaker_diarization_segments: ApiDiarizationSegment[];
  utterance_segments: ApiUtteranceSegment[];
  pipeline_runs: ApiPipelineRun[];
}

interface ApiPipelineRunDetail {
  run: ApiPipelineRun;
  stages: ApiStageRun[];
}

interface ApiRecordingList {
  items: ApiRecording[];
  total: number;
  page: number;
  page_size: number;
  stats: Record<string, number>;
}

const pythonApiOrigin = process.env.PYTHON_API_ORIGIN ?? "http://127.0.0.1:8000";

async function pythonApiGet<Response>(path: string): Promise<Response | null> {
  const response = await fetch(`${pythonApiOrigin}${path}`, { cache: "no-store" });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`Python API request failed: ${response.status}`);
  return response.json() as Promise<Response>;
}

function mapRecording(recording: ApiRecording): Recording {
  return {
    id: recording.id,
    title: recording.title,
    fileName: recording.file_name,
    storagePath: recording.storage_path,
    location: recording.location,
    mimeType: recording.mime_type,
    fileSizeBytes: recording.file_size_bytes,
    durationSeconds: recording.duration_seconds,
    processingDurationMs: null,
    status: recording.status,
    errorMessage: recording.error_message,
    uploadedAt: recording.uploaded_at,
    createdAt: recording.created_at,
    updatedAt: recording.updated_at
  };
}

function mapTranscription(value: ApiTranscription): Transcription {
  return {
    id: value.id,
    recordingId: value.recording_id,
    language: value.language,
    modelName: value.model_name,
    fullText: value.full_text,
    segmentCount: value.segment_count,
    createdAt: value.created_at,
    updatedAt: value.updated_at
  };
}

function mapSummary(value: ApiRecordingSummary): RecordingSummary {
  return {
    id: value.id,
    recordingId: value.recording_id,
    provider: value.provider === "deepseek_api" ? "deepseek_api" : "local_llm",
    modelName: value.model_name,
    summaryText: value.summary_text,
    createdAt: value.created_at,
    updatedAt: value.updated_at
  };
}

function mapTranscriptionSegment(value: ApiTranscriptionSegment): TranscriptionSegment {
  return {
    id: value.id,
    recordingId: value.recording_id,
    transcriptionId: value.transcription_id,
    segmentIndex: value.segment_index,
    startMs: value.start_ms,
    endMs: value.end_ms,
    text: value.text,
    speakerLabel: value.speaker_label,
    speakerClusterId: value.speaker_cluster_id,
    speakerConfidence: value.speaker_confidence,
    isTargetPerson: value.is_target_person,
    targetPersonConfidence: value.target_person_confidence,
    diarizationSegmentId: value.diarization_segment_id,
    matchedSpeakerProfileId: value.matched_speaker_profile_id,
    createdAt: value.created_at
  };
}

function mapDiarizationSegment(value: ApiDiarizationSegment): SpeakerDiarizationSegment {
  return {
    id: value.id,
    recordingId: value.recording_id,
    speakerClusterId: value.speaker_cluster_id,
    speakerLabel: value.speaker_label,
    startMs: value.start_ms,
    endMs: value.end_ms,
    confidence: value.confidence,
    isTargetPerson: value.is_target_person,
    targetPersonConfidence: value.target_person_confidence,
    matchedSpeakerProfileId: value.matched_speaker_profile_id,
    createdAt: value.created_at
  };
}

function mapUtteranceSegment(value: ApiUtteranceSegment): UtteranceSegment {
  return {
    id: value.id,
    recordingId: value.recording_id,
    utteranceIndex: value.utterance_index,
    startMs: value.start_ms,
    endMs: value.end_ms,
    text: value.text,
    speakerLabel: value.speaker_label,
    speakerClusterId: value.speaker_cluster_id,
    sourceTranscriptionSegmentIds: value.source_transcription_segment_ids,
    isTargetPerson: value.is_target_person,
    targetPersonConfidence: value.target_person_confidence,
    matchedSpeakerProfileId: value.matched_speaker_profile_id,
    mergeReason: value.merge_reason,
    createdAt: value.created_at
  };
}

function mapPipelineRun(value: ApiPipelineRun, stages: ApiStageRun[]): PipelineRun {
  return {
    id: value.id,
    status: value.status,
    errorMessage: value.error_message,
    startedAt: value.started_at,
    finishedAt: value.finished_at,
    stages: stages.map((stage) => ({
      id: stage.id,
      nodeName: stage.node_name,
      stageName: stage.stage_name,
      stageVersion: stage.stage_version,
      required: stage.required,
      resourceQueue: stage.resource_queue,
      status: stage.status,
      attemptCount: stage.attempt_count,
      maxAttempts: stage.max_attempts,
      progressPercent: stage.progress_percent,
      progressMessage: stage.progress_message,
      progressUpdatedAt: stage.progress_updated_at,
      errorMessage: stage.error_message,
      startedAt: stage.started_at,
      finishedAt: stage.finished_at
    }))
  };
}

export async function listPythonRecordings(params: { status?: string; page?: number; pageSize?: number }): Promise<{
  items: Recording[];
  total: number;
  stats: Record<string, number>;
  page: number;
  pageSize: number;
}> {
  const query = new URLSearchParams();
  if (params.status && params.status !== "all") query.set("status", params.status);
  query.set("page", String(params.page ?? 1));
  query.set("page_size", String(params.pageSize ?? 20));
  const response = await pythonApiGet<ApiRecordingList>(`/api/recordings?${query}`);
  if (!response) throw new Error("Python recordings API is unavailable");
  return {
    items: response.items.map(mapRecording),
    total: response.total,
    stats: response.stats,
    page: response.page,
    pageSize: response.page_size
  };
}

export async function getPythonRecordingDetail(recordingId: string): Promise<PipelineRecordingDetail | null> {
  const response = await pythonApiGet<ApiRecordingDetail>(`/api/recordings/${recordingId}`);
  if (!response) return null;
  const latestRun = response.pipeline_runs[0];
  const runDetail = latestRun ? await pythonApiGet<ApiPipelineRunDetail>(`/api/recordings/pipeline-runs/${latestRun.id}`) : null;
  return {
    recording: mapRecording(response.recording),
    summary: response.summary ? mapSummary(response.summary) : null,
    transcription: response.transcription ? mapTranscription(response.transcription) : null,
    transcriptionSegments: response.transcription_segments.map(mapTranscriptionSegment),
    speakerDiarizationSegments: response.speaker_diarization_segments.map(mapDiarizationSegment),
    utteranceSegments: response.utterance_segments.map(mapUtteranceSegment),
    speakerProfiles: [],
    pipelineRun: latestRun ? mapPipelineRun(latestRun, runDetail?.stages ?? []) : null
  };
}
