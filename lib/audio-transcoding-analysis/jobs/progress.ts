import "server-only";

export interface RecordingProgress {
  recordingId: string;
  task: "transcription" | "speaker_diarization" | "speaker_identification" | "text_correction" | "embedding_indexing" | "summary";
  stage: string;
  message: string;
  percent: number;
  updatedAt: string;
}

declare global {
  // eslint-disable-next-line no-var
  var __aiRecordSummaryProgress: Map<string, RecordingProgress> | undefined;
}

function progressMap() {
  globalThis.__aiRecordSummaryProgress ??= new Map<string, RecordingProgress>();
  return globalThis.__aiRecordSummaryProgress;
}

export function setRecordingProgress(progress: Omit<RecordingProgress, "updatedAt">) {
  const value = {
    ...progress,
    updatedAt: new Date().toISOString()
  };
  progressMap().set(progress.recordingId, value);
  console.log("[progress] updated", value);
}

export function clearRecordingProgress(recordingId: string) {
  progressMap().delete(recordingId);
  console.log("[progress] cleared", { recordingId });
}

export function getRecordingProgress(recordingId: string) {
  return progressMap().get(recordingId) ?? null;
}
