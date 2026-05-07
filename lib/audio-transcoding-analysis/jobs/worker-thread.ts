import { parentPort } from "node:worker_threads";
import { setGlobalAppConfig } from "../../config/app-config.ts";
import { diarizeRecording } from "../diarization/pyannote.ts";
import { identifyTargetSpeakers } from "../speaker-identification/speechbrain.ts";
import { transcribeRecording } from "../transcription/index.ts";
import { processJob, type AudioAnalyzer } from "./process.ts";

if (process.env.AI_RECORD_SUMMARY_CONFIG) {
  setGlobalAppConfig(JSON.parse(process.env.AI_RECORD_SUMMARY_CONFIG));
}

parentPort?.on("message", async (message) => {
  if (!message?.id || !message?.type) return;
  const startedAt = Date.now();
  console.log("[audio-worker] task received", {
    requestId: message.id,
    type: message.type,
    recordingId: message.recording?.id ?? message.job?.recordingId
  });
  try {
    if (message.type === "processJob") {
      const job = message.job;
      const analyzer: AudioAnalyzer = {
        transcribe: (recording, onProgress) =>
          transcribeRecording(recording, (progress) => {
            parentPort?.postMessage({ type: "progress", requestId: message.id, recordingId: recording.id, task: "transcription", ...progress });
            onProgress?.(progress);
          }),
        diarize: (recording, onProgress) =>
          diarizeRecording(recording, (progress) => {
            parentPort?.postMessage({ type: "progress", requestId: message.id, recordingId: recording.id, task: "speaker_diarization", ...progress });
            onProgress?.(progress);
          }),
        identifySpeakers: (recording, diarizationSegments, profiles) => identifyTargetSpeakers(recording, diarizationSegments, profiles)
      };
      const result = await processJob(analyzer, job);
      console.log("[audio-worker] job completed", {
        requestId: message.id,
        jobId: job.id,
        recordingId: job.recordingId,
        jobType: job.jobType,
        status: result.status,
        durationMs: Date.now() - startedAt
      });
      parentPort?.postMessage({ id: message.id, ok: true, result: { job: result.job, status: result.status } });
      return;
    }
    if (message.type === "transcribe") {
      const result = await transcribeRecording(message.recording, (progress) => {
        parentPort?.postMessage({ type: "progress", requestId: message.id, recordingId: message.recording.id, task: "transcription", ...progress });
      });
      console.log("[audio-worker] task completed", {
        requestId: message.id,
        type: message.type,
        durationMs: Date.now() - startedAt
      });
      parentPort?.postMessage({ id: message.id, ok: true, result });
      return;
    }
    if (message.type === "diarize") {
      const result = await diarizeRecording(message.recording, (progress) => {
        parentPort?.postMessage({ type: "progress", requestId: message.id, recordingId: message.recording.id, task: "speaker_diarization", ...progress });
      });
      console.log("[audio-worker] task completed", {
        requestId: message.id,
        type: message.type,
        durationMs: Date.now() - startedAt
      });
      parentPort?.postMessage({ id: message.id, ok: true, result });
      return;
    }
    if (message.type === "identifySpeakers") {
      const result = await identifyTargetSpeakers(message.recording, message.diarizationSegments, message.profiles);
      console.log("[audio-worker] task completed", {
        requestId: message.id,
        type: message.type,
        durationMs: Date.now() - startedAt
      });
      parentPort?.postMessage({ id: message.id, ok: true, result });
      return;
    }
    throw new Error(`Unsupported audio task type: ${message.type}`);
  } catch (error) {
    console.error("[audio-worker] task failed", {
      requestId: message.id,
      type: message.type,
      durationMs: Date.now() - startedAt,
      error: error instanceof Error ? error.message : String(error)
    });
    parentPort?.postMessage({
      id: message.id,
      ok: false,
      error: error instanceof Error ? error.message : String(error)
    });
  }
});

parentPort?.postMessage({ type: "ready" });
