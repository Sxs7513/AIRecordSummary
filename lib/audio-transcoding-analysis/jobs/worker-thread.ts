import { parentPort } from "node:worker_threads";
import { setGlobalAppConfig } from "../../config/app-config.ts";
import { diarizeRecording } from "../diarization/pyannote.ts";
import { identifyTargetSpeakers } from "../speaker-identification/speechbrain.ts";
import { transcribeRecording } from "../transcription/index.ts";

if (process.env.AI_RECORD_SUMMARY_CONFIG) {
  setGlobalAppConfig(JSON.parse(process.env.AI_RECORD_SUMMARY_CONFIG));
}

parentPort?.on("message", async (message) => {
  if (!message?.id || !message?.type) return;
  const startedAt = Date.now();
  console.log("[audio-worker] task received", {
    requestId: message.id,
    type: message.type,
    recordingId: message.recording?.id
  });
  try {
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
