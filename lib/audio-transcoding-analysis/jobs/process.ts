import { alignTranscriptionSegments, applySpeakerIdentification, completeJob, failJob, generateUtteranceSegments, getRecordingForWorker, saveDiarization, saveTranscription } from "../../db/recordings";
import { listSpeakerProfiles } from "../../db/speaker-profiles";
import { redactSecrets } from "../../security/redact";
import { getAppConfig } from "../../config/app-config";
import { indexRecordingForSearch } from "../../search/indexing";
import { correctUtteranceTexts, mergeCorrectedUtterances } from "../text-correction/correct-utterance";
import type { AudioProgressEvent, DiarizationOutput, ProcessingJob, Recording, SpeakerDiarizationSegment, SpeakerIdentificationMatch, SpeakerProfileWithSamples, TranscriptionOutput } from "../../types/models";

export interface AudioAnalyzer {
  transcribe(recording: Recording, onProgress?: (progress: AudioProgressEvent) => void): Promise<TranscriptionOutput>;
  diarize(recording: Recording, onProgress?: (progress: AudioProgressEvent) => void): Promise<DiarizationOutput>;
  identifySpeakers(recording: Recording, diarizationSegments: SpeakerDiarizationSegment[], profiles: SpeakerProfileWithSamples[]): Promise<SpeakerIdentificationMatch[]>;
}

export async function processJob(analyzer: AudioAnalyzer, job: ProcessingJob) {
  const startedAt = Date.now();
  console.log("[jobs] processing job", {
    jobId: job.id,
    recordingId: job.recordingId,
    jobType: job.jobType
  });
  try {
    const context = await getRecordingForWorker(job.recordingId);

    if (job.jobType === "transcription") {
      const output = await analyzer.transcribe(context.recording);
      await saveTranscription(context.recording.id, output);
      await completeJob(job.id, { nextJobType: "speaker_diarization" });
      console.log("[jobs] transcription job done", {
        jobId: job.id,
        recordingId: job.recordingId,
        durationMs: Date.now() - startedAt
      });
      return { job, status: "completed" as const };
    }

    if (job.jobType === "speaker_diarization") {
      const freshContext = await getRecordingForWorker(job.recordingId);
      const output = await analyzer.diarize(freshContext.recording);
      await saveDiarization(freshContext.recording.id, output);
      await alignTranscriptionSegments(freshContext.recording.id);
      await generateUtteranceSegments(freshContext.recording.id);
      await completeJob(job.id, { nextJobType: "speaker_identification" });
      console.log("[jobs] diarization job done", {
        jobId: job.id,
        recordingId: job.recordingId,
        durationMs: Date.now() - startedAt
      });
      return { job, status: "completed" as const };
    }

    if (job.jobType === "speaker_identification") {
      const freshContext = await getRecordingForWorker(job.recordingId);
      const profiles = await listSpeakerProfiles();
      const matches = await analyzer.identifySpeakers(freshContext.recording, freshContext.diarizationSegments, profiles);
      await applySpeakerIdentification(freshContext.recording.id, matches);
      await completeJob(job.id, { nextJobType: "text_correction" });
      console.log("[jobs] speaker identification job done", {
        jobId: job.id,
        recordingId: job.recordingId,
        durationMs: Date.now() - startedAt
      });
      return { job, status: "completed" as const };
    }

    if (job.jobType === "text_correction") {
      await generateUtteranceSegments(context.recording.id, { correctTexts: correctUtteranceTexts, mergeUtterances: mergeCorrectedUtterances });
      await completeJob(job.id, getAppConfig().search.embeddingEnabled ? { nextJobType: "embedding_indexing" } : { recordingStatus: "completed" });
      console.log("[jobs] text correction job done", {
        jobId: job.id,
        recordingId: job.recordingId,
        durationMs: Date.now() - startedAt
      });
      return { job, status: "completed" as const };
    }

    if (job.jobType === "embedding_indexing") {
      const result = await indexRecordingForSearch(context.recording.id);
      await completeJob(job.id, { recordingStatus: "completed" });
      console.log("[jobs] embedding indexing job done", {
        jobId: job.id,
        recordingId: job.recordingId,
        chunkCount: result.chunkCount,
        skipped: result.skipped,
        durationMs: Date.now() - startedAt
      });
      return { job, status: "completed" as const };
    }

    throw new Error(`Unsupported job type: ${job.jobType}`);
  } catch (error) {
    console.error("[jobs] processing job error", {
      jobId: job.id,
      recordingId: job.recordingId,
      jobType: job.jobType,
      durationMs: Date.now() - startedAt,
      error: redactSecrets(error instanceof Error ? error.message : String(error))
    });
    await failJob(job.id, error);
    return { job, status: "failed" as const, error };
  }
}
