import "server-only";
import { randomUUID } from "node:crypto";
import path from "node:path";
import { Worker } from "node:worker_threads";
import { getAppConfig } from "../../config/app-config";
import { getNextPendingJob } from "../../db/recordings";
import { pool } from "../../db/pool";
import { clearRecordingProgress, setRecordingProgress } from "./progress";
import { processJob, type AudioAnalyzer } from "./process";
import type { DiarizationOutput, ProcessingJob, SpeakerIdentificationMatch, TranscriptionOutput } from "../../types/models";

declare global {
  // eslint-disable-next-line no-var
  var __aiRecordSummaryScheduler: EmbeddedJobScheduler | undefined;
}

type SchedulerState = "stopped" | "starting" | "running";
type PendingRequest = {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
};
type WorkerSlot = {
  id: number;
  worker: Worker;
  activeRequestId: string | null;
  reserved: boolean;
};

class EmbeddedJobScheduler {
  private state: SchedulerState = "stopped";
  private workers: WorkerSlot[] = [];
  private pending = new Map<string, PendingRequest>();
  private activeJobIds = new Set<string>();
  private activeRecordingIds = new Set<string>();
  private draining = false;
  private interval: NodeJS.Timeout | null = null;

  async start() {
    if (this.state !== "stopped") return;
    this.state = "starting";
    const config = getAppConfig();
    console.log("[jobs] starting embedded scheduler", {
      intervalMs: config.jobs.intervalMs,
      batchSize: config.jobs.batchSize,
      workerConcurrency: config.jobs.workerConcurrency
    });

    const workerPath = path.join(process.cwd(), "lib", "audio-transcoding-analysis", "jobs", "worker-thread.ts");
    console.log("[jobs] starting audio worker pool", { workerPath, workerConcurrency: config.jobs.workerConcurrency });
    for (let index = 0; index < config.jobs.workerConcurrency; index += 1) {
      this.startWorker(index + 1, workerPath, config);
    }

    const client = await pool.connect();
    client.on("notification", () => this.kick());
    client.on("error", (error) => {
      console.error("[jobs] listen connection error", error);
    });
    await client.query("listen processing_jobs");
    console.log("[jobs] listening for PostgreSQL notifications", { channel: "processing_jobs" });

    this.interval = setInterval(() => this.kick(), config.jobs.intervalMs);
    this.interval.unref();

    this.state = "running";
    this.kick();
  }

  private startWorker(workerId: number, workerPath: string, config: ReturnType<typeof getAppConfig>) {
    const worker = new Worker(workerPath, {
      execArgv: ["--import", "tsx"],
      env: {
        ...process.env,
        AI_RECORD_SUMMARY_CONFIG: JSON.stringify(config)
      }
    });
    const slot: WorkerSlot = { id: workerId, worker, activeRequestId: null, reserved: false };
    this.workers.push(slot);
    worker.unref();

    worker.on("message", (message) => {
      if (message?.id) {
        const pending = this.pending.get(message.id);
        if (!pending) return;
        this.pending.delete(message.id);
        slot.activeRequestId = null;
        if (message.ok) pending.resolve(message.result);
        else pending.reject(new Error(message.error || "Audio worker task failed"));
        this.kick();
        return;
      }
      if (message?.type === "progress") {
        setRecordingProgress({
          recordingId: message.recordingId,
          task: message.task,
          stage: message.stage,
          message: message.message,
          percent: message.percent
        });
        return;
      }
      if (message?.type === "error" || message?.type === "fatal") {
        console.error(`[jobs:worker:${workerId}]`, message.message);
        return;
      }
      if (message?.type === "ready") {
        console.log(`[jobs:worker:${workerId}] ready`);
      }
    });

    worker.on("error", (error) => {
      console.error(`[jobs:worker:${workerId}]`, error);
    });
    worker.on("exit", (code) => {
      this.workers = this.workers.filter((item) => item !== slot);
      if (slot.activeRequestId) {
        const pending = this.pending.get(slot.activeRequestId);
        this.pending.delete(slot.activeRequestId);
        pending?.reject(new Error(`Audio worker ${workerId} exited with code ${code}`));
      }
      if (code !== 0) {
        console.error(`[jobs:worker:${workerId}] exited with code ${code}`);
      }
      if (this.workers.length === 0) {
        this.state = "stopped";
        for (const pending of this.pending.values()) {
          pending.reject(new Error("Audio worker pool stopped"));
        }
        this.pending.clear();
      }
    });
  }

  kick() {
    console.log("[jobs] scheduler kicked");
    void this.drain();
  }

  private async drain() {
    if (this.draining) return;
    this.draining = true;
    try {
      const config = getAppConfig();
      let dispatchedCount = 0;
      while (dispatchedCount < config.jobs.batchSize) {
        const slot = this.findAvailableWorker();
        if (!slot) break;

        slot.reserved = true;
        const job = await getNextPendingJob([...this.activeRecordingIds]);
        if (!job) {
          slot.reserved = false;
          break;
        }

        this.activeJobIds.add(job.id);
        this.activeRecordingIds.add(job.recordingId);
        dispatchedCount += 1;
        console.log("[jobs] dispatching claimed job", {
          workerId: slot.id,
          jobId: job.id,
          recordingId: job.recordingId,
          jobType: job.jobType,
          activeJobCount: this.activeJobIds.size,
          activeRecordingCount: this.activeRecordingIds.size
        });
        this.runClaimedJob(slot, job);
      }

      if (dispatchedCount > 0) {
        console.log("[jobs] drain dispatched jobs", { dispatchedCount });
      }
    } catch (error) {
      console.error("[jobs] drain failed", error);
    } finally {
      this.draining = false;
    }
  }

  private runClaimedJob(slot: WorkerSlot, job: ProcessingJob) {
    const analyzer = this.createAnalyzerForSlot(slot);
    void processJob(analyzer, job)
      .then((result) => {
        if (result.status === "completed" || result.status === "failed") {
          clearRecordingProgress(result.job.recordingId);
        }
      })
      .catch((error) => {
        console.error("[jobs] claimed job failed outside processor", {
          workerId: slot.id,
          jobId: job.id,
          recordingId: job.recordingId,
          error
        });
      })
      .finally(() => {
        slot.reserved = false;
        this.activeJobIds.delete(job.id);
        this.activeRecordingIds.delete(job.recordingId);
        this.kick();
      });
  }

  private createAnalyzerForSlot(slot: WorkerSlot): AudioAnalyzer {
    return {
      transcribe: (recording) => this.runAudioTaskOnSlot<TranscriptionOutput>(slot, "transcribe", { recording }),
      diarize: (recording) => this.runAudioTaskOnSlot<DiarizationOutput>(slot, "diarize", { recording }),
      identifySpeakers: (recording, diarizationSegments, profiles) =>
        this.runAudioTaskOnSlot<SpeakerIdentificationMatch[]>(slot, "identifySpeakers", { recording, diarizationSegments, profiles })
    };
  }

  private runAudioTaskOnSlot<T>(slot: WorkerSlot, type: string, payload: Record<string, unknown>): Promise<T> {
    if (slot.activeRequestId) {
      return Promise.reject(new Error(`Audio worker ${slot.id} is busy`));
    }
    const id = randomUUID();
    slot.activeRequestId = id;
    slot.reserved = false;
    console.log("[jobs] dispatching audio task", { requestId: id, type, workerId: slot.id });
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, {
        resolve: (value) => resolve(value as T),
        reject
      });
      slot.worker.postMessage({ id, type, ...payload });
    });
  }

  private findAvailableWorker() {
    return this.workers.find((worker) => !worker.activeRequestId && !worker.reserved) ?? null;
  }
}

export function getScheduler() {
  globalThis.__aiRecordSummaryScheduler ??= new EmbeddedJobScheduler();
  return globalThis.__aiRecordSummaryScheduler;
}

export async function startEmbeddedJobScheduler() {
  if (!getAppConfig().jobs.embeddedWorkerEnabled) return;
  await getScheduler().start();
}

export async function kickEmbeddedJobScheduler() {
  if (!getAppConfig().jobs.embeddedWorkerEnabled) return;
  getScheduler().kick();
}
