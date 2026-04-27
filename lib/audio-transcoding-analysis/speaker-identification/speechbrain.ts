import { execFile } from "node:child_process";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import { getAppConfig } from "../../config/app-config.ts";
import { audioProcessEnv } from "../runtime/runtime-env.ts";
import type { Recording, SpeakerDiarizationSegment, SpeakerIdentificationMatch, SpeakerProfileWithSamples } from "../../types/models.ts";

const execFileAsync = promisify(execFile);

function parseJsonOutput<T>(stdout: string, context: Record<string, unknown>): T {
  try {
    return JSON.parse(stdout) as T;
  } catch (error) {
    console.error("[speaker-id] invalid JSON from speechbrain", {
      ...context,
      stdoutPreview: stdout.slice(0, 500)
    });
    throw error;
  }
}

export async function identifyTargetSpeakers(
  recording: Recording,
  diarizationSegments: SpeakerDiarizationSegment[],
  profiles: SpeakerProfileWithSamples[]
): Promise<SpeakerIdentificationMatch[]> {
  const activeProfiles = profiles.filter((profile) => profile.status === "active" && profile.samples.length > 0);
  if (activeProfiles.length === 0 || diarizationSegments.length === 0) {
    console.log("[speaker-id] skipped", {
      recordingId: recording.id,
      activeProfileCount: activeProfiles.length,
      diarizationSegmentCount: diarizationSegments.length
    });
    return [];
  }

  const startedAt = Date.now();
  const config = getAppConfig();
  const pythonBin = config.audio.speechbrainPythonBin;
  const script = path.join(process.cwd(), "lib", "audio-transcoding-analysis", "scripts", "run_speechbrain.py");
  const cacheDir = path.join(process.cwd(), config.audio.modelCacheRoot, "speechbrain", "spkrec-ecapa-voxceleb");
  const payload = {
    recordingPath: path.join(process.cwd(), recording.storagePath),
    cacheDir,
    threshold: config.audio.targetSpeakerThreshold,
    minSegmentMs: 1000,
    minSpeakerMs: 3000,
    maxSpeakerMs: 30000,
    diarizationSegments,
    profiles: activeProfiles.map((profile) => ({
      id: profile.id,
      displayName: profile.displayName,
      samplePaths: profile.samples.map((sample) => path.join(process.cwd(), sample.storagePath))
    }))
  };
  console.log("[speaker-id] starting speechbrain", {
    recordingId: recording.id,
    pythonBin,
    diarizationSegmentCount: diarizationSegments.length,
    activeProfileCount: activeProfiles.length,
    threshold: config.audio.targetSpeakerThreshold
  });
  const tempDir = await mkdtemp(path.join(tmpdir(), "ai-record-speaker-id-"));
  const payloadPath = path.join(tempDir, "payload.json");
  await writeFile(payloadPath, JSON.stringify(payload), "utf8");
  let stdout: string;
  try {
    const result = await execFileAsync(pythonBin, [script, "--payload-file", payloadPath], {
      env: audioProcessEnv(pythonBin, config.audio.modelCacheRoot),
      maxBuffer: 1024 * 1024 * 50
    });
    stdout = result.stdout;
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
  const matches = parseJsonOutput<SpeakerIdentificationMatch[]>(stdout, { recordingId: recording.id });
  console.log("[speaker-id] speechbrain complete", {
    recordingId: recording.id,
    durationMs: Date.now() - startedAt,
    matchCount: matches.length,
    positiveCount: matches.filter((match) => match.isTargetPerson).length
  });
  return matches;
}
