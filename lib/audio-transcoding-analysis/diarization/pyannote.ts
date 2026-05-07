import path from "node:path";
import { getAppConfig } from "../../config/app-config.ts";
import { type AudioProgressEvent, runPythonJson } from "../runtime/python-json.ts";
import { audioProcessEnv } from "../runtime/runtime-env.ts";
import { redactSecrets } from "../../security/redact.ts";
import type { DiarizationOutput, Recording } from "../../types/models.ts";

function normalizePyannoteError(error: unknown): Error {
  const err = error as { message?: string; stderr?: string; stdout?: string };
  const output = `${err.stderr || ""}\n${err.stdout || ""}\n${err.message || ""}`;
  if (
    output.includes("Cannot access gated repo")
    || output.includes("Access to model pyannote/speaker-diarization-3.1 is restricted")
    || output.includes("Access to model pyannote/segmentation-3.0 is restricted")
    || output.includes("Access to model pyannote/speaker-diarization-community-1 is restricted")
  ) {
    return new Error(
      "Pyannote model access denied. Accept the user conditions for https://hf.co/pyannote/speaker-diarization-3.1, https://hf.co/pyannote/segmentation-3.0, and https://hf.co/pyannote/speaker-diarization-community-1, then make sure PYANNOTE_AUTH_TOKEN belongs to that authorized HuggingFace account."
    );
  }
  if (output.includes("PYANNOTE_AUTH_TOKEN is required")) {
    return new Error("PYANNOTE_AUTH_TOKEN is required for pyannote/speaker-diarization-3.1. Set it in .env and restart the app.");
  }
  return new Error(redactSecrets((err.stderr || err.message || String(error)).trim()));
}

export async function diarizeRecording(recording: Recording, onProgress?: (progress: AudioProgressEvent) => void): Promise<DiarizationOutput> {
  const startedAt = Date.now();
  const config = getAppConfig();
  const pythonBin = config.audio.pyannotePythonBin;
  const script = path.join(process.cwd(), "lib", "audio-transcoding-analysis", "scripts", "run_pyannote.py");
  const audioPath = path.join(process.cwd(), recording.storagePath);
  const cacheDir = path.join(process.cwd(), config.audio.modelCacheRoot, "huggingface", "hub");
  const args = [script, audioPath, "--cache-dir", cacheDir];
  if (!config.audio.pyannoteUseLocalConfig) args.push("--no-local-config");
  console.log("[diarization] starting pyannote", {
    recordingId: recording.id,
    pythonBin,
    audioPath,
    cacheDir,
    useLocalConfig: config.audio.pyannoteUseLocalConfig,
    hasAuthToken: Boolean(config.audio.pyannoteAuthToken)
  });
  let output: DiarizationOutput;
  try {
    output = await runPythonJson<DiarizationOutput>({
      pythonBin,
      args,
      env: {
        ...audioProcessEnv(pythonBin, config.audio.modelCacheRoot),
        PYANNOTE_AUTH_TOKEN: config.audio.pyannoteAuthToken
      },
      logPrefix: "[diarization]",
      onProgress
    });
  } catch (error) {
    throw normalizePyannoteError(error);
  }
  console.log("[diarization] pyannote complete", {
    recordingId: recording.id,
    durationMs: Date.now() - startedAt,
    segmentCount: output.segments.length
  });
  return output;
}
