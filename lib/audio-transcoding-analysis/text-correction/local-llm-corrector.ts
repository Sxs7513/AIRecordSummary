import path from "node:path";
import { spawn } from "node:child_process";
import { audioProcessEnv } from "../runtime/runtime-env.ts";
import type { TranscriptionPromptConfig } from "../transcription/prompt-config.ts";

export interface LocalLlmCorrectionOptions {
  pythonBin: string;
  modelCacheRoot: string;
  modelRepo: string;
  modelFile: string;
  contextSize: number;
  timeoutMs: number;
  config: TranscriptionPromptConfig | null;
}

export interface LocalLlmMergeSegment {
  id: string;
  startMs: number;
  endMs: number;
  text: string;
}

export interface LocalLlmMergeCandidate {
  groupId: string;
  speakerLabel: string | null;
  segments: LocalLlmMergeSegment[];
}

export interface LocalLlmMergeResult {
  groupId: string;
  groups: Array<{
    sourceIds: string[];
    text: string;
  }>;
}

function resolveModelPath(options: Pick<LocalLlmCorrectionOptions, "modelCacheRoot" | "modelRepo" | "modelFile">) {
  const modelDir = path.join(process.cwd(), options.modelCacheRoot, "llm-correction", options.modelRepo.replaceAll("/", "__"));
  const modelFile = options.modelFile.split(",")[0]?.trim() ?? options.modelFile;
  return path.join(modelDir, modelFile);
}

export async function correctTextsWithLocalLlm(texts: string[], options: LocalLlmCorrectionOptions): Promise<string[]> {
  if (texts.length === 0) return [];

  const script = path.join(process.cwd(), "lib", "audio-transcoding-analysis", "text-correction", "scripts", "run_llm_corrector.py");
  const modelPath = resolveModelPath(options);

  return new Promise<string[]>((resolve, reject) => {
    const child = spawn(options.pythonBin, [script], {
      cwd: process.cwd(),
      env: audioProcessEnv(options.pythonBin, options.modelCacheRoot)
    });

    const timeout = setTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error(`local LLM correction timed out after ${options.timeoutMs}ms`));
    }, options.timeoutMs);

    let settled = false;
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      reject(error);
    });
    child.on("exit", (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      if (code !== 0) {
        reject(new Error(`local LLM correction exited with code ${code}\n${stderr}`));
        return;
      }
      try {
        const payload = JSON.parse(stdout) as { texts?: string[] };
        resolve(payload.texts ?? texts);
      } catch (error) {
        reject(error);
      }
    });

    child.stdin.end(JSON.stringify({
      mode: "correctTexts",
      texts,
      modelPath,
      contextSize: options.contextSize,
      config: options.config
    }));
  });
}

export async function mergeUtteranceGroupsWithLocalLlm(candidates: LocalLlmMergeCandidate[], options: LocalLlmCorrectionOptions): Promise<LocalLlmMergeResult[]> {
  if (candidates.length === 0) return [];

  const script = path.join(process.cwd(), "lib", "audio-transcoding-analysis", "text-correction", "scripts", "run_llm_corrector.py");
  const modelPath = resolveModelPath(options);

  return new Promise<LocalLlmMergeResult[]>((resolve, reject) => {
    const child = spawn(options.pythonBin, [script], {
      cwd: process.cwd(),
      env: audioProcessEnv(options.pythonBin, options.modelCacheRoot)
    });

    const timeout = setTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error(`local LLM utterance merge timed out after ${options.timeoutMs}ms`));
    }, options.timeoutMs);

    let settled = false;
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      reject(error);
    });
    child.on("exit", (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      if (code !== 0) {
        reject(new Error(`local LLM utterance merge exited with code ${code}\n${stderr}`));
        return;
      }
      try {
        const payload = JSON.parse(stdout) as { results?: LocalLlmMergeResult[] };
        resolve(payload.results ?? []);
      } catch (error) {
        reject(error);
      }
    });

    child.stdin.end(JSON.stringify({
      mode: "mergeUtterances",
      candidates,
      modelPath,
      contextSize: options.contextSize,
      config: options.config
    }));
  });
}
