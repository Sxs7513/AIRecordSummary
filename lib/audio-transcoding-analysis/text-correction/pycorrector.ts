import path from "node:path";
import { spawn } from "node:child_process";
import { audioProcessEnv } from "../runtime/runtime-env.ts";
import type { TranscriptionPromptConfig } from "../transcription/prompt-config.ts";

export async function correctTextsWithPycorrector(options: {
  pythonBin: string;
  modelCacheRoot: string;
  texts: string[];
  config: TranscriptionPromptConfig | null;
}): Promise<string[]> {
  if (options.texts.length === 0) return [];

  const protectedTerms = Array.from(new Set([
    ...(options.config?.protectTerms ?? []),
    ...(options.config?.people ?? [])
  ].map((term) => String(term).trim()).filter(Boolean))).sort((a, b) => b.length - a.length);

  const script = path.join(process.cwd(), "lib", "audio-transcoding-analysis", "text-correction", "scripts", "run_pycorrector.py");
  return new Promise<string[]>((resolve, reject) => {
    const child = spawn(options.pythonBin, [script], {
      cwd: process.cwd(),
      env: audioProcessEnv(options.pythonBin, options.modelCacheRoot)
    });

    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", reject);
    child.on("exit", (code) => {
      if (code !== 0) {
        reject(new Error(`pycorrector exited with code ${code}\n${stderr}`));
        return;
      }
      try {
        const payload = JSON.parse(stdout) as { texts?: string[] };
        resolve(payload.texts ?? options.texts);
      } catch (error) {
        reject(error);
      }
    });

    child.stdin.end(JSON.stringify({ texts: options.texts, protectedTerms }));
  });
}
