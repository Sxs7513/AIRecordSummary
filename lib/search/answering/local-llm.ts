import path from "node:path";
import { spawn } from "node:child_process";
import { audioProcessEnv } from "../../audio-transcoding-analysis/runtime/runtime-env";
import { resolveSharedLocalLlmModelPath } from "../../local-llm/model-path";
import type { RagAnswerInput, RagAnswerOutput } from "../types";
import type { RagAnswerProvider } from "./provider";

export class LocalLlmAnswerProvider implements RagAnswerProvider {
  constructor(private readonly options: { pythonBin: string; modelCacheRoot: string; modelRepo: string; modelFile: string; contextSize: number; timeoutMs: number }) {}

  async generateAnswer(input: RagAnswerInput): Promise<RagAnswerOutput> {
    const script = path.join(process.cwd(), "lib", "search", "answering", "scripts", "run_llm_answer.py");
    const { modelPath } = resolveSharedLocalLlmModelPath(this.options);

    return new Promise<RagAnswerOutput>((resolve, reject) => {
      const child = spawn(this.options.pythonBin, [script], {
        cwd: process.cwd(),
        env: audioProcessEnv(this.options.pythonBin, this.options.modelCacheRoot)
      });
      const timer = setTimeout(() => {
        child.kill("SIGTERM");
        reject(new Error(`local RAG answer timed out after ${this.options.timeoutMs}ms`));
      }, this.options.timeoutMs);

      let stdout = "";
      let stderr = "";
      let settled = false;
      child.stdout.on("data", (chunk) => {
        stdout += chunk.toString();
      });
      child.stderr.on("data", (chunk) => {
        stderr += chunk.toString();
      });
      child.on("error", (error) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        reject(error);
      });
      child.on("exit", (code) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        if (code !== 0) {
          reject(new Error(`local RAG answer exited with code ${code}\n${stderr}`));
          return;
        }
        try {
          resolve(JSON.parse(stdout) as RagAnswerOutput);
        } catch (error) {
          reject(error);
        }
      });

      child.stdin.end(JSON.stringify({
        query: input.query,
        evidence: input.evidence,
        modelPath,
        contextSize: this.options.contextSize
      }));
    });
  }
}
