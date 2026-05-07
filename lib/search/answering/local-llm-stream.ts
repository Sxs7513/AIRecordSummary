import path from "node:path";
import { spawn } from "node:child_process";
import { audioProcessEnv } from "../../audio-transcoding-analysis/runtime/runtime-env";
import { resolveSharedLocalLlmModelPath } from "../../local-llm/model-path";
import type { RagAnswerInput, RagAnswerOutput } from "../types";

export async function streamLocalLlmAnswer(
  input: RagAnswerInput,
  options: { pythonBin: string; modelCacheRoot: string; modelRepo: string; modelFile: string; contextSize: number; timeoutMs: number },
  onDelta: (text: string) => void,
  onThinking?: (event: "start" | "done", text?: string) => void
): Promise<RagAnswerOutput> {
  const script = path.join(process.cwd(), "lib", "search", "answering", "scripts", "run_llm_answer_stream.py");
  const { modelPath } = resolveSharedLocalLlmModelPath(options);

  return new Promise<RagAnswerOutput>((resolve, reject) => {
    const child = spawn(options.pythonBin, [script], {
      cwd: process.cwd(),
      env: audioProcessEnv(options.pythonBin, options.modelCacheRoot)
    });
    const timer = setTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error(`local RAG answer stream timed out after ${options.timeoutMs}ms`));
    }, options.timeoutMs);

    let stderr = "";
    let stdoutBuffer = "";
    let finalAnswer: RagAnswerOutput | null = null;
    let settled = false;

    child.stdout.on("data", (chunk) => {
      stdoutBuffer += chunk.toString();
      const lines = stdoutBuffer.split(/\r?\n/);
      stdoutBuffer = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.trim()) continue;
        const payload = JSON.parse(line) as { type?: string; text?: string; answer?: RagAnswerOutput };
        if (payload.type === "delta" && payload.text) {
          onDelta(payload.text);
        }
        if (payload.type === "thinking_start") {
          onThinking?.("start");
        }
        if (payload.type === "thinking_done") {
          onThinking?.("done", payload.text ?? "");
        }
        if (payload.type === "done" && payload.answer) {
          finalAnswer = payload.answer;
        }
      }
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
        reject(new Error(`local RAG answer stream exited with code ${code}\n${stderr}`));
        return;
      }
      if (!finalAnswer) {
        reject(new Error(`local RAG answer stream returned no final answer\n${stderr}`));
        return;
      }
      resolve(finalAnswer);
    });

    child.stdin.end(JSON.stringify({
      query: input.query,
      evidence: input.evidence,
      modelPath,
      contextSize: options.contextSize
    }));
  });
}
