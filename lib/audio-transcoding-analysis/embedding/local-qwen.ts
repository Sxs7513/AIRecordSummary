import path from "node:path";
import { audioProcessEnv } from "../runtime/runtime-env";
import { runPythonJson } from "../runtime/python-json";
import type { EmbeddingProvider } from "./provider";

interface LocalEmbeddingOutput {
  embeddings: number[][];
}

export class LocalQwenEmbeddingProvider implements EmbeddingProvider {
  constructor(
    private readonly options: {
      pythonBin: string;
      modelName: string;
      modelCacheDir: string;
      device: string;
      batchSize: number;
    }
  ) {}

  async embedTexts(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    return this.run(texts, "document");
  }

  async embedQuery(query: string): Promise<number[]> {
    const [embedding] = await this.run([query], "query");
    if (!embedding) throw new Error("Embedding provider returned no query embedding");
    return embedding;
  }

  private async run(texts: string[], mode: "query" | "document") {
    const script = path.join(process.cwd(), "lib", "audio-transcoding-analysis", "embedding", "scripts", "run_qwen_embedding.py");
    const output = await runPythonJson<LocalEmbeddingOutput>({
      pythonBin: this.options.pythonBin,
      args: [
        script,
        "--model",
        this.options.modelName,
        "--cache-dir",
        this.options.modelCacheDir,
        "--device",
        this.options.device,
        "--batch-size",
        String(this.options.batchSize),
        "--mode",
        mode,
        "--texts-json",
        JSON.stringify(texts)
      ],
      env: audioProcessEnv(this.options.pythonBin, this.options.modelCacheDir),
      logPrefix: "[embedding]"
    });
    return output.embeddings;
  }
}
