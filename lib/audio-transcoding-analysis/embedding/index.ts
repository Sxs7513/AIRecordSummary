import { getAppConfig } from "../../config/app-config";
import { LocalQwenEmbeddingProvider } from "./local-qwen";
import type { EmbeddingProvider } from "./provider";

export function getEmbeddingProvider(): EmbeddingProvider {
  const config = getAppConfig().search;
  return new LocalQwenEmbeddingProvider({
    pythonBin: config.embeddingPythonBin,
    modelName: config.embeddingModel,
    modelCacheDir: config.embeddingModelCacheDir,
    device: config.embeddingDevice,
    batchSize: config.embeddingBatchSize
  });
}
