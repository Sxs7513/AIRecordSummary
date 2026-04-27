import { getAppConfig } from "../../config/app-config";
import { DeepSeekAnswerProvider } from "./deepseek-api";
import { ExtractiveAnswerProvider } from "./extractive";
import { LocalLlmAnswerProvider } from "./local-llm";
import type { RagAnswerProvider } from "./provider";

export function getAnswerProvider(): RagAnswerProvider {
  const appConfig = getAppConfig();
  const config = appConfig.search;
  if (config.answerProvider === "deepseek_api") {
    return new DeepSeekAnswerProvider({
      apiKey: config.deepseekApiKey,
      baseUrl: config.deepseekBaseUrl,
      model: config.deepseekModel,
      timeoutMs: config.answerTimeoutMs
    });
  }
  if (config.answerProvider === "extractive") return new ExtractiveAnswerProvider();
  return new LocalLlmAnswerProvider({
    pythonBin: config.embeddingPythonBin,
    modelCacheRoot: appConfig.audio.modelCacheRoot,
    modelRepo: config.answerModelRepo,
    modelFile: config.answerModelFile,
    contextSize: config.answerContextSize,
    timeoutMs: config.answerTimeoutMs
  });
}
