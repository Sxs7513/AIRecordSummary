import path from "node:path";
import { spawn } from "node:child_process";
import { getAppConfig } from "../../config/app-config";
import { audioProcessEnv } from "../../audio-transcoding-analysis/runtime/runtime-env";
import { resolveSharedLocalLlmModelPath } from "../../local-llm/model-path";
import { ragLog, textPreview } from "../debug";
import { buildRouterPrompt, firstJsonObject } from "./prompt";
import { fallbackRoute, parseRagRoute, type RagRoute } from "./route-schema";
import { routeQueryWithRules } from "./rule-fallback-router";
import { normalizeRouteDateRange } from "./date-range";

export async function routeQueryWithLocalLlm(query: string): Promise<RagRoute> {
  const appConfig = getAppConfig();
  const config = appConfig.search;
  if (!config.answerEnabled || config.answerProvider !== "local_llm") {
    return routeQueryWithRules(query);
  }

  const script = path.join(process.cwd(), "lib", "search", "router", "scripts", "run_llm_router.py");
  const { modelPath } = resolveSharedLocalLlmModelPath({
    modelCacheRoot: appConfig.audio.modelCacheRoot,
    modelRepo: config.localLlmModelRepo,
    modelFile: config.localLlmModelFile
  });

  try {
    const route = await new Promise<RagRoute>((resolve, reject) => {
      const child = spawn(config.embeddingPythonBin, [script], {
        cwd: process.cwd(),
        env: audioProcessEnv(config.embeddingPythonBin, appConfig.audio.modelCacheRoot)
      });
      const timer = setTimeout(() => {
        child.kill("SIGTERM");
        reject(new Error(`local router timed out after ${config.answerTimeoutMs}ms`));
      }, Math.min(config.answerTimeoutMs, 120000));
      let stdout = "";
      let stderr = "";
      child.stdout.on("data", (chunk) => {
        stdout += chunk.toString();
      });
      child.stderr.on("data", (chunk) => {
        stderr += chunk.toString();
      });
      child.on("error", (error) => {
        clearTimeout(timer);
        reject(error);
      });
      child.on("exit", (code) => {
        clearTimeout(timer);
        if (code !== 0) {
          reject(new Error(`local router exited with code ${code}\n${stderr}`));
          return;
        }
        try {
          const payload = JSON.parse(stdout) as { text?: string };
          const jsonText = firstJsonObject(payload.text ?? "");
          if (!jsonText) throw new Error("router did not return JSON");
          resolve(normalizeRouteDateRange(parseRagRoute(JSON.parse(jsonText), query), query));
        } catch (error) {
          reject(error);
        }
      });
      child.stdin.end(JSON.stringify({
        prompt: buildRouterPrompt(query),
        modelPath,
        contextSize: Math.min(config.answerContextSize, 4096),
        maxTokens: 600,
        temperature: 0,
        stop: ["</s>", "<|im_end|>"]
      }));
    });
    ragLog("router.llm_done", {
      queryPreview: textPreview(query),
      strategy: route.strategy,
      intent: route.intent,
      recordingLimit: route.recordingLimit,
      topic: route.topic,
      dateRange: route.dateRange,
      personNames: route.filters.personNames,
      locations: route.filters.locations,
      speakerProfileIds: route.filters.speakerProfileIds,
      reason: route.reason
    });
    return route;
  } catch (error) {
    ragLog("router.llm_error", {
      queryPreview: textPreview(query),
      message: error instanceof Error ? error.message : String(error)
    });
    try {
      return normalizeRouteDateRange(await routeQueryWithRules(query), query);
    } catch {
      return normalizeRouteDateRange(fallbackRoute(query), query);
    }
  }
}
