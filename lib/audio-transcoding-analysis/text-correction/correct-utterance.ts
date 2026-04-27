import { getAppConfig } from "../../config/app-config.ts";
import { correctTextsWithLocalLlm, mergeUtteranceGroupsWithLocalLlm, type LocalLlmMergeCandidate, type LocalLlmMergeResult } from "./local-llm-corrector.ts";
import { correctTextsWithPycorrector } from "./pycorrector.ts";
import { applyTranscriptionTextCorrections, loadTranscriptionPromptConfig } from "../transcription/prompt-config.ts";

export async function correctUtteranceTexts(texts: string[]): Promise<string[]> {
  const config = getAppConfig();
  const promptConfig = await loadTranscriptionPromptConfig(config.audio.whisperInitialPromptConfigPath);
  let correctedTexts = texts;

  if (config.audio.transcriptionCorrectionEnabled) {
    try {
      correctedTexts = await correctTextsWithPycorrector({
        pythonBin: config.audio.whisperPythonBin,
        modelCacheRoot: config.audio.modelCacheRoot,
        texts,
        config: promptConfig
      }) ?? texts;
    } catch (error) {
      console.error("[utterance] pycorrector failed, falling back to rule corrections", {
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  const ruleCorrectedTexts = correctedTexts.map((text) => applyTranscriptionTextCorrections(text, promptConfig));

  if (config.audio.llmCorrectionEnabled) {
    try {
      const llmCorrectedTexts = await correctTextsWithLocalLlm(ruleCorrectedTexts, {
        pythonBin: config.audio.llmCorrectionPythonBin,
        modelCacheRoot: config.audio.modelCacheRoot,
        modelRepo: config.audio.llmCorrectionModelRepo,
        modelFile: config.audio.llmCorrectionModelFile,
        contextSize: config.audio.llmCorrectionContextSize,
        timeoutMs: config.audio.llmCorrectionTimeoutMs,
        config: promptConfig
      });
      return llmCorrectedTexts.map((text) => applyTranscriptionTextCorrections(text, promptConfig));
    } catch (error) {
      console.error("[utterance] local LLM correction failed, falling back to rule corrections", {
        modelRepo: config.audio.llmCorrectionModelRepo,
        modelFile: config.audio.llmCorrectionModelFile,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  return ruleCorrectedTexts;
}

export async function mergeCorrectedUtterances(candidates: LocalLlmMergeCandidate[]): Promise<LocalLlmMergeResult[]> {
  if (candidates.length === 0) return [];
  const config = getAppConfig();
  if (!config.audio.llmCorrectionEnabled) return [];
  const promptConfig = await loadTranscriptionPromptConfig(config.audio.whisperInitialPromptConfigPath);

  try {
    const results = await mergeUtteranceGroupsWithLocalLlm(candidates, {
      pythonBin: config.audio.llmCorrectionPythonBin,
      modelCacheRoot: config.audio.modelCacheRoot,
      modelRepo: config.audio.llmCorrectionModelRepo,
      modelFile: config.audio.llmCorrectionModelFile,
      contextSize: config.audio.llmCorrectionContextSize,
      timeoutMs: config.audio.llmCorrectionTimeoutMs,
      config: promptConfig
    });
    return results.map((result) => ({
      ...result,
      groups: result.groups.map((group) => ({
        ...group,
        text: applyTranscriptionTextCorrections(group.text, promptConfig)
      }))
    }));
  } catch (error) {
    console.error("[utterance] local LLM merge failed, keeping corrected utterances", {
      modelRepo: config.audio.llmCorrectionModelRepo,
      modelFile: config.audio.llmCorrectionModelFile,
      error: error instanceof Error ? error.message : String(error)
    });
    return [];
  }
}
