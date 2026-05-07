import fs from "node:fs";
import path from "node:path";

const LOCAL_LLM_CACHE_DIR = "local-llm";
const LEGACY_RAG_ANSWER_CACHE_DIR = "rag-answer";

export function resolveSharedLocalLlmModelPath(options: { modelCacheRoot: string; modelRepo: string; modelFile: string }) {
  const candidates = options.modelFile.split(",").map((item) => item.trim()).filter(Boolean);
  const repoDirName = options.modelRepo.replaceAll("/", "__");
  const modelDirs = [
    path.join(process.cwd(), options.modelCacheRoot, LOCAL_LLM_CACHE_DIR, repoDirName),
    path.join(process.cwd(), options.modelCacheRoot, LEGACY_RAG_ANSWER_CACHE_DIR, repoDirName)
  ];

  for (const modelDir of modelDirs) {
    for (const candidate of candidates) {
      const modelPath = path.join(modelDir, candidate);
      if (fs.existsSync(modelPath)) return { modelDir, modelFile: candidate, modelPath };
    }
  }

  const available = modelDirs.flatMap((modelDir) => {
    if (!fs.existsSync(modelDir)) return [];
    return fs.readdirSync(modelDir)
      .filter((item) => item.toLowerCase().endsWith(".gguf"))
      .map((item) => path.join(modelDir, item));
  });
  throw new Error(
    `local LLM model file not found. expected one of: ${candidates.join(", ") || options.modelFile}; modelDirs: ${modelDirs.join(", ")}; available gguf: ${available.join(", ") || "none"}`
  );
}
