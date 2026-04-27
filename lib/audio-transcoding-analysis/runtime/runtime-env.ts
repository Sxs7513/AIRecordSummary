import path from "node:path";

export function audioProcessEnv(pythonBin: string, modelCacheRoot = "model-cache"): NodeJS.ProcessEnv {
  const venvBin = path.dirname(path.resolve(process.cwd(), pythonBin));
  const cacheRoot = path.resolve(process.cwd(), modelCacheRoot);
  const huggingFaceHome = path.join(cacheRoot, "huggingface");
  return {
    ...process.env,
    HF_HOME: huggingFaceHome,
    HUGGINGFACE_HUB_CACHE: path.join(huggingFaceHome, "hub"),
    TORCH_HOME: path.join(cacheRoot, "torch"),
    XDG_CACHE_HOME: path.join(cacheRoot, "xdg"),
    PATH: `${venvBin}${path.delimiter}${process.env.PATH || ""}`
  };
}
