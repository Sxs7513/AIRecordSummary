import { spawn } from "node:child_process";
import path from "node:path";
import { getAppConfig } from "../../config/app-config";

declare global {
  // eslint-disable-next-line no-var
  var __aiRecordSummaryAudioDepsReady: Promise<void> | undefined;
}

function isBuildPhase() {
  return process.env.NEXT_PHASE === "phase-production-build";
}

async function runInstallScript() {
  const config = getAppConfig();
  if (!config.audio.autoInstallDependencies) {
    console.log("[audio-deps] auto install disabled");
    return;
  }
  if (isBuildPhase()) {
    console.log("[audio-deps] skipped during build phase");
    return;
  }

  const scriptPath = path.join(process.cwd(), "scripts", "install_audio_dependencies.sh");
  const startedAt = Date.now();
  console.log("[audio-deps] ensuring audio dependencies", { scriptPath });
  await new Promise<void>((resolve, reject) => {
    const child = spawn("zsh", ["-lc", `source ~/.bash_profile >/dev/null 2>&1 || true; bash "${scriptPath}"`], {
      cwd: process.cwd(),
      env: process.env,
      stdio: "inherit"
    });
    child.on("error", reject);
    child.on("exit", (code) => {
      if (code === 0) {
        console.log("[audio-deps] ready", { durationMs: Date.now() - startedAt });
        resolve();
      } else {
        console.error("[audio-deps] installer failed", { code, durationMs: Date.now() - startedAt });
        reject(new Error(`Audio dependency installer exited with code ${code}`));
      }
    });
  });
}

export function ensureAudioDependencies(): Promise<void> {
  globalThis.__aiRecordSummaryAudioDepsReady ??= runInstallScript();
  return globalThis.__aiRecordSummaryAudioDepsReady;
}
