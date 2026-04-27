export async function register() {
  if (process.env.NEXT_RUNTIME === "nodejs") {
    const { setGlobalAppConfig } = await import("./lib/config/app-config");
    const { ensureAudioDependencies } = await import("./lib/audio-transcoding-analysis/runtime/dependencies");
    const { initDatabase } = await import("@/lib/db/init");
    const { startEmbeddedJobScheduler } = await import("@/lib/audio-transcoding-analysis/jobs/scheduler");
    setGlobalAppConfig();
    await ensureAudioDependencies();
    await initDatabase();
    await startEmbeddedJobScheduler();
  }
}
