import { initDatabase } from "../../lib/db/init";
import { enqueueEmbeddingIndexing, listCompletedRecordingIds } from "../../lib/db/search";

function argValue(name: string) {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

async function main() {
  await initDatabase();
  const recordingId = argValue("--recording-id");
  const force = process.argv.includes("--force");
  const ids = recordingId ? [recordingId] : await listCompletedRecordingIds();
  for (const id of ids) {
    await enqueueEmbeddingIndexing(id, { force });
    console.log(`[search:reindex] queued ${id}`);
  }
  console.log(`[search:reindex] queued ${ids.length} recording(s)`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
