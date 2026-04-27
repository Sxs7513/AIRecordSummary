import { NextRequest, NextResponse } from "next/server";
import { deleteRecording } from "@/lib/db/recordings";
import { clearRecordingProgress } from "@/lib/audio-transcoding-analysis/jobs/progress";
import { deleteStoredFile } from "@/lib/storage/local-storage";

export async function POST(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  try {
    const recording = await deleteRecording(id);
    if (!recording) {
      return NextResponse.json({ error: "Recording not found" }, { status: 404 });
    }

    clearRecordingProgress(recording.id);
    await deleteStoredFile(recording.storagePath);

    if (request.headers.get("accept")?.includes("text/html")) {
      return NextResponse.redirect(new URL("/recordings", request.url), 303);
    }
    return NextResponse.json({ deleted: true, recordingId: recording.id });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error("[recordings] delete failed", { recordingId: id, error: message });
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
