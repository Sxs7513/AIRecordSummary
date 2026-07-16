import { NextRequest, NextResponse } from "next/server";
import { kickEmbeddedJobScheduler } from "@/lib/audio-transcoding-analysis/jobs/scheduler";
import { updateRecordingSpeakerLabels } from "@/lib/db/recordings";

export async function POST(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  try {
    const formData = await request.formData();
    const mappings: Array<{ from: string; to: string }> = [];
    for (const [key, value] of formData.entries()) {
      if (!key.startsWith("speaker:") || typeof value !== "string") continue;
      mappings.push({
        from: key.slice("speaker:".length),
        to: value
      });
    }
    await updateRecordingSpeakerLabels(id, mappings);
    void kickEmbeddedJobScheduler();
    if (request.headers.get("accept")?.includes("text/html")) {
      return NextResponse.redirect(new URL(`/recordings/${id}`, request.url), 303);
    }
    return NextResponse.json({ ok: true });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json({ error: message }, { status: 400 });
  }
}

