import { NextRequest, NextResponse } from "next/server";
import { retryFailedJob } from "@/lib/db/recordings";
import { kickEmbeddedJobScheduler } from "@/lib/audio-transcoding-analysis/jobs/scheduler";

export async function POST(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  try {
    const job = await retryFailedJob(id);
    void kickEmbeddedJobScheduler();
    if (request.headers.get("accept")?.includes("text/html")) {
      return NextResponse.redirect(new URL(`/recordings/${job.recordingId}`, request.url), 303);
    }
    return NextResponse.json({ job });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
