import { NextRequest, NextResponse } from "next/server";
import { getRecordingProgress } from "@/lib/audio-transcoding-analysis/jobs/progress";

export async function GET(_request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const progress = getRecordingProgress(id);
  return NextResponse.json({
    progress: progress
      ? {
          percent: progress.percent,
          updatedAt: progress.updatedAt
        }
      : null
  });
}
