import { NextRequest, NextResponse } from "next/server";
import { getRecordingDetail } from "@/lib/db/recordings";

export async function GET(_request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const detail = await getRecordingDetail(id);
  if (!detail) {
    return NextResponse.json({ error: "Recording not found" }, { status: 404 });
  }
  return NextResponse.json(detail);
}
