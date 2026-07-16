import { NextRequest, NextResponse } from "next/server";
import { updateRecordingLocation } from "@/lib/db/recordings";

export async function POST(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  try {
    const formData = await request.formData();
    const location = formData.get("location");
    await updateRecordingLocation(id, typeof location === "string" ? location : null);
    if (request.headers.get("accept")?.includes("text/html")) {
      return NextResponse.redirect(new URL(`/recordings/${id}`, request.url), 303);
    }
    return NextResponse.json({ ok: true });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json({ error: message }, { status: 400 });
  }
}

