import { NextRequest, NextResponse } from "next/server";
import { deleteSpeakerProfile } from "@/lib/db/speaker-profiles";
import { deleteStoredFile } from "@/lib/storage/local-storage";

export async function POST(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  try {
    const deleted = await deleteSpeakerProfile(id);
    if (!deleted) {
      return NextResponse.json({ error: "Speaker profile not found" }, { status: 404 });
    }

    await Promise.all(deleted.samples.map((sample) => deleteStoredFile(sample.storagePath)));

    if (request.headers.get("accept")?.includes("text/html")) {
      return NextResponse.redirect(new URL("/speaker-profiles", request.url), 303);
    }
    return NextResponse.json({ deleted: true, speakerProfileId: deleted.profile.id });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error("[speaker-profile] delete failed", { speakerProfileId: id, error: message });
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
