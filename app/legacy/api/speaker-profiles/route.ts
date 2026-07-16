import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { createSpeakerProfile, listSpeakerProfiles } from "@/lib/db/speaker-profiles";

const schema = z.object({
  displayName: z.string().trim().min(1),
  status: z.enum(["active", "inactive"]).default("active"),
  notes: z.string().optional()
});

export async function GET() {
  return NextResponse.json(await listSpeakerProfiles());
}

export async function POST(request: NextRequest) {
  const formData = await request.formData();
  const parsed = schema.safeParse({
    displayName: formData.get("displayName"),
    status: formData.get("status") || "active",
    notes: formData.get("notes") || ""
  });
  if (!parsed.success) {
    return NextResponse.json({ error: "目标人物名称不能为空" }, { status: 400 });
  }

  const profile = await createSpeakerProfile(parsed.data);
  if (request.headers.get("accept")?.includes("text/html")) {
    return NextResponse.redirect(new URL("/speaker-profiles", request.url), 303);
  }
  return NextResponse.json(profile, { status: 201 });
}
