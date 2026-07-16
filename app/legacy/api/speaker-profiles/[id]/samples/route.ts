import { NextRequest, NextResponse } from "next/server";
import { createSpeakerProfileSample } from "@/lib/db/speaker-profiles";
import { saveAudioFile } from "@/lib/storage/local-storage";

export async function POST(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  console.log("[speaker-sample-upload] request received", { speakerProfileId: id });
  const formData = await request.formData();
  const file = formData.get("audio");
  if (!(file instanceof File) || file.size === 0) {
    console.log("[speaker-sample-upload] rejected empty file", { speakerProfileId: id });
    return NextResponse.json({ error: "请选择参考音频样本" }, { status: 400 });
  }

  console.log("[speaker-sample-upload] saving file", {
    speakerProfileId: id,
    name: file.name,
    type: file.type,
    size: file.size
  });
  const stored = await saveAudioFile(file, "speaker-samples");
  const sample = await createSpeakerProfileSample(id, stored);
  console.log("[speaker-sample-upload] sample created", {
    speakerProfileId: id,
    sampleId: sample.id,
    storagePath: sample.storagePath
  });
  if (request.headers.get("accept")?.includes("text/html")) {
    return NextResponse.redirect(new URL("/speaker-profiles", request.url), 303);
  }
  return NextResponse.json(sample, { status: 201 });
}
