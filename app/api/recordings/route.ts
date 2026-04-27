import { NextRequest, NextResponse } from "next/server";
import { createRecording, listRecordings } from "@/lib/db/recordings";
import { kickEmbeddedJobScheduler } from "@/lib/audio-transcoding-analysis/jobs/scheduler";
import { saveAudioFile } from "@/lib/storage/local-storage";

export async function GET(request: NextRequest) {
  const status = request.nextUrl.searchParams.get("status");
  const page = Number(request.nextUrl.searchParams.get("page") || 1);
  const pageSize = Number(request.nextUrl.searchParams.get("pageSize") || 10);
  const result = await listRecordings({ status, page, pageSize });
  return NextResponse.json(result);
}

export async function POST(request: NextRequest) {
  console.log("[upload] request received");
  const formData = await request.formData();
  const files = formData.getAll("audio").filter((file): file is File => file instanceof File && file.size > 0);
  if (files.length === 0) {
    console.log("[upload] rejected empty file");
    return NextResponse.json({ error: "请选择录音文件" }, { status: 400 });
  }

  const titleInput = String(formData.get("title") || "").trim();
  const createdItems = [];
  for (const file of files) {
    console.log("[upload] saving file", {
      name: file.name,
      type: file.type,
      size: file.size
    });
    const stored = await saveAudioFile(file, "recordings");
    const title = files.length === 1 && titleInput ? titleInput : stored.fileName;
    const created = await createRecording({ ...stored, title });
    console.log("[upload] recording created", {
      recordingId: created.recording.id,
      jobId: created.job.id,
      storagePath: created.recording.storagePath
    });
    createdItems.push(created);
  }

  void kickEmbeddedJobScheduler();
  console.log("[upload] scheduler kicked", {
    recordingCount: createdItems.length,
    jobIds: createdItems.map((item) => item.job.id)
  });

  if (request.headers.get("accept")?.includes("text/html")) {
    const location = createdItems.length === 1 ? `/recordings/${createdItems[0].recording.id}` : "/recordings";
    return NextResponse.redirect(new URL(location, request.url), 303);
  }

  return NextResponse.json(
    {
      recordings: createdItems.map((item) => item.recording),
      jobIds: createdItems.map((item) => item.job.id)
    },
    { status: 201 }
  );
}
