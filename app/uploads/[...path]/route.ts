import { readFile } from "node:fs/promises";
import path from "node:path";
import { NextRequest, NextResponse } from "next/server";

const MIME: Record<string, string> = {
  ".mp3": "audio/mpeg",
  ".m4a": "audio/mp4",
  ".mp4": "audio/mp4",
  ".wav": "audio/wav"
};

export async function GET(_request: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  const { path: pathParts } = await params;
  const storageRoot = process.env.LOCAL_STORAGE_ROOT || "uploads";
  const safeRelativePath = path.normalize(path.join(storageRoot, ...pathParts));
  const absolutePath = path.join(process.cwd(), safeRelativePath);
  const allowedRoot = path.join(process.cwd(), storageRoot);
  if (!absolutePath.startsWith(allowedRoot)) {
    return new NextResponse("Not found", { status: 404 });
  }
  const file = await readFile(absolutePath);
  return new NextResponse(file, {
    headers: {
      "content-type": MIME[path.extname(absolutePath).toLowerCase()] || "application/octet-stream"
    }
  });
}
