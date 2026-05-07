import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { NextRequest, NextResponse } from "next/server";

const MIME: Record<string, string> = {
  ".mp3": "audio/mpeg",
  ".m4a": "audio/mp4",
  ".mp4": "audio/mp4",
  ".wav": "audio/wav"
};

function contentType(filePath: string) {
  return MIME[path.extname(filePath).toLowerCase()] || "application/octet-stream";
}

function parseRange(rangeHeader: string | null, fileSize: number) {
  if (!rangeHeader) return null;
  const match = rangeHeader.match(/^bytes=(\d*)-(\d*)$/);
  if (!match) return null;
  const [, rawStart, rawEnd] = match;
  if (!rawStart && !rawEnd) return null;

  if (!rawStart) {
    const suffixLength = Number(rawEnd);
    if (!Number.isFinite(suffixLength) || suffixLength <= 0) return null;
    const start = Math.max(0, fileSize - suffixLength);
    return { start, end: fileSize - 1 };
  }

  const start = Number(rawStart);
  const end = rawEnd ? Number(rawEnd) : fileSize - 1;
  if (!Number.isFinite(start) || !Number.isFinite(end) || start > end || start >= fileSize) return null;
  return { start, end: Math.min(end, fileSize - 1) };
}

export async function GET(request: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  const { path: pathParts } = await params;
  const storageRoot = process.env.LOCAL_STORAGE_ROOT || "uploads";
  const allowedRoot = path.resolve(process.cwd(), storageRoot);
  const absolutePath = path.resolve(process.cwd(), storageRoot, ...pathParts);
  if (absolutePath !== allowedRoot && !absolutePath.startsWith(`${allowedRoot}${path.sep}`)) {
    return new NextResponse("Not found", { status: 404 });
  }

  let fileSize = 0;
  try {
    fileSize = (await stat(absolutePath)).size;
  } catch {
    return new NextResponse("Not found", { status: 404 });
  }

  const range = parseRange(request.headers.get("range"), fileSize);
  if (request.headers.get("range") && !range) {
    return new NextResponse("Requested Range Not Satisfiable", {
      status: 416,
      headers: {
        "accept-ranges": "bytes",
        "content-range": `bytes */${fileSize}`
      }
    });
  }

  if (range) {
    const file = await readFile(absolutePath);
    const chunk = file.subarray(range.start, range.end + 1);
    return new NextResponse(chunk, {
      status: 206,
      headers: {
        "accept-ranges": "bytes",
        "content-length": String(chunk.byteLength),
        "content-range": `bytes ${range.start}-${range.end}/${fileSize}`,
        "content-type": contentType(absolutePath)
      }
    });
  }

  const file = await readFile(absolutePath);
  return new NextResponse(file, {
    headers: {
      "accept-ranges": "bytes",
      "content-length": String(fileSize),
      "content-type": contentType(absolutePath)
    }
  });
}
