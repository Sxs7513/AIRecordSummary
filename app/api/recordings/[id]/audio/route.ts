import { NextRequest, NextResponse } from "next/server";

const pythonApiOrigin = process.env.PYTHON_API_ORIGIN ?? "http://localhost:8000";

/** Same-origin audio proxy: the browser never receives a raw storage path. */
export async function GET(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const headers = new Headers();
  const cookie = request.headers.get("cookie");
  const range = request.headers.get("range");
  if (cookie) headers.set("cookie", cookie);
  if (range) headers.set("range", range);
  const upstream = await fetch(`${pythonApiOrigin}/api/recordings/${encodeURIComponent(id)}/audio`, { headers, cache: "no-store" });
  const responseHeaders = new Headers();
  for (const name of ["accept-ranges", "content-length", "content-range", "content-type"]) {
    const value = upstream.headers.get(name);
    if (value) responseHeaders.set(name, value);
  }
  return new NextResponse(upstream.body, { status: upstream.status, headers: responseHeaders });
}
