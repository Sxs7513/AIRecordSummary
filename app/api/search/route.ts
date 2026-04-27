import { NextResponse } from "next/server";
import { retrieveSearchEvidence } from "@/lib/search/search";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const result = await retrieveSearchEvidence({
      query: String(body.query ?? ""),
      limit: typeof body.limit === "number" ? body.limit : undefined,
      filters: body.filters ?? {}
    });
    return NextResponse.json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
