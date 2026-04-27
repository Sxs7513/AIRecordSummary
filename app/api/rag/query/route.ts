import { NextResponse } from "next/server";
import { ragLog, textPreview } from "@/lib/search/debug";
import { runRagQuery } from "@/lib/search/rag";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const startedAt = Date.now();
  try {
    const body = await request.json();
    ragLog("api.request", {
      mode: body.mode === "retrieve_only" ? "retrieve_only" : "answer",
      limit: typeof body.limit === "number" ? body.limit : null,
      queryLength: String(body.query ?? "").length,
      queryPreview: textPreview(String(body.query ?? "")),
      filters: body.filters ?? {}
    });
    const result = await runRagQuery({
      query: String(body.query ?? ""),
      mode: body.mode === "retrieve_only" ? "retrieve_only" : "answer",
      limit: typeof body.limit === "number" ? body.limit : undefined,
      filters: body.filters ?? {}
    });
    ragLog("api.response", {
      queryId: result.queryId,
      evidenceCount: result.evidence.length,
      hasAnswer: Boolean(result.answer),
      answerLength: result.answer?.text.length ?? 0,
      notEnoughEvidence: result.answer?.notEnoughEvidence ?? null,
      durationMs: Date.now() - startedAt
    });
    return NextResponse.json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    ragLog("api.error", { message, durationMs: Date.now() - startedAt });
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
