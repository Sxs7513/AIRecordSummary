import { NextRequest, NextResponse } from "next/server";
import { query } from "@/lib/db/pool";

export async function GET(_request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const rows = await query("select * from processing_jobs where id = $1", [id]);
  if (rows.length === 0) {
    return NextResponse.json({ error: "Job not found" }, { status: 404 });
  }
  return NextResponse.json(rows[0]);
}
