import { NextResponse } from "next/server";

/** Storage keys are internal implementation details; use the authorized recording audio route instead. */
export function GET() { return new NextResponse("Not found", { status: 404 }); }
