import { NextResponse, type NextRequest } from "next/server";

const protectedPrefixes = ["/recordings", "/chat", "/speaker-profiles", "/account"];
const sessionCookieName = "ai_record_summary_session";

/**
 * Keep unauthenticated users off server-rendered application routes.
 * The Python API remains the source of truth and validates the opaque cookie.
 */
export function middleware(request: NextRequest) {
  if (!protectedPrefixes.some((prefix) => request.nextUrl.pathname.startsWith(prefix))) return NextResponse.next();
  if (request.cookies.has(sessionCookieName)) return NextResponse.next();
  const loginUrl = new URL("/login", request.url);
  loginUrl.searchParams.set("next", request.nextUrl.pathname);
  return NextResponse.redirect(loginUrl);
}

export const config = { matcher: ["/recordings/:path*", "/chat/:path*", "/speaker-profiles/:path*", "/account/:path*"] };
