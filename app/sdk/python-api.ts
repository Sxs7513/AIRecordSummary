/** Browser-facing origin for the Python HTTP service. */
export const pythonApiOrigin = process.env.NEXT_PUBLIC_PYTHON_API_ORIGIN ?? "http://localhost:8000";

export function pythonApiUrl(path: string): string {
  return `${pythonApiOrigin}${path}`;
}

export async function responseDetail(response: Response, fallback: string): Promise<string> {
  try {
    const payload: unknown = await response.json();
    if (typeof payload === "object" && payload !== null && "detail" in payload && typeof payload.detail === "string") {
      return payload.detail;
    }
  } catch {
    // A malformed error response must not hide the action's useful fallback.
  }
  return fallback;
}
