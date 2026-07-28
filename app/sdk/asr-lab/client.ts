"use client";

import { pythonApiUrl, responseDetail } from "@/app/sdk/python-api";

export async function asrLabRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(pythonApiUrl(path), {
    ...init,
    credentials: "include",
    headers: init?.body instanceof FormData
      ? init.headers
      : { "Content-Type": "application/json", ...init?.headers }
  });
  if (!response.ok) throw new Error(await responseDetail(response, `请求失败（${response.status}）`));
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export function assetAudioUrl(assetId: string): string {
  return pythonApiUrl(`/api/evaluation/assets/${assetId}/audio`);
}

