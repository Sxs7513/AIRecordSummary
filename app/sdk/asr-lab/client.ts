"use client";

import { responseDetail } from "@/app/sdk/python-api";

const evaluationApiOrigin = process.env.NEXT_PUBLIC_EVALUATION_API_ORIGIN ?? "http://localhost:8001";
const trainingApiOrigin = process.env.NEXT_PUBLIC_TRAINING_API_ORIGIN ?? "http://localhost:8002";

function asrLabApiUrl(path: string): string {
  const origin = path.startsWith("/api/training-runs") ? trainingApiOrigin : evaluationApiOrigin;
  return `${origin}${path}`;
}

export async function asrLabRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(asrLabApiUrl(path), {
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
  return `${evaluationApiOrigin}/api/evaluation/assets/${assetId}/audio`;
}
