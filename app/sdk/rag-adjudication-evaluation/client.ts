"use client";

import { asrLabRequest } from "@/app/sdk/asr-lab/client";

export function ragAdjudicationEvaluationRequest<T>(path: string, init?: RequestInit): Promise<T> {
  return asrLabRequest<T>(`/api/evaluation/rag-adjudication${path}`, init);
}

