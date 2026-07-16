"use client";

import { RotateCcw } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { pythonApiUrl, responseDetail } from "@/app/sdk/python-api";

const ACTIVE_STATUSES = new Set(["pending", "running"]);

export function RetryEmbeddingIndexButton({ recordingId, stageStatus }: { recordingId: string; stageStatus: string }) {
  const router = useRouter();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const disabled = submitting || ACTIVE_STATUSES.has(stageStatus);

  async function retry() {
    setSubmitting(true);
    try {
      const response = await fetch(
        pythonApiUrl(`/api/recordings/${encodeURIComponent(recordingId)}/stages/embedding_indexing/retry`),
        { method: "POST", credentials: "include" }
      );
      if (!response.ok) {
        setError(await responseDetail(response, "向量索引重试失败"));
        return;
      }
      setError(null);
      router.refresh();
    } catch {
      setError("向量索引重试失败：无法连接服务端");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <span>
      <button className="secondary" type="button" onClick={retry} disabled={disabled}>
        <RotateCcw size={14} />
        {submitting ? "提交中" : "重试"}
      </button>
      {error ? <span className="subtle">{error}</span> : null}
    </span>
  );
}
