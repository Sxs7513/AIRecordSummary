"use client";

import { RotateCcw } from "lucide-react";
import { useState } from "react";
import { StreamingSummary } from "@/components/streaming-summary";
import { pythonApiUrl, responseDetail } from "@/app/sdk/python-api";
import { useGenerationStore } from "@/app/sdk/generation/store";

export function RecordingSummaryPanel({
  recordingId,
  initialGenerationRunId,
  persistedMarkdown
}: {
  recordingId: string;
  initialGenerationRunId: string | null;
  persistedMarkdown: string | null;
}) {
  const [generationRunId, setGenerationRunId] = useState(initialGenerationRunId);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const generationStatus = useGenerationStore((state) => generationRunId ? state.runs[generationRunId]?.status : null);
  const isGenerating = starting || generationStatus === "queued" || generationStatus === "running";

  async function regenerate() {
    setStarting(true);
    try {
      const response = await fetch(pythonApiUrl(`/api/recordings/${encodeURIComponent(recordingId)}/summary/regenerate`), {
        method: "POST",
        credentials: "include"
      });
      if (!response.ok) {
        setError(await responseDetail(response, "重新生成总结失败"));
        return;
      }
      const payload: unknown = await response.json();
      if (typeof payload !== "object" || payload === null || !("generation_run_id" in payload) || typeof payload.generation_run_id !== "string") {
        setError("重新生成总结失败：服务端返回无效响应");
        return;
      }
      setError(null);
      setGenerationRunId(payload.generation_run_id);
    } catch {
      setError("重新生成总结失败：无法连接服务端");
    } finally {
      setStarting(false);
    }
  }

  return (
    <>
      <div className="toolbar" style={{ marginBottom: 12 }}>
        <button className="secondary" type="button" onClick={regenerate} disabled={isGenerating}>
          <RotateCcw size={16} />
          {isGenerating ? "正在生成" : "重新生成"}
        </button>
        {error ? <span className="subtle">{error}</span> : null}
      </div>
      <StreamingSummary generationRunId={generationRunId} persistedMarkdown={persistedMarkdown} />
    </>
  );
}
