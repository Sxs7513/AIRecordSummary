"use client";

import { useEffect } from "react";
import { MarkdownSummary } from "@/components/markdown-summary";
import { GenerationStreamClient } from "@/app/sdk/generation/client";
import { selectGenerationText } from "@/app/sdk/generation/selectors";
import { useGenerationStore } from "@/app/sdk/generation/store";

export function StreamingSummary({ generationRunId, persistedMarkdown }: { generationRunId: string | null; persistedMarkdown: string | null }) {
  const generation = useGenerationStore((state) => generationRunId ? state.runs[generationRunId] : undefined);

  useEffect(() => {
    if (!generationRunId) return;
    const client = new GenerationStreamClient();
    client.connect(generationRunId);
    return () => client.close(generationRunId);
  }, [generationRunId]);

  const streamedMarkdown = selectGenerationText(generation);
  const markdown = streamedMarkdown || persistedMarkdown;
  if (markdown) return <MarkdownSummary markdown={markdown} streaming={generation?.status === "running"} />;
  if (generation?.error) return <div className="empty">总结生成失败：{generation.error.message}</div>;
  if (generation?.phase) return <div className="empty">{generation.phase.label}</div>;
  return <div className="empty">总结尚未生成</div>;
}
