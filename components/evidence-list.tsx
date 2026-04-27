"use client";

import Link from "next/link";
import { Play } from "lucide-react";
import { formatMs } from "@/lib/types/format";
import type { SearchEvidence } from "@/lib/types/models";

export function EvidenceList({ evidence }: { evidence: SearchEvidence[] }) {
  if (evidence.length === 0) return <div className="empty">暂无召回片段</div>;
  return (
    <div className="evidence-list">
      {evidence.map((item) => {
        const speaker = item.chunk.matchedSpeakerProfiles.map((profile) => profile.displayName).join("、") || item.chunk.speakerLabels.join("、") || "Unknown Speaker";
        return (
          <article className="evidence-card" key={item.chunk.id}>
            <div className="segment-head">
              <div className="meta">
                <strong>[{item.index}] {item.recording.title}</strong>
                <span>{formatMs(item.chunk.startMs)} - {formatMs(item.chunk.endMs)}</span>
                <span>{speaker}</span>
                <span>{(item.score * 100).toFixed(0)}%</span>
              </div>
              <Link className="button secondary" href={item.url} target="_blank" rel="noreferrer">
                <Play size={16} />
                播放片段
              </Link>
            </div>
            <p>{item.chunk.text}</p>
          </article>
        );
      })}
    </div>
  );
}
