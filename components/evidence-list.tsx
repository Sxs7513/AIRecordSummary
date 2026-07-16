"use client";

import Link from "next/link";
import { ChevronDown, Play } from "lucide-react";
import { useState } from "react";
import { formatMs } from "@/app/shared/format";
import type { SearchEvidence } from "@/app/shared/models";

export function EvidenceList({
  evidence,
  activeIndex,
  onSelect
}: {
  evidence: SearchEvidence[];
  activeIndex: number | null;
  onSelect: (index: number) => void;
}) {
  const [openIndexes, setOpenIndexes] = useState<Set<number>>(new Set());

  if (evidence.length === 0) return <div className="empty">暂无召回片段</div>;
  return (
    <div className="evidence-list">
      {evidence.map((item) => {
        const speaker = item.chunk.matchedSpeakerProfiles.map((profile) => profile.displayName).join("、") || item.chunk.speakerLabels.join("、") || "Unknown Speaker";
        const isOpen = openIndexes.has(item.index);
        const isActive = activeIndex === item.index;
        function toggle() {
          onSelect(item.index);
          setOpenIndexes((current) => {
            const next = new Set(current);
            if (next.has(item.index)) {
              next.delete(item.index);
            } else {
              next.add(item.index);
            }
            return next;
          });
        }
        return (
          <article className={`evidence-card ${isActive ? "active" : ""}`} id={`evidence-${item.index}`} key={item.chunk.id}>
            <button className="evidence-card-toggle" type="button" onClick={toggle} aria-expanded={isOpen}>
              <div className="meta">
                <strong className="evidence-title">[{item.index}] {item.recording.title}</strong>
                {item.recording.location ? <span>{item.recording.location}</span> : null}
                <span>{formatMs(item.chunk.startMs)} - {formatMs(item.chunk.endMs)}</span>
                <span>{speaker}</span>
                <span>{(item.score * 100).toFixed(0)}%</span>
              </div>
              <ChevronDown className={`evidence-chevron ${isOpen ? "open" : ""}`} size={18} />
            </button>
            {isOpen ? (
              <div className="evidence-card-body">
                {item.chunk.text ? <p>{item.chunk.text}</p> : null}
                <div className="evidence-card-actions">
                  <Link className="button secondary" href={item.url} target="_blank" rel="noreferrer">
                    <Play size={16} />
                    播放片段
                  </Link>
                  <Link className="button secondary" href={item.url} target="_blank" rel="noreferrer">
                    打开录音
                  </Link>
                </div>
              </div>
            ) : null}
          </article>
        );
      })}
    </div>
  );
}
