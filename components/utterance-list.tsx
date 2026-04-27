"use client";

import { useEffect } from "react";
import type { SpeakerProfile, UtteranceSegment } from "@/lib/types/models";
import { formatMs } from "@/lib/types/format";

export function UtteranceList({ segments, speakerProfiles, highlightMs }: { segments: UtteranceSegment[]; speakerProfiles: SpeakerProfile[]; highlightMs: number | null }) {
  useEffect(() => {
    document.querySelector(".segment.highlight")?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [highlightMs]);
  const profileById = new Map(speakerProfiles.map((profile) => [profile.id, profile]));
  if (segments.length === 0) return <div className="empty">连续发言尚未生成</div>;
  return (
    <div className="segments">
      {segments.map((segment) => {
        const profile = segment.matchedSpeakerProfileId ? profileById.get(segment.matchedSpeakerProfileId) : null;
        const highlighted = highlightMs !== null && highlightMs >= segment.startMs && highlightMs <= segment.endMs;
        return (
          <article className={`segment ${segment.isTargetPerson ? "target" : ""} ${highlighted ? "highlight" : ""}`} id={`utterance-${segment.id}`} key={segment.id}>
            <div className="segment-head">
              <div className="meta">
                <span>
                  {formatMs(segment.startMs)} - {formatMs(segment.endMs)}
                </span>
                <strong>{segment.speakerLabel || "Unknown Speaker"}</strong>
              </div>
              {segment.isTargetPerson ? (
                <span className="badge matched">
                  目标人物 {profile?.displayName || ""} {segment.targetPersonConfidence ? `${(segment.targetPersonConfidence * 100).toFixed(0)}%` : ""}
                </span>
              ) : null}
            </div>
            <div>{segment.text}</div>
          </article>
        );
      })}
    </div>
  );
}
