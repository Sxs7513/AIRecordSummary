"use client";

import { useEffect } from "react";
import type { SpeakerProfile, UtteranceSegment } from "@/lib/types/models";
import { formatMs } from "@/lib/types/format";

function playUtteranceSegment(segment: UtteranceSegment) {
  window.dispatchEvent(new CustomEvent("recording-play-segment", {
    detail: {
      startMs: segment.startMs,
      endMs: segment.endMs
    }
  }));
}

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
        const playSegment = () => playUtteranceSegment(segment);
        return (
          <article
            className={`segment utterance-segment ${segment.isTargetPerson ? "target" : ""} ${highlighted ? "highlight" : ""}`}
            id={`utterance-${segment.id}`}
            key={segment.id}
            onClick={playSegment}
            onKeyDown={(event) => {
              if (event.key !== "Enter" && event.key !== " ") return;
              event.preventDefault();
              playSegment();
            }}
            role="button"
            tabIndex={0}
            title="点击播放该段录音"
          >
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
