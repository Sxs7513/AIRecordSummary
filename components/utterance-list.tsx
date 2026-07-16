"use client";

import { Fragment, useEffect, useMemo, useState } from "react";
import type { SpeakerProfile, TranscriptionToken, UtteranceSegment } from "@/app/shared/models";
import { formatMs } from "@/app/shared/format";

interface PlaybackPositionEventDetail {
  currentMs: number | null;
}

function timedTokenAt(tokens: TranscriptionToken[], currentMs: number): TranscriptionToken | null {
  let left = 0;
  let right = tokens.length - 1;
  while (left <= right) {
    const middle = Math.floor((left + right) / 2);
    const token = tokens[middle];
    if (currentMs < token.startMs) {
      right = middle - 1;
    } else if (currentMs > token.endMs) {
      left = middle + 1;
    } else {
      return token;
    }
  }
  return null;
}

function tokenContent(text: string, active: boolean) {
  return Array.from(text).map((character, index) => (
    /[\p{P}\s]/u.test(character)
      ? <Fragment key={index}>{character}</Fragment>
      : <span className={active ? "token-character active" : "token-character"} key={index}>{character}</span>
  ));
}

export function UtteranceList({ segments, tokens = [], speakerProfiles, highlightMs }: { segments: UtteranceSegment[]; tokens?: TranscriptionToken[]; speakerProfiles: SpeakerProfile[]; highlightMs: number | null }) {
  const [activeTokenId, setActiveTokenId] = useState<string | null>(null);
  const sortedTokens = useMemo(() => [...tokens].sort((left, right) => left.startMs - right.startMs || left.endMs - right.endMs), [tokens]);
  const tokensBySegmentId = useMemo(
    () => new Map(segments.map((segment) => [
      segment.id,
      sortedTokens.filter((token) => token.startMs >= segment.startMs && token.endMs <= segment.endMs)
    ])),
    [segments, sortedTokens]
  );

  useEffect(() => {
    document.querySelector(".segment.highlight")?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [highlightMs]);

  useEffect(() => {
    const updateActiveToken = (event: Event) => {
      const { currentMs } = (event as CustomEvent<PlaybackPositionEventDetail>).detail;
      setActiveTokenId(currentMs === null ? null : timedTokenAt(sortedTokens, currentMs)?.id ?? null);
    };
    window.addEventListener("recording-playback-position", updateActiveToken);
    return () => window.removeEventListener("recording-playback-position", updateActiveToken);
  }, [sortedTokens]);

  const profileById = new Map(speakerProfiles.map((profile) => [profile.id, profile]));
  if (segments.length === 0) return <div className="empty">连续发言尚未生成</div>;
  return (
    <div className="segments">
      {segments.map((segment) => {
        const profile = segment.matchedSpeakerProfileId ? profileById.get(segment.matchedSpeakerProfileId) : null;
        const highlighted = highlightMs !== null && highlightMs >= segment.startMs && highlightMs <= segment.endMs;
        return (
          <article
            className={`segment utterance-segment ${segment.isTargetPerson ? "target" : ""} ${highlighted ? "highlight" : ""}`}
            id={`utterance-${segment.id}`}
            key={segment.id}
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
            <div>{(() => {
              const segmentTokens = tokensBySegmentId.get(segment.id) ?? [];
              return segmentTokens.length
                ? segmentTokens.map((token) => (
                    <button
                      key={token.id}
                      type="button"
                      className="token"
                      onClick={() => {
                        window.dispatchEvent(
                          new CustomEvent("recording-play-segment", { detail: { startMs: token.startMs } })
                        );
                      }}
                    >
                      {tokenContent(token.text, token.id === activeTokenId)}
                    </button>
                  ))
                : segment.text;
            })()}</div>
          </article>
        );
      })}
    </div>
  );
}
