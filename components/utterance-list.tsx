"use client";

import { Fragment, useEffect, useMemo, useState } from "react";
import type { SpeakerProfile, TranscriptionSegment, TranscriptionToken, UtteranceSegment } from "@/app/shared/models";
import { formatMs } from "@/app/shared/format";

interface PlaybackPositionEventDetail {
  currentMs: number | null;
}

interface HighlightRange {
  startMs: number;
  endMs: number;
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

function highlightOffsets(text: string, phrase: string | null): Set<number> {
  if (!phrase) return new Set();
  const characters = Array.from(text);
  const phraseCharacters = Array.from(phrase);
  const offsets = new Set<number>();
  if (phraseCharacters.length === 0 || phraseCharacters.length > characters.length) return offsets;
  for (let index = 0; index <= characters.length - phraseCharacters.length; index += 1) {
    if (!phraseCharacters.every((character, phraseIndex) => characters[index + phraseIndex] === character)) continue;
    for (let phraseIndex = 0; phraseIndex < phraseCharacters.length; phraseIndex += 1) offsets.add(index + phraseIndex);
  }
  return offsets;
}

function tokenContent(text: string, active: boolean, highlightedOffsets: Set<number>, offset: number) {
  return Array.from(text).map((character, index) => {
    const highlighted = highlightedOffsets.has(offset + index);
    if (/[\p{P}\s]/u.test(character)) {
      return highlighted ? <mark className="expression-highlight" key={index}>{character}</mark> : <Fragment key={index}>{character}</Fragment>;
    }
    const className = ["token-character", active ? "active" : "", highlighted ? "expression-highlight" : ""].filter(Boolean).join(" ");
    return <span className={className} key={index}>{character}</span>;
  });
}

function tokenOverlapMs(token: TranscriptionToken, segment: UtteranceSegment): number {
  return Math.max(0, Math.min(token.endMs, segment.endMs) - Math.max(token.startMs, segment.startMs));
}

function tokenDistanceMs(token: TranscriptionToken, segment: UtteranceSegment): number {
  if (token.endMs < segment.startMs) return segment.startMs - token.endMs;
  if (token.startMs > segment.endMs) return token.startMs - segment.endMs;
  return 0;
}

function assignTokensToSegments(segments: UtteranceSegment[], tokens: TranscriptionToken[]): Map<string, TranscriptionToken[]> {
  const assigned = new Map(segments.map((segment) => [segment.id, [] as TranscriptionToken[]]));
  for (const token of tokens) {
    const overlapping = segments
      .map((segment) => ({ segment, overlapMs: tokenOverlapMs(token, segment) }))
      .filter((candidate) => candidate.overlapMs > 0);
    const candidates = overlapping.length > 0
      ? overlapping
      : segments.map((segment) => ({ segment, overlapMs: 0 }));
    if (candidates.length === 0) continue;

    const sameSpeaker = token.speakerClusterId
      ? candidates.filter((candidate) => candidate.segment.speakerClusterId === token.speakerClusterId)
      : [];
    const ranked = sameSpeaker.length > 0 ? sameSpeaker : candidates;
    ranked.sort((left, right) => {
      if (left.overlapMs !== right.overlapMs) return right.overlapMs - left.overlapMs;
      const leftDistance = tokenDistanceMs(token, left.segment);
      const rightDistance = tokenDistanceMs(token, right.segment);
      return leftDistance - rightDistance || left.segment.utteranceIndex - right.segment.utteranceIndex;
    });
    assigned.get(ranked[0].segment.id)?.push(token);
  }
  return assigned;
}

export function UtteranceList({ segments, transcriptionSegments = [], tokens = [], speakerProfiles, highlightRange, highlightText = null }: { segments: UtteranceSegment[]; transcriptionSegments?: TranscriptionSegment[]; tokens?: TranscriptionToken[]; speakerProfiles: SpeakerProfile[]; highlightRange: HighlightRange | null; highlightText?: string | null }) {
  const [activeTokenId, setActiveTokenId] = useState<string | null>(null);
  const [originalVisibleIds, setOriginalVisibleIds] = useState<Set<string>>(() => new Set());
  const sortedTokens = useMemo(() => [...tokens].sort((left, right) => left.startMs - right.startMs || left.endMs - right.endMs), [tokens]);
  const tokensBySegmentId = useMemo(() => assignTokensToSegments(segments, sortedTokens), [segments, sortedTokens]);
  const originalByUtteranceId = useMemo(() => {
    const transcriptionById = new Map(transcriptionSegments.map((segment) => [segment.id, segment]));
    return new Map(segments.map((segment) => {
      const texts = segment.sourceTranscriptionSegmentIds
        .map((id) => transcriptionById.get(id)?.originalText?.trim())
        .filter((text): text is string => Boolean(text));
      return [segment.id, [...new Set(texts)].join("\n") || null];
    }));
  }, [segments, transcriptionSegments]);
  const expressionFound = useMemo(() => Boolean(highlightText) && segments.some((segment) => {
    const inRange = highlightRange === null || (segment.startMs <= highlightRange.endMs && segment.endMs >= highlightRange.startMs);
    return inRange && highlightOffsets(segment.text, highlightText).size > 0;
  }), [highlightRange, highlightText, segments]);

  useEffect(() => {
    document.querySelector(".expression-highlight")?.scrollIntoView({ block: "center", behavior: "smooth" });
    document.querySelector(".segment.highlight")?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [highlightRange?.startMs, highlightRange?.endMs, highlightText]);

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
    <div className="transcript-views">
      <div className="segments">
          {segments.map((segment) => {
            const profile = segment.matchedSpeakerProfileId ? profileById.get(segment.matchedSpeakerProfileId) : null;
            const originalText = originalByUtteranceId.get(segment.id);
            const showOriginal = originalVisibleIds.has(segment.id);
            const highlighted = highlightRange !== null
              && segment.startMs <= highlightRange.endMs
              && segment.endMs >= highlightRange.startMs;
            const expressionOffsets = highlighted ? highlightOffsets(segment.text, highlightText) : new Set<number>();
            const showRangeHighlight = highlighted && (!highlightText || !expressionFound);
            return (
              <article
                className={`segment utterance-segment ${segment.isTargetPerson ? "target" : ""} ${showRangeHighlight ? "highlight" : ""}`}
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
                  {originalText ? (
                    <button
                      className="secondary"
                      onClick={() => setOriginalVisibleIds((current) => {
                        const next = new Set(current);
                        if (next.has(segment.id)) next.delete(segment.id); else next.add(segment.id);
                        return next;
                      })}
                      type="button"
                    >
                      {showOriginal ? "返回润色" : "查看原文"}
                    </button>
                  ) : null}
                </div>
                <div>{showOriginal ? originalText : (() => {
                  const segmentTokens = tokensBySegmentId.get(segment.id) ?? [];
                  const reconstructedText = segmentTokens.map((token) => token.text).join("");
                  if (segmentTokens.length > 0 && reconstructedText === segment.text) {
                    let offset = 0;
                    return segmentTokens.map((token) => {
                      const tokenOffset = offset;
                      offset += Array.from(token.text).length;
                      return (
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
                          {tokenContent(token.text, token.id === activeTokenId, expressionOffsets, tokenOffset)}
                        </button>
                      );
                    });
                  }
                  return tokenContent(segment.text, false, expressionOffsets, 0);
                })()}</div>
              </article>
            );
          })}
      </div>
    </div>
  );
}
