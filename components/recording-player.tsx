"use client";

import { useEffect, useRef } from "react";

interface PlaySegmentEventDetail {
  startMs: number;
  endMs?: number;
}

function dispatchPlaybackPosition(currentMs: number | null) {
  window.dispatchEvent(new CustomEvent("recording-playback-position", { detail: { currentMs } }));
}

export function RecordingPlayer({ src, seekMs }: { src: string; seekMs: number | null }) {
  const ref = useRef<HTMLAudioElement>(null);
  const clipEndSecondsRef = useRef<number | null>(null);
  const programmaticSeekRef = useRef(false);
  const pendingPlayStartSecondsRef = useRef<number | null>(null);

  useEffect(() => {
    const audio = ref.current;
    if (!audio || seekMs === null) return;
    const seek = () => {
      clipEndSecondsRef.current = null;
      programmaticSeekRef.current = true;
      audio.currentTime = Math.max(0, seekMs / 1000);
    };
    if (audio.readyState >= 1) seek();
    audio.addEventListener("loadedmetadata", seek, { once: true });
    return () => audio.removeEventListener("loadedmetadata", seek);
  }, [seekMs]);

  useEffect(() => {
    const audio = ref.current;
    if (!audio) return;
    let animationFrame: number | null = null;

    const seekTo = (seconds: number) => {
      programmaticSeekRef.current = true;
      if (typeof audio.fastSeek === "function") {
        audio.fastSeek(seconds);
      } else {
        audio.currentTime = seconds;
      }
    };

    const playAfterSeek = () => {
      const startSeconds = pendingPlayStartSecondsRef.current;
      if (startSeconds === null) return;
      pendingPlayStartSecondsRef.current = null;
      const playWhenSeeked = () => {
        void audio.play();
      };
      audio.addEventListener("seeked", playWhenSeeked, { once: true });
      seekTo(startSeconds);
      window.setTimeout(() => {
        audio.removeEventListener("seeked", playWhenSeeked);
        if (Math.abs(audio.currentTime - startSeconds) < 0.25) void audio.play();
      }, 250);
    };

    const playSegment = (event: Event) => {
      const { startMs, endMs } = (event as CustomEvent<PlaySegmentEventDetail>).detail;
      const startSeconds = Math.max(0, startMs / 1000);
      clipEndSecondsRef.current = endMs === undefined ? null : Math.max(startSeconds, endMs / 1000);
      pendingPlayStartSecondsRef.current = startSeconds;
      if (audio.readyState >= 1) {
        playAfterSeek();
      } else {
        audio.addEventListener("loadedmetadata", playAfterSeek, { once: true });
        audio.load();
      }
    };

    const stopAtSegmentEnd = () => {
      const clipEndSeconds = clipEndSecondsRef.current;
      if (clipEndSeconds === null || audio.currentTime < clipEndSeconds) return;
      audio.pause();
      audio.currentTime = clipEndSeconds;
      clipEndSecondsRef.current = null;
    };

    const clearSegmentOnManualSeek = () => {
      if (programmaticSeekRef.current) {
        programmaticSeekRef.current = false;
        return;
      }
      clipEndSecondsRef.current = null;
      pendingPlayStartSecondsRef.current = null;
    };

    const publishPlaybackPosition = () => {
      dispatchPlaybackPosition(Math.round(audio.currentTime * 1000));
      animationFrame = window.requestAnimationFrame(publishPlaybackPosition);
    };

    const startPublishingPlaybackPosition = () => {
      if (animationFrame !== null) return;
      animationFrame = window.requestAnimationFrame(publishPlaybackPosition);
    };

    const stopPublishingPlaybackPosition = () => {
      if (animationFrame !== null) {
        window.cancelAnimationFrame(animationFrame);
        animationFrame = null;
      }
      dispatchPlaybackPosition(null);
    };

    window.addEventListener("recording-play-segment", playSegment);
    audio.addEventListener("timeupdate", stopAtSegmentEnd);
    audio.addEventListener("seeking", clearSegmentOnManualSeek);
    audio.addEventListener("play", startPublishingPlaybackPosition);
    audio.addEventListener("pause", stopPublishingPlaybackPosition);
    audio.addEventListener("ended", stopPublishingPlaybackPosition);
    return () => {
      window.removeEventListener("recording-play-segment", playSegment);
      audio.removeEventListener("timeupdate", stopAtSegmentEnd);
      audio.removeEventListener("seeking", clearSegmentOnManualSeek);
      audio.removeEventListener("play", startPublishingPlaybackPosition);
      audio.removeEventListener("pause", stopPublishingPlaybackPosition);
      audio.removeEventListener("ended", stopPublishingPlaybackPosition);
      if (animationFrame !== null) window.cancelAnimationFrame(animationFrame);
      dispatchPlaybackPosition(null);
    };
  }, []);

  return <audio controls ref={ref} src={src} />;
}
