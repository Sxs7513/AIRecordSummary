"use client";

import { useEffect, useRef } from "react";

export function RecordingPlayer({ src, seekMs }: { src: string; seekMs: number | null }) {
  const ref = useRef<HTMLAudioElement>(null);
  useEffect(() => {
    const audio = ref.current;
    if (!audio || seekMs === null) return;
    const seek = () => {
      audio.currentTime = Math.max(0, seekMs / 1000);
    };
    if (audio.readyState >= 1) seek();
    audio.addEventListener("loadedmetadata", seek, { once: true });
    return () => audio.removeEventListener("loadedmetadata", seek);
  }, [seekMs]);

  return <audio controls ref={ref} src={src} />;
}
