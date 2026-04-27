"use client";

import { useEffect, useState } from "react";

interface Progress {
  percent: number;
  updatedAt: string;
}

export function RecordingProgress({
  recordingId,
  status
}: {
  recordingId: string;
  status: string;
}) {
  const [progress, setProgress] = useState<Progress | null>(null);
  const enabled = status === "running";

  useEffect(() => {
    if (!enabled) {
      setProgress(null);
      return;
    }

    let cancelled = false;
    async function load() {
      try {
        const response = await fetch(`/api/recordings/${recordingId}/progress`, { cache: "no-store" });
        const payload = await response.json();
        if (!cancelled) setProgress(payload.progress ?? null);
      } catch {
        if (!cancelled) setProgress(null);
      }
    }

    load();
    const timer = setInterval(load, 1200);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [enabled, recordingId]);

  if (status === "completed") return <span className="progress-percent">100%</span>;
  if (!enabled) return <span className="subtle">-</span>;
  if (!progress) return <span className="progress-percent">0%</span>;

  return (
    <div className="job-progress">
      <div className="job-progress-head">
        <span>{progress.percent}%</span>
      </div>
      <div className="job-progress-track">
        <div className="progress-bar" style={{ width: `${Math.max(0, Math.min(100, progress.percent))}%` }} />
      </div>
    </div>
  );
}
