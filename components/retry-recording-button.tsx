"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { pythonApiUrl, responseDetail } from "@/app/sdk/python-api";

export function RetryRecordingButton({ recordingId }: { recordingId: string }) {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);

  async function retry() {
    const response = await fetch(pythonApiUrl(`/api/recordings/${recordingId}/retry`), { method: "POST", credentials: "include" });
    if (!response.ok) {
      setError(await responseDetail(response, "创建重试任务失败"));
      return;
    }
    setError(null);
    router.refresh();
  }

  return (
    <>
      <button className="secondary" type="button" onClick={retry}>重新处理</button>
      {error ? <span className="subtle">{error}</span> : null}
    </>
  );
}
