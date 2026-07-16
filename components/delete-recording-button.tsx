"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { pythonApiUrl, responseDetail } from "@/app/sdk/python-api";

export function DeleteRecordingButton({ recordingId }: { recordingId: string }) {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);

  async function remove() {
    if (!window.confirm("确定要删除这条录音及其处理结果吗？")) return;
    const response = await fetch(pythonApiUrl(`/api/recordings/${recordingId}`), { method: "DELETE", credentials: "include" });
    if (!response.ok) {
      setError(await responseDetail(response, "删除录音失败"));
      return;
    }
    router.push("/recordings");
    router.refresh();
  }

  return (
    <>
      <button className="danger" type="button" onClick={remove}>
        删除
      </button>
      {error ? <span className="subtle">{error}</span> : null}
    </>
  );
}
