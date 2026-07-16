"use client";

import { Pencil } from "lucide-react";
import { useRouter } from "next/navigation";
import { type FormEvent, useState } from "react";
import { pythonApiUrl, responseDetail } from "@/app/sdk/python-api";

export function RecordingTitleForm({ recordingId, title }: { recordingId: string; title: string }) {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const value = new FormData(event.currentTarget).get("title");
    const response = await fetch(pythonApiUrl(`/api/recordings/${recordingId}`), {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ title: typeof value === "string" ? value : "" }),
      credentials: "include"
    });
    if (!response.ok) {
      setError(await responseDetail(response, "保存标题失败"));
      return;
    }
    setError(null);
    router.refresh();
  }

  return (
    <form className="toolbar" onSubmit={submit}>
      <label>
        录音标题
        <input name="title" defaultValue={title} required maxLength={160} placeholder="输入用于问答和展示的录音标题" />
      </label>
      <button type="submit">
        <Pencil size={16} />
        保存标题
      </button>
      {error ? <span className="subtle">{error}</span> : null}
    </form>
  );
}
