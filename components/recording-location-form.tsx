"use client";

import { MapPin } from "lucide-react";
import { useRouter } from "next/navigation";
import { type FormEvent, useState } from "react";
import { pythonApiUrl, responseDetail } from "@/app/sdk/python-api";

export function RecordingLocationForm({ recordingId, location }: { recordingId: string; location: string | null }) {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const value = new FormData(event.currentTarget).get("location");
    const response = await fetch(pythonApiUrl(`/api/recordings/${recordingId}`), {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ location: typeof value === "string" ? value : null }),
      credentials: "include"
    });
    if (!response.ok) {
      setError(await responseDetail(response, "保存地点失败"));
      return;
    }
    setError(null);
    router.refresh();
  }

  return (
    <form className="toolbar" onSubmit={submit}>
      <label>
        录音发生地点
        <input name="location" defaultValue={location ?? ""} placeholder="例如 北京、上海、会议室 A" />
      </label>
      <button type="submit">
        <MapPin size={16} />
        保存地点
      </button>
      {error ? <span className="subtle">{error}</span> : null}
    </form>
  );
}
