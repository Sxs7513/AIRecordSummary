"use client";

import { UserRound } from "lucide-react";
import { useRouter } from "next/navigation";
import { type FormEvent, useState } from "react";
import { pythonApiUrl, responseDetail } from "@/app/sdk/python-api";

type SpeakerOption = {
  speakerClusterId: string;
  displayName: string;
};

export function SpeakerLabelForm({ recordingId, speakers }: { recordingId: string; speakers: SpeakerOption[] }) {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  if (speakers.length === 0) {
    return <div className="empty">暂无可配置的说话人标签</div>;
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const mappings = speakers.map((speaker) => ({
      speaker_cluster_id: speaker.speakerClusterId,
      display_name: String(formData.get(`speaker:${speaker.speakerClusterId}`) || "")
    }));
    const response = await fetch(pythonApiUrl(`/api/recordings/${recordingId}/speaker-mappings`), {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ mappings }),
      credentials: "include"
    });
    if (!response.ok) {
      setError(await responseDetail(response, "保存说话人失败"));
      return;
    }
    setError(null);
    router.refresh();
  }

  return (
    <form className="speaker-label-form" onSubmit={submit}>
      <div className="speaker-label-grid">
        {speakers.map((speaker) => (
          <label key={speaker.speakerClusterId}>
            <span>{speaker.speakerClusterId}</span>
            <input
              name={`speaker:${speaker.speakerClusterId}`}
              defaultValue={speaker.displayName}
              placeholder="输入真实姓名或展示名称"
            />
          </label>
        ))}
      </div>
      <button type="submit">
        <UserRound size={16} />
        保存说话人
      </button>
      {error ? <span className="subtle">{error}</span> : null}
    </form>
  );
}
