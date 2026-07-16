"use client";

import { Upload } from "lucide-react";
import { type FormEvent, useState } from "react";
import { useRouter } from "next/navigation";
import { pythonApiUrl, responseDetail } from "@/app/sdk/python-api";

export function DeleteSpeakerProfileButton({ profileId }: { profileId: string }) {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);

  async function remove() {
    if (!window.confirm("确定要删除该目标人物及其参考样本吗？")) return;
    const response = await fetch(pythonApiUrl(`/api/speaker-profiles/${profileId}/delete`), { method: "POST", credentials: "include" });
    if (!response.ok) {
      setError(await responseDetail(response, "删除目标人物失败"));
      return;
    }
    setError(null);
    router.refresh();
  }

  return <><button className="danger" type="button" onClick={() => void remove()}>删除</button>{error ? <span className="subtle">{error}</span> : null}</>;
}

export function SpeakerProfileSampleForm({ profileId }: { profileId: string }) {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setUploading(true);
    setError(null);
    try {
      const response = await fetch(pythonApiUrl(`/api/speaker-profiles/${profileId}/samples`), {
        method: "POST",
        body: new FormData(event.currentTarget),
        credentials: "include"
      });
      if (!response.ok) throw new Error(await responseDetail(response, "上传参考音频失败"));
      event.currentTarget.reset();
      router.refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "上传参考音频失败");
    } finally {
      setUploading(false);
    }
  }

  return (
    <form className="toolbar" onSubmit={submit}>
      <label>
        参考音频
        <input name="audio" type="file" accept="audio/*" required />
      </label>
      <button type="submit" disabled={uploading}>
        <Upload size={16} />
        {uploading ? "上传中" : "上传样本"}
      </button>
      {error ? <span className="subtle">{error}</span> : null}
    </form>
  );
}
