"use client";

import { useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { Upload } from "lucide-react";
import { pythonApiUrl, responseDetail } from "@/app/sdk/python-api";

export function RecordingUploadForm() {
  const router = useRouter();
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    setError(null);
    setIsUploading(true);
    try {
      const formData = new FormData(form);
      const response = await fetch(pythonApiUrl("/api/recordings"), {
        method: "POST",
        body: formData,
        credentials: "include"
      });
      if (!response.ok) throw new Error(await responseDetail(response, "上传失败"));
      const payload = await response.json() as { recording: { id: string } };
      form.reset();
      router.push(`/recordings/${payload.recording.id}`);
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsUploading(false);
    }
  }

  return (
    <form className="toolbar" onSubmit={handleSubmit}>
      <label>
        录音文件
        <input name="audio" type="file" accept="audio/*" required />
      </label>
      <label>
        标题
        <input name="title" placeholder="可选" />
      </label>
      <button type="submit" disabled={isUploading}>
        <Upload size={16} />
        {isUploading ? "上传中" : "上传录音"}
      </button>
      {error ? <span className="subtle">{error}</span> : null}
    </form>
  );
}
