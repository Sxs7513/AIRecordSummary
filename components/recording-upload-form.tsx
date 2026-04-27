"use client";

import { useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { Upload } from "lucide-react";

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
      const response = await fetch("/api/recordings", {
        method: "POST",
        body: formData
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "上传失败");
      }
      form.reset();
      if (payload.recordings?.length === 1) {
        router.push(`/recordings/${payload.recordings[0].id}?jobId=${payload.jobIds?.[0] || ""}`);
      } else {
        router.push("/recordings");
      }
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
        <input name="audio" type="file" accept="audio/*" multiple required />
      </label>
      <label>
        标题
        <input name="title" placeholder="单文件上传时可指定" />
      </label>
      <button type="submit" disabled={isUploading}>
        <Upload size={16} />
        {isUploading ? "上传中" : "上传录音"}
      </button>
      {error ? <span className="subtle">{error}</span> : null}
    </form>
  );
}
