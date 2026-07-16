"use client";

import { UserPlus } from "lucide-react";
import { type FormEvent, useState } from "react";
import { useRouter } from "next/navigation";
import { pythonApiUrl, responseDetail } from "@/app/sdk/python-api";

export function SpeakerProfileForm() {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const response = await fetch(pythonApiUrl("/api/speaker-profiles"), {
        method: "POST",
        body: new FormData(event.currentTarget),
        credentials: "include"
      });
      if (!response.ok) throw new Error(await responseDetail(response, "创建目标人物失败"));
      event.currentTarget.reset();
      router.refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "创建目标人物失败");
    } finally {
      setSaving(false);
    }
  }

  return (
    <form className="toolbar" onSubmit={submit}>
      <label>
        姓名
        <input name="displayName" required />
      </label>
      <label>
        状态
        <select name="status" defaultValue="active">
          <option value="active">active</option>
          <option value="inactive">inactive</option>
        </select>
      </label>
      <label>
        备注
        <textarea name="notes" rows={1} />
      </label>
      <button type="submit" disabled={saving}>
        <UserPlus size={16} />
        {saving ? "创建中" : "创建目标人物"}
      </button>
      {error ? <span className="subtle">{error}</span> : null}
    </form>
  );
}
