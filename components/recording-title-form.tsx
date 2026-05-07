import { Pencil } from "lucide-react";

export function RecordingTitleForm({ recordingId, title }: { recordingId: string; title: string }) {
  return (
    <form className="toolbar" action={`/api/recordings/${recordingId}/title`} method="post">
      <label>
        录音标题
        <input name="title" defaultValue={title} required maxLength={160} placeholder="输入用于问答和展示的录音标题" />
      </label>
      <button type="submit">
        <Pencil size={16} />
        保存标题
      </button>
    </form>
  );
}
