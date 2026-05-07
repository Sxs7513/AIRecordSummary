import { UserRound } from "lucide-react";

export function SpeakerLabelForm({ recordingId, speakerLabels }: { recordingId: string; speakerLabels: string[] }) {
  if (speakerLabels.length === 0) {
    return <div className="empty">暂无可配置的说话人标签</div>;
  }

  return (
    <form className="speaker-label-form" action={`/api/recordings/${recordingId}/speakers`} method="post">
      <div className="speaker-label-grid">
        {speakerLabels.map((label) => (
          <label key={label}>
            <span>{label}</span>
            <input name={`speaker:${label}`} defaultValue={label} placeholder="输入真实姓名或展示名称" />
          </label>
        ))}
      </div>
      <button type="submit">
        <UserRound size={16} />
        保存说话人
      </button>
    </form>
  );
}

