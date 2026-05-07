import { MapPin } from "lucide-react";

export function RecordingLocationForm({ recordingId, location }: { recordingId: string; location: string | null }) {
  return (
    <form className="toolbar" action={`/api/recordings/${recordingId}/location`} method="post">
      <label>
        录音发生地点
        <input name="location" defaultValue={location ?? ""} placeholder="例如 北京、上海、会议室 A" />
      </label>
      <button type="submit">
        <MapPin size={16} />
        保存地点
      </button>
    </form>
  );
}

