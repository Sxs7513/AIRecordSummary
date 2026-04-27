import { UserPlus } from "lucide-react";

export function SpeakerProfileForm() {
  return (
    <form className="toolbar" action="/api/speaker-profiles" method="post">
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
      <button type="submit">
        <UserPlus size={16} />
        创建目标人物
      </button>
    </form>
  );
}
