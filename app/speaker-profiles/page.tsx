import { Upload } from "lucide-react";
import { SpeakerProfileForm } from "@/components/speaker-profile-form";
import { StatusBadge } from "@/components/status-badge";
import { listSpeakerProfiles } from "@/lib/db/speaker-profiles";
import { formatBytes, formatDate } from "@/lib/types/format";

export const dynamic = "force-dynamic";

export default async function SpeakerProfilesPage() {
  const profiles = await listSpeakerProfiles();

  return (
    <>
      <div className="topbar">
        <div>
          <h1>目标人物</h1>
          <p className="subtle">管理目标人物参考样本，识别结果会在录音详情中以置信度展示。</p>
        </div>
      </div>

      <section className="panel">
        <SpeakerProfileForm />
      </section>

      <section className="grid" style={{ marginTop: 16 }}>
        {profiles.length === 0 ? (
          <div className="empty">还没有目标人物</div>
        ) : (
          profiles.map((profile) => (
            <article className="panel" key={profile.id}>
              <div className="topbar" style={{ marginBottom: 10 }}>
                <div>
                  <h2>{profile.displayName}</h2>
                  <p className="subtle">
                    {profile.notes || "无备注"} · <StatusBadge status={profile.status} />
                  </p>
                </div>
                <form action={`/api/speaker-profiles/${profile.id}/delete`} method="post">
                  <button className="danger" type="submit">
                    删除
                  </button>
                </form>
              </div>

              <form className="toolbar" action={`/api/speaker-profiles/${profile.id}/samples`} method="post" encType="multipart/form-data">
                <label>
                  参考音频
                  <input name="audio" type="file" accept="audio/*" required />
                </label>
                <button type="submit">
                  <Upload size={16} />
                  上传样本
                </button>
              </form>

              <table style={{ marginTop: 12 }}>
                <thead>
                  <tr>
                    <th>样本</th>
                    <th>大小</th>
                    <th>状态</th>
                    <th>创建时间</th>
                  </tr>
                </thead>
                <tbody>
                  {profile.samples.length === 0 ? (
                    <tr>
                      <td colSpan={4} className="subtle">
                        暂无样本
                      </td>
                    </tr>
                  ) : (
                    profile.samples.map((sample) => (
                      <tr key={sample.id}>
                        <td>{sample.fileName}</td>
                        <td>{formatBytes(sample.fileSizeBytes)}</td>
                        <td>
                          <StatusBadge status={sample.status} />
                        </td>
                        <td>{formatDate(sample.createdAt)}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </article>
          ))
        )}
      </section>
    </>
  );
}
