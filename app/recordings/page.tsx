import Link from "next/link";
import { AutoRefresh } from "@/components/auto-refresh";
import { RecordingUploadForm } from "@/components/recording-upload-form";
import { StatusBadge } from "@/components/status-badge";
import { listRecordings } from "@/lib/db/recordings";
import { formatDate, formatDurationMs } from "@/lib/types/format";

export const dynamic = "force-dynamic";

export default async function RecordingsPage({
  searchParams
}: {
  searchParams: Promise<{ status?: string; page?: string }>;
}) {
  const params = await searchParams;
  const status = params.status || "all";
  const page = Number(params.page || 1);
  const data = await listRecordings({ status, page, pageSize: 10 });
  const hasActiveJobs = data.items.some((item) => item.status === "uploaded" || item.status === "processing");

  return (
    <>
      <AutoRefresh enabled={hasActiveJobs} />
      <div className="topbar">
        <div>
          <h1>录音管理</h1>
          <p className="subtle">上传录音，查看处理状态、转写结果、说话人标签和目标人物命中。</p>
        </div>
      </div>

      <section className="grid stats">
        {["uploaded", "processing", "completed", "failed"].map((key) => (
          <div className="card stat" key={key}>
            <span className="subtle">{key}</span>
            <strong>{data.stats[key] ?? 0}</strong>
          </div>
        ))}
      </section>

      <section className="panel" style={{ marginTop: 16 }}>
        <RecordingUploadForm />
      </section>

      <section className="panel" style={{ marginTop: 16 }}>
        <form className="toolbar" method="get" action="/recordings">
          <label>
            状态筛选
            <select name="status" defaultValue={status}>
              {["all", "uploaded", "processing", "completed", "failed"].map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          <button type="submit">筛选</button>
        </form>

        {data.items.length === 0 ? (
          <div className="empty" style={{ marginTop: 14 }}>
            还没有录音记录
          </div>
        ) : (
          <table style={{ marginTop: 14 }}>
            <thead>
              <tr>
                <th>标题</th>
                <th>状态</th>
                <th>处理耗时</th>
                <th>上传时间</th>
                <th>更新时间</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {data.items.map((item) => (
                <tr key={item.id}>
                  <td>
                    <Link href={`/recordings/${item.id}`}>
                      <strong>{item.title}</strong>
                      <br />
                      <span className="subtle">{item.fileName}</span>
                      {item.location ? (
                        <>
                          <br />
                          <span className="subtle">地点：{item.location}</span>
                        </>
                      ) : null}
                    </Link>
                  </td>
                  <td>
                    <StatusBadge status={item.status} />
                  </td>
                  <td>{formatDurationMs(item.processingDurationMs)}</td>
                  <td>{formatDate(item.uploadedAt)}</td>
                  <td>{formatDate(item.updatedAt)}</td>
                  <td>
                    <form action={`/api/recordings/${item.id}/delete`} method="post">
                      <button className="danger" type="submit">
                        删除
                      </button>
                    </form>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </>
  );
}
