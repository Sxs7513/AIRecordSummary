import Link from "next/link";
import { redirect } from "next/navigation";
import { AutoRefresh } from "@/components/auto-refresh";
import { DeleteRecordingButton } from "@/components/delete-recording-button";
import { RecordingUploadForm } from "@/components/recording-upload-form";
import { StatusBadge } from "@/components/status-badge";
import { listPythonRecordings } from "@/app/data/recordings";
import { formatDate, formatDurationMs } from "@/app/shared/format";

export const dynamic = "force-dynamic";

const PAGE_SIZE = 10;

function recordingListHref(status: string, page: number) {
  const query = new URLSearchParams();
  if (status !== "all") query.set("status", status);
  if (page > 1) query.set("page", String(page));
  const search = query.toString();
  return search ? `/recordings?${search}` : "/recordings";
}

export default async function RecordingsPage({
  searchParams
}: {
  searchParams: Promise<{ status?: string; page?: string }>;
}) {
  const params = await searchParams;
  const status = params.status || "all";
  const parsedPage = Number(params.page);
  const page = Number.isInteger(parsedPage) && parsedPage > 0 ? parsedPage : 1;
  const data = await listPythonRecordings({ status, page, pageSize: PAGE_SIZE });
  const totalPages = Math.max(1, Math.ceil(data.total / data.pageSize));
  if (data.total > 0 && page > totalPages) redirect(recordingListHref(status, totalPages));
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
                    <DeleteRecordingButton recordingId={item.id} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {data.total > 0 ? (
          <nav
            aria-label="录音列表分页"
            className="toolbar"
            style={{ justifyContent: "space-between", marginTop: 14 }}
          >
            <span className="subtle">
              共 {data.total} 条，第 {page} / {totalPages} 页
            </span>
            <div className="toolbar">
              {page > 1 ? (
                <Link className="button secondary" href={recordingListHref(status, page - 1)} rel="prev">
                  上一页
                </Link>
              ) : (
                <span aria-disabled="true" className="button secondary" style={{ cursor: "default", opacity: 0.5 }}>
                  上一页
                </span>
              )}
              {page < totalPages ? (
                <Link className="button secondary" href={recordingListHref(status, page + 1)} rel="next">
                  下一页
                </Link>
              ) : (
                <span aria-disabled="true" className="button secondary" style={{ cursor: "default", opacity: 0.5 }}>
                  下一页
                </span>
              )}
            </div>
          </nav>
        ) : null}
      </section>
    </>
  );
}
