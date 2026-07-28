import Link from "next/link";

export default function AsrLabLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="asr-lab-page">
      <div className="asr-lab-tabs">
        <Link href="/asr-lab/datasets">数据标注</Link>
        <Link href="/asr-lab/evaluations">模型评测</Link>
        <Link href="/asr-lab/training-runs">训练记录</Link>
      </div>
      {children}
    </div>
  );
}

