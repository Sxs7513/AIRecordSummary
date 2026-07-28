import { Suspense } from "react";
import { AsrDatasetWorkspace } from "@/components/asr-lab/asr-dataset-workspace";

export default function AsrDatasetsPage() {
  return <Suspense fallback={<div className="panel">正在加载 ASR 数据集…</div>}><AsrDatasetWorkspace /></Suspense>;
}
