import { RagChat } from "@/components/rag-chat";

export const dynamic = "force-dynamic";

export default function ChatPage() {
  return (
    <>
      <div className="topbar">
        <div>
          <h1>问录音</h1>
          <p className="subtle">基于已完成录音片段检索证据并生成带引用的回答</p>
        </div>
      </div>
      <RagChat />
    </>
  );
}
