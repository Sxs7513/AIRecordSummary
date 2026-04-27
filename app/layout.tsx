import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI 录音检索",
  description: "录音入库、转写、说话人分离和目标人物识别"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>
        <div className="shell">
          <aside className="sidebar">
            <p className="brand">AI 录音检索</p>
            <nav className="nav">
              <Link href="/chat">问录音</Link>
              <Link href="/recordings">录音管理</Link>
              <Link href="/speaker-profiles">目标人物</Link>
            </nav>
          </aside>
          <main className="main">{children}</main>
        </div>
      </body>
    </html>
  );
}
