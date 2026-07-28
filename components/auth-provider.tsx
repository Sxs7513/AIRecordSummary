"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect } from "react";

import { useAuthStore } from "@/app/sdk/auth/store";

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const { user, loading, load } = useAuthStore();
  const isLoginPage = pathname === "/login";
  const isChatPage = pathname === "/chat" || pathname.startsWith("/chat/");

  useEffect(() => { void load(); }, [load]);
  useEffect(() => {
    if (!isLoginPage && !loading && user === null) router.replace(`/login?next=${encodeURIComponent(pathname)}`);
  }, [isLoginPage, loading, pathname, router, user]);

  if (isLoginPage) return <main className="login-main">{children}</main>;
  if (loading || user === null) return <main className="auth-loading">正在验证登录状态…</main>;
  if (isChatPage) return <main className="chat-main">{children}</main>;
  return <div className="shell"><aside className="sidebar"><p className="brand">AI 录音检索</p><nav className="nav"><Link href="/chat">问录音</Link><Link href="/recordings">录音管理</Link><Link href="/asr-lab">ASR Lab</Link><Link href="/account">账号</Link></nav></aside><main className="main">{children}</main></div>;
}
