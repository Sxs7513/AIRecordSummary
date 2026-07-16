"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/app/sdk/auth/store";

export function AccountPanel() {
  const router = useRouter();
  const { user, loading, load, signOut } = useAuthStore();
  useEffect(() => { void load(); }, [load]);
  if (loading) return <p>正在读取账号信息…</p>;
  if (!user) return <p>当前未登录。</p>;
  const workspace = user.memberships.find((item) => item.id === user.current_workspace_id);
  return <section className="card"><h1>账号</h1><p>{user.display_name} · {user.email}</p><p>当前默认工作区：{workspace?.name ?? "未配置"}</p><p>角色：{workspace?.role ?? "—"}</p><button onClick={async () => { await signOut(); router.replace("/login"); router.refresh(); }}>退出登录</button></section>;
}
