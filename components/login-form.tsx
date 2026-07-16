"use client";

import { useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useAuthStore } from "@/app/sdk/auth/store";

export function LoginForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const signIn = useAuthStore((state) => state.signIn);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null); setSubmitting(true);
    try {
      await signIn(email, password);
      const next = searchParams.get("next");
      router.replace(next?.startsWith("/") ? next : "/recordings");
      router.refresh();
    }
    catch (reason) { setError(reason instanceof Error ? reason.message : "登录失败"); }
    finally { setSubmitting(false); }
  }

  return <form className="card" onSubmit={submit} style={{ maxWidth: 420 }}>
    <h1>登录</h1>
    <label>邮箱<input type="email" required value={email} onChange={(event) => setEmail(event.target.value)} /></label>
    <label>密码<input type="password" required value={password} onChange={(event) => setPassword(event.target.value)} /></label>
    {error ? <p className="error">{error}</p> : null}
    <button type="submit" disabled={submitting}>{submitting ? "登录中…" : "登录"}</button>
  </form>;
}
