import { pythonApiUrl, responseDetail } from "@/app/sdk/python-api";
import type { AuthUser } from "./types";

export async function getCurrentUser(): Promise<AuthUser | null> {
  const response = await fetch(pythonApiUrl("/api/auth/me"), { credentials: "include", cache: "no-store" });
  if (response.status === 401) return null;
  if (!response.ok) throw new Error(await responseDetail(response, "读取账号状态失败"));
  return response.json() as Promise<AuthUser>;
}

export async function login(email: string, password: string): Promise<AuthUser> {
  const response = await fetch(pythonApiUrl("/api/auth/login"), {
    method: "POST",
    credentials: "include",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ email, password })
  });
  if (!response.ok) throw new Error(await responseDetail(response, "登录失败"));
  const payload = await response.json() as { user: AuthUser };
  return payload.user;
}

export async function logout(): Promise<void> {
  const response = await fetch(pythonApiUrl("/api/auth/logout"), { method: "POST", credentials: "include" });
  if (!response.ok) throw new Error(await responseDetail(response, "退出登录失败"));
}
