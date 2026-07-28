import { Suspense } from "react";
import { LoginForm } from "@/components/login-form";

export default function LoginPage() {
  return <Suspense fallback={<div className="card">正在加载登录页面…</div>}><LoginForm /></Suspense>;
}
