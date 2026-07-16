"use client";

import { create } from "zustand";
import { getCurrentUser, login, logout } from "./client";
import type { AuthUser } from "./types";

type AuthState = {
  user: AuthUser | null;
  loading: boolean;
  load: () => Promise<void>;
  signIn: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
};

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  loading: true,
  load: async () => {
    set({ loading: true });
    try { set({ user: await getCurrentUser() }); } finally { set({ loading: false }); }
  },
  signIn: async (email, password) => set({ user: await login(email, password), loading: false }),
  signOut: async () => { await logout(); set({ user: null, loading: false }); }
}));
