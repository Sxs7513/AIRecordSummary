import { pythonApiUrl, responseDetail } from "@/app/sdk/python-api";
import type { Conversation, ConversationTurn, MessagePage } from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(pythonApiUrl(path), { credentials: "include", cache: "no-store", ...init });
  if (!response.ok) throw new Error(await responseDetail(response, `请求失败：${response.status}`));
  return response.json() as Promise<T>;
}

export const listConversations = () => request<Conversation[]>("/api/conversations");
export const createConversation = () => request<Conversation>("/api/conversations", { method: "POST", headers: { "content-type": "application/json" }, body: "{}" });
export async function deleteConversation(conversationId: string): Promise<void> {
  const response = await fetch(pythonApiUrl(`/api/conversations/${encodeURIComponent(conversationId)}`), {
    method: "DELETE", credentials: "include"
  });
  if (!response.ok) throw new Error(await responseDetail(response, `删除对话失败：${response.status}`));
}
export const getMessages = (conversationId: string) => request<MessagePage>(`/api/conversations/${encodeURIComponent(conversationId)}/messages`);
export const sendMessage = (conversationId: string, text: string, clientMessageId: string) => request<ConversationTurn>(
  `/api/conversations/${encodeURIComponent(conversationId)}/messages`,
  { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ content_blocks: [{ type: "text", value: text }], client_message_id: clientMessageId }) }
);
