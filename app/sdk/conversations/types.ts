import type { ContentBlock } from "@/app/sdk/generation/types";

export type Conversation = { id: string; workspace_id: string; owner_user_id: string; title: string; archived_at: string | null; created_at: string; updated_at: string };
export type MessageStatus = "pending" | "streaming" | "completed" | "failed" | "cancelled";
export type ConversationMessage = {
  id: string; conversation_id: string; role: "user" | "assistant"; sequence: number; reply_to_message_id: string | null;
  content_blocks: ContentBlock[]; sources: Record<string, unknown>[]; generation_run_id: string | null; status: MessageStatus;
  client_message_id: string | null; error_message: string | null; created_at: string; updated_at: string;
};
export type MessagePage = { items: ConversationMessage[]; next_before: number | null; has_more: boolean };
export type ConversationTurn = { user_message: ConversationMessage; assistant_message: ConversationMessage; generation_run_id: string };
