"use client";

import { create } from "zustand";
import type { Conversation, ConversationMessage, ConversationTurn, MessagePage } from "./types";

type ConversationState = {
  conversations: Conversation[];
  activeConversationId: string | null;
  messagesByConversation: Record<string, ConversationMessage[]>;
  setConversations: (items: Conversation[]) => void;
  removeConversation: (id: string) => void;
  setActiveConversation: (id: string | null) => void;
  hydrateMessages: (id: string, page: MessagePage) => void;
  reconcileTurn: (turn: ConversationTurn) => void;
  createOptimisticTurn: (conversationId: string, clientMessageId: string, text: string) => void;
  reconcileInitialTurn: (temporaryId: string, conversation: Conversation, turn: ConversationTurn) => void;
};

const order = (items: ConversationMessage[]) => [...items].sort((left, right) => left.sequence - right.sequence);

export const useConversationStore = create<ConversationState>((set) => ({
  conversations: [], activeConversationId: null, messagesByConversation: {},
  setConversations: (conversations) => set((state) => {
    const byId = new Map(conversations.map((item) => [item.id, item]));
    for (const item of state.conversations) if (item.id.startsWith("temporary:") && !byId.has(item.id)) byId.set(item.id, item);
    return { conversations: [...byId.values()] };
  }),
  removeConversation: (id) => set((state) => {
    const { [id]: _removed, ...messagesByConversation } = state.messagesByConversation;
    return {
      conversations: state.conversations.filter((conversation) => conversation.id !== id),
      messagesByConversation,
      activeConversationId: state.activeConversationId === id ? null : state.activeConversationId,
    };
  }),
  setActiveConversation: (activeConversationId) => set({ activeConversationId }),
  hydrateMessages: (id, page) => set((state) => {
    const byId = new Map((state.messagesByConversation[id] ?? []).map((item) => [item.id, item]));
    for (const item of page.items) byId.set(item.id, item);
    return { messagesByConversation: { ...state.messagesByConversation, [id]: order([...byId.values()]) } };
  }),
  reconcileTurn: (turn) => set((state) => {
    const id = turn.user_message.conversation_id;
    const existing = state.messagesByConversation[id] ?? [];
    const byId = new Map(existing.map((item) => [item.id, item]));
    byId.set(turn.user_message.id, turn.user_message); byId.set(turn.assistant_message.id, turn.assistant_message);
    return { messagesByConversation: { ...state.messagesByConversation, [id]: order([...byId.values()]) } };
  }),
  createOptimisticTurn: (conversationId, clientMessageId, text) => set((state) => {
    const now = new Date().toISOString();
    const conversation: Conversation = {
      id: conversationId, workspace_id: "", owner_user_id: "", title: text.slice(0, 80),
      archived_at: null, created_at: now, updated_at: now,
    };
    const userMessage: ConversationMessage = {
      id: `${conversationId}:user`, conversation_id: conversationId, role: "user", sequence: 1,
      reply_to_message_id: null, content_blocks: [{ type: "text", value: text }], sources: [],
      generation_run_id: null, status: "completed", client_message_id: clientMessageId,
      error_message: null, created_at: now, updated_at: now,
    };
    const assistantMessage: ConversationMessage = {
      ...userMessage, id: `${conversationId}:assistant`, role: "assistant", sequence: 2,
      reply_to_message_id: userMessage.id, content_blocks: [], status: "pending", client_message_id: null,
    };
    return {
      activeConversationId: conversationId,
      conversations: [conversation, ...state.conversations.filter((item) => item.id !== conversationId)],
      messagesByConversation: { ...state.messagesByConversation, [conversationId]: [userMessage, assistantMessage] },
    };
  }),
  reconcileInitialTurn: (temporaryId, conversation, turn) => set((state) => {
    const { [temporaryId]: _temporary, ...messagesByConversation } = state.messagesByConversation;
    const existing = messagesByConversation[conversation.id] ?? [];
    const byId = new Map(existing.map((item) => [item.id, item]));
    byId.set(turn.user_message.id, turn.user_message);
    byId.set(turn.assistant_message.id, turn.assistant_message);
    return {
      activeConversationId: conversation.id,
      conversations: [conversation, ...state.conversations.filter((item) => item.id !== temporaryId && item.id !== conversation.id)],
      messagesByConversation: { ...messagesByConversation, [conversation.id]: order([...byId.values()]) },
    };
  }),
}));
