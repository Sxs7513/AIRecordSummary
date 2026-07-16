"use client";

import { create } from "zustand";
import type { Conversation, ConversationMessage, ConversationTurn, MessagePage } from "./types";

type ConversationState = {
  conversations: Conversation[];
  activeConversationId: string | null;
  messagesByConversation: Record<string, ConversationMessage[]>;
  setConversations: (items: Conversation[]) => void;
  removeConversation: (id: string) => void;
  setActiveConversation: (id: string) => void;
  hydrateMessages: (id: string, page: MessagePage) => void;
  reconcileTurn: (turn: ConversationTurn) => void;
};

const order = (items: ConversationMessage[]) => [...items].sort((left, right) => left.sequence - right.sequence);

export const useConversationStore = create<ConversationState>((set) => ({
  conversations: [], activeConversationId: null, messagesByConversation: {},
  setConversations: (conversations) => set({ conversations }),
  removeConversation: (id) => set((state) => {
    const { [id]: _removed, ...messagesByConversation } = state.messagesByConversation;
    return {
      conversations: state.conversations.filter((conversation) => conversation.id !== id),
      messagesByConversation,
      activeConversationId: state.activeConversationId === id ? null : state.activeConversationId,
    };
  }),
  setActiveConversation: (activeConversationId) => set({ activeConversationId }),
  hydrateMessages: (id, page) => set((state) => ({ messagesByConversation: { ...state.messagesByConversation, [id]: order(page.items) } })),
  reconcileTurn: (turn) => set((state) => {
    const id = turn.user_message.conversation_id;
    const existing = state.messagesByConversation[id] ?? [];
    const byId = new Map(existing.map((item) => [item.id, item]));
    byId.set(turn.user_message.id, turn.user_message); byId.set(turn.assistant_message.id, turn.assistant_message);
    return { messagesByConversation: { ...state.messagesByConversation, [id]: order([...byId.values()]) } };
  })
}));
