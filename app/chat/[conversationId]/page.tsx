import { ConversationChat } from "@/components/conversation-chat";

export const dynamic = "force-dynamic";

export default async function ConversationPage({ params }: { params: Promise<{ conversationId: string }> }) {
  const { conversationId } = await params;
  return <ConversationChat conversationId={conversationId} />;
}
