import { MarkdownContent } from "@/components/markdown-content";

export function MarkdownSummary({ markdown, streaming = false }: { markdown: string; streaming?: boolean }) {
  return <MarkdownContent className="markdown-summary" markdown={markdown} streaming={streaming} />;
}
