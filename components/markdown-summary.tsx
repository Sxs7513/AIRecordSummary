import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

function normalizeSummaryMarkdown(markdown: string) {
  return markdown
    .replace(/\r\n?/g, "\n")
    .split("\n")
    .map((line) => {
      const trimmed = line.trimStart();
      const leadingSpaces = line.length - trimmed.length;
      if (/^[·•]\s+/.test(trimmed)) return `${" ".repeat(Math.min(leadingSpaces, 2))}- ${trimmed.replace(/^[·•]\s+/, "")}`;
      if (/^[\u3000 ]{2,}\S/.test(line) && !/^(\s*)([-*+]|\d+\.)\s+/.test(line)) return line.trimStart();
      return line;
    })
    .join("\n");
}

export function MarkdownSummary({ markdown }: { markdown: string }) {
  const normalizedMarkdown = normalizeSummaryMarkdown(markdown);

  return (
    <div className="markdown-summary">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        allowedElements={[
          "p",
          "strong",
          "em",
          "ul",
          "ol",
          "li",
          "h1",
          "h2",
          "h3",
          "h4",
          "blockquote",
          "code",
          "pre",
          "br",
          "table",
          "thead",
          "tbody",
          "tr",
          "th",
          "td",
          "hr"
        ]}
      >
        {normalizedMarkdown}
      </ReactMarkdown>
    </div>
  );
}
