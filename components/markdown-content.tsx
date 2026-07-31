"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export type CitationTarget = {
  index: number;
  href: string;
  title: string;
};

function normalizeMarkdown(markdown: string) {
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

export function MarkdownContent({ markdown, className, streaming = false, citations = [] }: {
  markdown: string;
  className?: string;
  streaming?: boolean;
  citations?: CitationTarget[];
}) {
  const [visibleLength, setVisibleLength] = useState(() => splitGraphemes(markdown).length);
  const targetRef = useRef(splitGraphemes(markdown));
  const visibleLengthRef = useRef(visibleLength);
  const animationFrame = useRef<number | null>(null);
  const isTyping = useRef(false);

  useEffect(() => {
    const target = splitGraphemes(markdown);
    const previousTarget = targetRef.current;
    const isAppend = markdown.startsWith(previousTarget.join(""));
    targetRef.current = target;

    if (!streaming || !isAppend) {
      if (!streaming && isTyping.current && visibleLengthRef.current < target.length) return;
      visibleLengthRef.current = target.length;
      isTyping.current = false;
      setVisibleLength(target.length);
      return;
    }
    if (visibleLengthRef.current < target.length) {
      isTyping.current = true;
      scheduleNextCharacter();
    }
  }, [markdown, streaming]);

  useEffect(() => () => {
    if (animationFrame.current !== null) window.cancelAnimationFrame(animationFrame.current);
  }, []);

  const visibleMarkdown = targetRef.current.slice(0, visibleLength).join("");

  function scheduleNextCharacter(): void {
    if (animationFrame.current !== null) return;
    animationFrame.current = window.requestAnimationFrame(() => {
      animationFrame.current = null;
      const nextLength = Math.min(visibleLengthRef.current + 1, targetRef.current.length);
      visibleLengthRef.current = nextLength;
      setVisibleLength(nextLength);
      if (nextLength < targetRef.current.length) scheduleNextCharacter();
      else isTyping.current = false;
    });
  }

  return (
    <div className={className}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, ...props }) => (
            <a {...props} className="citation-link" href={href} rel="noreferrer" target="_blank" />
          )
        }}
        allowedElements={[
          "a",
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
        {linkifyCitations(normalizeMarkdown(visibleMarkdown), citations)}
      </ReactMarkdown>
    </div>
  );
}

function linkifyCitations(markdown: string, citations: CitationTarget[]): string {
  if (citations.length === 0) return markdown;
  const citationsByIndex = new Map(citations.map((citation) => [citation.index, citation]));
  return markdown.replace(/\[(\d+)\]/g, (reference, rawIndex: string) => {
    const citation = citationsByIndex.get(Number(rawIndex));
    if (!citation) return reference;
    return `[\\[${citation.index}\\]](${citation.href} \"${citation.title.replaceAll('"', "'")}\")`;
  });
}

function splitGraphemes(value: string): string[] {
  type Segmenter = { segment: (input: string) => Iterable<{ segment: string }> };
  type SegmenterFactory = new (locales: string, options: { granularity: "grapheme" }) => Segmenter;
  const Segmenter = (Intl as typeof Intl & { Segmenter?: SegmenterFactory }).Segmenter;
  return Segmenter ? Array.from(new Segmenter("zh-CN", { granularity: "grapheme" }).segment(value), (part) => part.segment) : Array.from(value);
}
