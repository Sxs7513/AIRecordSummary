import { readFile } from "node:fs/promises";
import path from "node:path";

export interface TranscriptionPromptConfig {
  enabled?: boolean;
  intro?: string;
  prompt?: string;
  terms?: string[];
  protectTerms?: string[];
  llmTerms?: string[];
  phrases?: string[];
  people?: string[];
  notes?: string[];
  llmCorrection?: {
    system?: string[];
    userTemplate?: string;
    mergeSystem?: string[];
    mergeUserTemplate?: string;
    maxLlmTerms?: number;
    maxPhrases?: number;
    maxPeople?: number;
  };
  replacements?: Record<string, string>;
  regexReplacements?: Array<{
    pattern: string;
    replace: string;
    flags?: string;
  }>;
}

function cleanItems(items: unknown): string[] {
  if (!Array.isArray(items)) return [];
  return items.map((item) => String(item).trim()).filter(Boolean);
}

export async function loadWhisperInitialPrompt(configPath: string): Promise<string | null> {
  const config = await loadTranscriptionPromptConfig(configPath);
  if (!config || config.enabled === false) return null;

  const prompt = String(config.prompt || "").trim();
  const terms = cleanItems(config.terms);
  const phrases = cleanItems(config.phrases);
  const people = cleanItems(config.people);
  const notes = cleanItems(config.notes);

  if (!prompt && terms.length === 0 && phrases.length === 0 && people.length === 0 && notes.length === 0) {
    return null;
  }

  const sections = [
    String(config.intro || "").trim(),
    prompt ? `上下文：${prompt}` : "",
    terms.length > 0 ? `专业名词：${terms.join("、")}` : "",
    phrases.length > 0 ? `固定短语：${phrases.join("、")}` : "",
    people.length > 0 ? `人名：${people.join("、")}` : "",
    notes.length > 0 ? `其他提示：${notes.join("；")}` : ""
  ].filter(Boolean);

  return sections.join("\n");
}

export async function loadAsrHotwords(configPath: string, maxHotwords: number): Promise<string[]> {
  if (maxHotwords <= 0) return [];
  const config = await loadTranscriptionPromptConfig(configPath);
  if (!config || config.enabled === false) return [];

  const seen = new Set<string>();
  const hotwords: string[] = [];
  for (const item of [...cleanItems(config.terms), ...cleanItems(config.protectTerms)]) {
    if (seen.has(item)) continue;
    seen.add(item);
    hotwords.push(item);
    if (hotwords.length >= maxHotwords) break;
  }
  return hotwords;
}

export async function loadQwenAsrContext(configPath: string, maxItems: number, extraContext = ""): Promise<string | null> {
  const config = await loadTranscriptionPromptConfig(configPath);
  if (!config || config.enabled === false) {
    const fallback = extraContext.trim();
    return fallback || null;
  }

  const takeUnique = (items: string[], seen: Set<string>, remaining: () => number): string[] => {
    const output: string[] = [];
    for (const item of items) {
      if (maxItems > 0 && remaining() <= 0) break;
      if (seen.has(item)) continue;
      seen.add(item);
      output.push(item);
    }
    return output;
  };

  const seen = new Set<string>();
  let usedItems = 0;
  const remaining = () => maxItems - usedItems;
  const take = (items: string[]) => {
    const selected = takeUnique(items, seen, remaining);
    usedItems += selected.length;
    return selected;
  };

  const terms = take(cleanItems(config.terms));
  const phrases = take(cleanItems(config.phrases));
  const people = take(cleanItems(config.people));
  const notes = take(cleanItems(config.notes));

  const sections = [
    extraContext.trim(),
    "这是半导体、光电子、材料物理相关的技术讨论。以下内容是可能出现的专业术语、人名或固定表达，仅在语音和上下文匹配时优先采用，不要主动补充未出现的信息。",
    terms.length > 0 ? `专业术语：${terms.join("、")}` : "",
    phrases.length > 0 ? `固定表达：${phrases.join("、")}` : "",
    people.length > 0 ? `人名：${people.join("、")}` : "",
    notes.length > 0 ? `其他提示：${notes.join("；")}` : ""
  ].filter(Boolean);

  if (sections.length === 1 && sections[0] === extraContext.trim()) {
    return sections[0] || null;
  }

  return sections.length > 0 ? sections.join("\n") : null;
}

export async function loadTranscriptionPromptConfig(configPath: string): Promise<TranscriptionPromptConfig | null> {
  const absolutePath = path.isAbsolute(configPath) ? configPath : path.join(process.cwd(), configPath);
  let rawConfig: string;
  try {
    rawConfig = await readFile(absolutePath, "utf8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return null;
    throw error;
  }

  return JSON.parse(rawConfig) as TranscriptionPromptConfig;
}

export function applyTranscriptionTextCorrections(text: string, config: TranscriptionPromptConfig | null): string {
  if (!config || config.enabled === false || !text) return text;

  let output = text;
  const replacements = Object.entries(config.replacements ?? {}).filter(([from]) => from);
  replacements.sort(([a], [b]) => b.length - a.length);
  for (const [from, to] of replacements) {
    output = output.replaceAll(from, to);
  }

  for (const replacement of config.regexReplacements ?? []) {
    if (!replacement.pattern) continue;
    const flags = replacement.flags?.includes("g") ? replacement.flags : `${replacement.flags || ""}g`;
    output = output.replace(new RegExp(replacement.pattern, flags), replacement.replace);
  }

  return output;
}
