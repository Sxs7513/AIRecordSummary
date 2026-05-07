import { readFile } from "node:fs/promises";
import path from "node:path";

export interface SummaryPromptConfig {
  enabled?: boolean;
  system?: string[];
  summary?: {
    enabled?: boolean;
    system?: string[];
  };
}

export const DEFAULT_SUMMARY_SYSTEM_PROMPT = [
  "请总结这段录音，只能根据录音文本写，不要编造。",
  "开头先写一个全局总结，用 1-2 段话说明这段录音整体在讨论什么、最重要的结论或结果是什么。",
  "按照录音里的先后顺序总结，可以自然分成几段，每段围绕一个真实讨论主题或连续发生的事情展开。",
  "每段标题要直接写具体主题，不要使用“阶段一/阶段二/片段一/片段二/第一阶段/第二阶段”这类流程标签。",
  "总结要比逐句复述更高一层，写清楚这段讨论在解决什么问题、形成了什么看法、有哪些结论或待办。",
  "不要机械写成“Speaker A 说……、Speaker B 说……”这种发言记录。只有人物身份本身重要时才提到人。",
  "每段都要保留具体事情、数字、结论和待办，不要只写空泛概括。",
  "用自然的大白话写，不要写成报告腔，也不要只写空泛概括。",
  "用 Markdown 输出。不要使用代码块或缩进代码格式。不要输出思考过程，不要输出 JSON。"
].join("\n");

export async function loadSummarySystemPrompt(configPath: string) {
  const absolutePath = path.isAbsolute(configPath) ? configPath : path.join(process.cwd(), configPath);
  try {
    const rawConfig = await readFile(absolutePath, "utf8");
    const config = JSON.parse(rawConfig) as SummaryPromptConfig;
    const summaryConfig = config.summary ?? config;
    if (summaryConfig.enabled === false) return DEFAULT_SUMMARY_SYSTEM_PROMPT;
    const system = Array.isArray(summaryConfig.system) ? summaryConfig.system.map((item) => String(item).trim()).filter(Boolean) : [];
    return system.length > 0 ? system.join("\n") : DEFAULT_SUMMARY_SYSTEM_PROMPT;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return DEFAULT_SUMMARY_SYSTEM_PROMPT;
    throw error;
  }
}
