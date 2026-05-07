export function buildRouterPrompt(query: string) {
  const today = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Shanghai",
    year: "numeric",
    month: "2-digit",
    day: "2-digit"
  }).format(new Date());
  return (
    "<|im_start|>system\n" +
    "你要读懂用户想在录音库里找什么。\n" +
    "不要回答用户问题。只输出一个 JSON 对象，不要输出解释文字。\n" +
    `今天是 ${today}，时区是 Asia/Shanghai。\n` +
    "先用普通人的方式理解用户意图：用户要么是在问一个话题，要么是在问一批录音本身。\n" +
    "如果用户想了解某个概念、问题、关键词、事项或领域内容，就当作话题查询。把这个话题放到 topic，strategy 用 chunk_search。\n" +
    "如果用户想知道某条、某几条、某天、某段时间、某地点或某个人相关的录音整体讲了什么，就当作录音整体总结。topic 填 null，strategy 用 scope_summary。\n" +
    "如果一句话里既限定了录音范围，又问了一个具体话题，优先当作话题查询；范围只是限制要在哪些录音里找。\n" +
    "地点、人名、时间、最近第几条录音等，只是帮助缩小范围，不要因为出现这些词就忽略用户真正关心的话题。\n" +
    "在用户明确提到了地点时，把地点加入到 locations 字段里作为筛选项 \n" +
    "用户说相对时间时，保留原话到 timeRange.text 即可，不要自己计算 ISO 时间。只有用户给了明确绝对时间时，才考虑填写 dateRange。\n" +
    "不要编造录音 id 或人物 id。没有真实 id 时，recordingIds 和 speakerProfileIds 必须是空数组。用户提到了人名则放到 personNames 字段里作为筛选项。\n" +
    "字段含义：strategy 只能是 scope_summary 或 chunk_search；topic 是用户关心的话题；recordingLimit 是最近 N 条；recordingRank 是按创建时间倒序第 N 条；timeRange 是用户说的时间；dateRange 是明确的绝对时间范围；filters 里放地点、人名、已知 id 和目标人物限制。\n" +
    "输出 schema: {\"intent\":\"...\",\"strategy\":\"scope_summary|chunk_search\",\"topic\":null,\"recordingLimit\":null,\"recordingRank\":null,\"timeRange\":null,\"dateRange\":null,\"filters\":{\"recordingIds\":[],\"speakerProfileIds\":[],\"personNames\":[],\"locations\":[],\"targetPersonOnly\":false},\"needsAnswer\":true,\"reason\":\"...\"}\n" +
    "<|im_end|>\n" +
    "<|im_start|>user\n" +
    `${query}\n` +
    "<|im_end|>\n<|im_start|>assistant\n"
  );
}

export function firstJsonObject(text: string) {
  const start = text.indexOf("{");
  if (start < 0) return null;
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let index = start; index < text.length; index += 1) {
    const char = text[index];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === "\"") {
        inString = false;
      }
      continue;
    }
    if (char === "\"") {
      inString = true;
    } else if (char === "{") {
      depth += 1;
    } else if (char === "}") {
      depth -= 1;
      if (depth === 0) return text.slice(start, index + 1);
    }
  }
  return null;
}
