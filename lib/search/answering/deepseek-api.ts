import type { RagAnswerInput, RagAnswerOutput } from "../types";
import type { RagAnswerProvider } from "./provider";

export class DeepSeekAnswerProvider implements RagAnswerProvider {
  constructor(private readonly options: { apiKey: string; baseUrl: string; model: string; timeoutMs: number }) {}

  async generateAnswer(input: RagAnswerInput): Promise<RagAnswerOutput> {
    if (!this.options.apiKey) throw new Error("DEEPSEEK_API_KEY is required");
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.options.timeoutMs);
    try {
      const response = await fetch(`${this.options.baseUrl.replace(/\/$/, "")}/chat/completions`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${this.options.apiKey}`
        },
        body: JSON.stringify({
          model: this.options.model,
          response_format: { type: "json_object" },
          messages: [
            { role: "system", content: systemPrompt() },
            { role: "user", content: JSON.stringify({ query: input.query, evidence: input.evidence }, null, 2) }
          ]
        }),
        signal: controller.signal
      });
      if (!response.ok) throw new Error(`DeepSeek API failed: ${response.status} ${await response.text()}`);
      const payload = await response.json() as { choices?: Array<{ message?: { content?: string } }> };
      return parseAnswer(payload.choices?.[0]?.message?.content ?? "");
    } finally {
      clearTimeout(timer);
    }
  }
}

function systemPrompt() {
  return "你只能基于用户提供的录音证据回答。输出 JSON：{\"text\":\"...\",\"citations\":[{\"index\":1,\"chunkId\":\"...\",\"recordingId\":\"...\",\"startMs\":0,\"endMs\":0}],\"notEnoughEvidence\":false}。关键结论必须带 [编号] 引用；证据不足时 notEnoughEvidence=true。";
}

function parseAnswer(text: string): RagAnswerOutput {
  const parsed = JSON.parse(text) as RagAnswerOutput;
  return {
    text: String(parsed.text ?? ""),
    citations: Array.isArray(parsed.citations) ? parsed.citations : [],
    notEnoughEvidence: Boolean(parsed.notEnoughEvidence)
  };
}
