import { getAppConfig } from "@/lib/config/app-config";
import { streamLocalLlmAnswer } from "@/lib/search/answering/local-llm-stream";
import { ExtractiveAnswerProvider } from "@/lib/search/answering/extractive";
import { getAnswerProvider } from "@/lib/search/answering";
import { ragLog, textPreview } from "@/lib/search/debug";
import { validateRagAnswer } from "@/lib/search/rag";
import { retrieveSearchEvidence } from "@/lib/search/search";
import type { RagAnswer } from "@/lib/types/models";

export const dynamic = "force-dynamic";

function event(name: string, data: unknown) {
  return `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;
}

function enqueue(controller: ReadableStreamDefaultController<Uint8Array>, encoder: TextEncoder, name: string, data: unknown) {
  controller.enqueue(encoder.encode(event(name, data)));
}

async function emitChunkedAnswer(answer: RagAnswer, emitDelta: (text: string) => void) {
  for (const piece of answer.text.match(/.{1,24}/gs) ?? []) {
    emitDelta(piece);
  }
}

export async function POST(request: Request) {
  const body = await request.json();
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      const startedAt = Date.now();
      try {
        const config = getAppConfig();
        const query = String(body.query ?? "");
        const mode = body.mode === "retrieve_only" ? "retrieve_only" : "answer";
        ragLog("stream.request", {
          mode,
          queryPreview: textPreview(query),
          limit: typeof body.limit === "number" ? body.limit : null,
          filters: body.filters ?? {}
        });

        const retrieved = await retrieveSearchEvidence({
          query,
          limit: typeof body.limit === "number" ? body.limit : undefined,
          filters: body.filters ?? {}
        });
        enqueue(controller, encoder, "evidence", {
          queryId: retrieved.queryId,
          evidence: retrieved.evidence,
          message: retrieved.message
        });

        if (mode === "retrieve_only") {
          enqueue(controller, encoder, "answer_done", { answer: null });
          return;
        }

        if (retrieved.evidence.length === 0) {
          const answer: RagAnswer = {
            text: "没有在录音中找到足够依据。",
            citations: [],
            notEnoughEvidence: true
          };
          enqueue(controller, encoder, "answer_delta", { text: answer.text });
          enqueue(controller, encoder, "answer_done", { answer });
          return;
        }

        let answer: RagAnswer;
        const emitDelta = (text: string) => enqueue(controller, encoder, "answer_delta", { text });
        const emitThinking = (thinkingEvent: "start" | "done", text?: string) => {
          enqueue(controller, encoder, thinkingEvent === "start" ? "thinking_start" : "thinking_done", { text: text ?? "" });
        };
        if (config.search.answerEnabled && config.search.answerProvider === "local_llm") {
          ragLog("stream.answer.start", {
            queryId: retrieved.queryId,
            provider: "local_llm",
            evidenceCount: retrieved.evidence.length
          });
          answer = validateRagAnswer(
            await streamLocalLlmAnswer(
              { query, evidence: retrieved.evidence, outputLanguage: "zh-CN" },
              {
                pythonBin: config.search.embeddingPythonBin,
                modelCacheRoot: config.audio.modelCacheRoot,
                modelRepo: config.search.answerModelRepo,
                modelFile: config.search.answerModelFile,
                contextSize: config.search.answerContextSize,
                timeoutMs: config.search.answerTimeoutMs
              },
              emitDelta,
              emitThinking
            ),
            retrieved.evidence
          );
        } else {
          const provider = config.search.answerEnabled ? getAnswerProvider() : new ExtractiveAnswerProvider();
          answer = validateRagAnswer(await provider.generateAnswer({ query, evidence: retrieved.evidence, outputLanguage: "zh-CN" }), retrieved.evidence);
          await emitChunkedAnswer(answer, emitDelta);
        }

        enqueue(controller, encoder, "answer_done", { answer });
        ragLog("stream.done", {
          queryId: retrieved.queryId,
          answerLength: answer.text.length,
          citationCount: answer.citations.length,
          durationMs: Date.now() - startedAt
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        ragLog("stream.error", { message, durationMs: Date.now() - startedAt });
        enqueue(controller, encoder, "error", { message });
      } finally {
        controller.close();
      }
    }
  });
  return new Response(stream, {
    headers: {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache, no-transform"
    }
  });
}
