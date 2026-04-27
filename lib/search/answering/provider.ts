import type { RagAnswerInput, RagAnswerOutput } from "../types";

export interface RagAnswerProvider {
  generateAnswer(input: RagAnswerInput): Promise<RagAnswerOutput>;
}
