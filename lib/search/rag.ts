import { runRagGraph } from "./graph";
import type { RagQueryInput } from "./types";
import type { RagQueryResponse } from "../types/models";

export { validateRagAnswer } from "./grading/answer-validator";

export async function runRagQuery(input: RagQueryInput): Promise<RagQueryResponse> {
  return runRagGraph(input);
}
