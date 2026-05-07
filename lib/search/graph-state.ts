import { Annotation } from "@langchain/langgraph";
import type { RagAnswer, RagQueryResponse, SearchEvidence, SearchFilters } from "../types/models";
import type { RagRoute } from "./router/route-schema";

export interface EvidenceGrade {
  sufficient: boolean;
  reason: string;
  missingAspects: string[];
  rewriteQuery: string | null;
  confidence: number;
}

export interface AnswerValidation {
  valid: boolean;
  failureKind: "none" | "bad_citation" | "unsupported_claim" | "wrong_evidence" | "not_enough_evidence";
  reason: string;
  unsupportedClaims: string[];
  badCitations: number[];
  rewriteInstruction: string | null;
  rewriteQuery: string | null;
}

export const RagGraphAnnotation = Annotation.Root({
  query: Annotation<string>,
  mode: Annotation<"answer" | "retrieve_only">,
  limit: Annotation<number | undefined>,
  filters: Annotation<SearchFilters | undefined>,
  route: Annotation<RagRoute | null>,
  retrievalQuery: Annotation<string>,
  retrievalAttempt: Annotation<number>,
  answerRewriteCount: Annotation<number>,
  queryId: Annotation<string | null>,
  evidence: Annotation<SearchEvidence[]>,
  message: Annotation<string | undefined>,
  grade: Annotation<EvidenceGrade | null>,
  answer: Annotation<RagAnswer | null>,
  validation: Annotation<AnswerValidation | null>,
  error: Annotation<string | null>
});

export type RagGraphState = typeof RagGraphAnnotation.State;
export type RagGraphUpdate = typeof RagGraphAnnotation.Update;

export function initialRagGraphState(input: {
  query: string;
  mode?: "answer" | "retrieve_only";
  limit?: number;
  filters?: SearchFilters;
}): RagGraphUpdate {
  return {
    query: input.query,
    mode: input.mode ?? "answer",
    limit: input.limit,
    filters: input.filters,
    route: null,
    retrievalQuery: input.query,
    retrievalAttempt: 0,
    answerRewriteCount: 0,
    queryId: null,
    evidence: [],
    message: undefined,
    grade: null,
    answer: null,
    validation: null,
    error: null
  };
}

export function graphStateToResponse(state: RagGraphState): RagQueryResponse {
  return {
    queryId: state.queryId ?? "",
    query: state.query,
    answer: state.answer,
    evidence: state.evidence,
    message: state.message
  };
}
