import { END, START, StateGraph } from "@langchain/langgraph";
import { getAppConfig } from "../config/app-config";
import { getRecordingSummaryEvidence, listCompletedRecordingIdsByRank, listRecentCompletedRecordingIds } from "../db/search";
import { getAnswerProvider } from "./answering";
import { ExtractiveAnswerProvider } from "./answering/extractive";
import { elapsedMs, ragLog, textPreview } from "./debug";
import { graphStateToResponse, initialRagGraphState, RagGraphAnnotation, type AnswerValidation, type EvidenceGrade, type RagGraphState, type RagGraphUpdate } from "./graph-state";
import { validateRagAnswer } from "./grading/answer-validator";
import { validateAnswerWithLocalLlm } from "./grading/llm-answer-validator";
import { retrieveSearchEvidence } from "./search";
import type { RagQueryInput } from "./types";
import type { RagAnswer } from "../types/models";
import { routeQueryWithLocalLlm } from "./router/local-llm-router";
import { query } from "../db/pool";

function mergeRecordingScopes(primary?: string[], secondary?: string[]) {
  if (!primary?.length) return secondary;
  if (!secondary?.length) return primary;
  const secondarySet = new Set(secondary);
  return primary.filter((id) => secondarySet.has(id));
}

async function buildScopeFilters(state: RagGraphState) {
  const route = state.route;
  const recentRecordingIds = route?.recordingLimit ? await listRecentCompletedRecordingIds(route.recordingLimit) : undefined;
  const rankedRecordingIds = route?.recordingRank ? await listCompletedRecordingIdsByRank(route.recordingRank) : undefined;
  const routeRecordingIds = route?.filters.recordingIds?.length ? route.filters.recordingIds : undefined;
  const stateRecordingIds = state.filters?.recordingIds;
  const recordingIds = mergeRecordingScopes(mergeRecordingScopes(mergeRecordingScopes(routeRecordingIds, stateRecordingIds), recentRecordingIds), rankedRecordingIds);
  return {
    filters: {
      ...(state.filters ?? {}),
      recordingIds,
      speakerProfileIds: route?.filters.speakerProfileIds?.length ? route.filters.speakerProfileIds : state.filters?.speakerProfileIds,
      personNames: route?.filters.personNames?.length ? route.filters.personNames : state.filters?.personNames,
      locations: route?.filters.locations?.length ? route.filters.locations : state.filters?.locations,
      targetPersonOnly: route?.filters.targetPersonOnly || state.filters?.targetPersonOnly || false,
      createdFrom: route?.dateRange?.from ?? state.filters?.createdFrom ?? state.filters?.uploadedFrom,
      createdTo: route?.dateRange?.to ?? state.filters?.createdTo ?? state.filters?.uploadedTo
    },
    missingRecordingScope: Boolean((route?.recordingLimit || route?.recordingRank) && recordingIds?.length === 0)
  };
}

async function routeNode(state: RagGraphState): Promise<RagGraphUpdate> {
  const startedAt = Date.now();
  const queryForRouting = state.retrievalAttempt > 0 ? state.retrievalQuery : state.query;
  const route = await routeQueryWithLocalLlm(queryForRouting);
  ragLog("graph.route", {
    queryPreview: textPreview(queryForRouting),
    strategy: route.strategy,
    intent: route.intent,
    topic: route.topic,
    personNames: route.filters.personNames,
    locations: route.filters.locations,
    speakerProfileIds: route.filters.speakerProfileIds,
    recordingLimit: route.recordingLimit,
    recordingRank: route.recordingRank,
    timeRange: route.timeRange,
    dateRange: route.dateRange,
    durationMs: elapsedMs(startedAt)
  });
  return {
    route,
    retrievalQuery: route.topic || state.query
  };
}

async function retrieveNode(state: RagGraphState): Promise<RagGraphUpdate> {
  const startedAt = Date.now();
  const route = state.route;
  const chunkSearch = route?.strategy === "chunk_search";
  const { filters, missingRecordingScope } = await buildScopeFilters(state);
  if (missingRecordingScope) {
    return {
      queryId: `scope-${Date.now()}`,
      evidence: [],
      message: "没有找到符合范围的已完成录音"
    };
  }

  if (!chunkSearch) {
    const evidence = await getRecordingSummaryEvidence({
      filters
    });
    ragLog("graph.retrieve_scope", {
      recordingLimit: route?.recordingLimit,
      recordingRank: route?.recordingRank,
      recordingScope: filters.recordingIds,
      personNames: filters.personNames,
      locations: filters.locations,
      createdFrom: filters.createdFrom,
      createdTo: filters.createdTo,
      evidenceCount: evidence.length,
      recordingIds: evidence.map((item) => item.recording.id),
      durationMs: elapsedMs(startedAt)
    });
    return {
      queryId: `scope-${Date.now()}`,
      evidence,
      message: evidence.length === 0 ? "没有找到符合范围的已完成录音" : undefined
    };
  }

  const result = await retrieveSearchEvidence({
    query: state.retrievalQuery || state.query,
    limit: state.limit,
    filters
  });
  ragLog("graph.retrieve_vector", {
    queryId: result.queryId,
    retrievalQuery: textPreview(state.retrievalQuery || state.query),
    recordingScope: filters.recordingIds,
    personNames: filters.personNames,
    locations: filters.locations,
    speakerProfileIds: filters.speakerProfileIds,
    targetPersonOnly: filters.targetPersonOnly,
    createdFrom: filters.createdFrom,
    createdTo: filters.createdTo,
    evidenceCount: result.evidence.length,
    durationMs: elapsedMs(startedAt)
  });
  return {
    queryId: result.queryId,
    evidence: result.evidence,
    message: result.message
  };
}

async function gradeNode(state: RagGraphState): Promise<RagGraphUpdate> {
  const route = state.route;
  const coveredRecordings = new Set(state.evidence.map((item) => item.recording.id));
  let sufficient = state.evidence.length > 0;
  const missingAspects: string[] = [];
  if (route?.strategy === "scope_summary" && route.recordingLimit && coveredRecordings.size < route.recordingLimit) {
    sufficient = false;
    missingAspects.push("requested_recent_recordings");
  }
  if (route?.strategy === "chunk_search" && state.evidence[0] && state.evidence[0].score < getAppConfig().search.minScore) {
    sufficient = false;
    missingAspects.push("high_confidence_topic_evidence");
  }
  const grade: EvidenceGrade = {
    sufficient,
    reason: sufficient ? "Evidence passes first-pass coverage checks" : `Evidence is insufficient: ${missingAspects.join(", ") || "empty evidence"}`,
    missingAspects,
    rewriteQuery: sufficient ? null : state.route?.topic || state.query,
    confidence: sufficient ? 0.75 : 0.35
  };
  ragLog("graph.grade", {
    sufficient: grade.sufficient,
    reason: grade.reason,
    missingAspects: grade.missingAspects,
    rewriteQuery: grade.rewriteQuery,
    retrievalAttempt: state.retrievalAttempt
  });
  return { grade };
}

async function rewriteQueryNode(state: RagGraphState): Promise<RagGraphUpdate> {
  const rewritten = state.validation?.rewriteQuery || state.grade?.rewriteQuery || state.route?.topic || state.query;
  ragLog("graph.rewrite_query", {
    from: textPreview(state.retrievalQuery || state.query),
    to: textPreview(rewritten),
    nextAttempt: state.retrievalAttempt + 1
  });
  return {
    retrievalQuery: rewritten,
    retrievalAttempt: state.retrievalAttempt + 1,
    route: null,
    grade: null,
    validation: null,
    answer: null
  };
}

async function answerNode(state: RagGraphState): Promise<RagGraphUpdate> {
  if (state.mode === "retrieve_only") return { answer: null };
  if (state.evidence.length === 0 || state.grade?.sufficient === false) {
    const answer: RagAnswer = {
      text: "没有在录音中找到足够依据。",
      citations: [],
      notEnoughEvidence: true
    };
    return { answer };
  }

  const config = getAppConfig().search;
  const provider = config.answerEnabled ? getAnswerProvider() : new ExtractiveAnswerProvider();
  const startedAt = Date.now();
  const answer = await provider.generateAnswer({ query: state.query, evidence: state.evidence, outputLanguage: "zh-CN" });
  ragLog("graph.answer", {
    provider: config.answerEnabled ? config.answerProvider : "extractive_disabled_answer",
    answerLength: answer.text.length,
    citationCount: answer.citations.length,
    durationMs: elapsedMs(startedAt),
    query: state.query,
  });
  return { answer };
}

async function validateNode(state: RagGraphState): Promise<RagGraphUpdate> {
  if (!state.answer) {
    return { validation: { valid: true, failureKind: "none", reason: "retrieve_only", unsupportedClaims: [], badCitations: [], rewriteInstruction: null, rewriteQuery: null } };
  }
  try {
    const answer = validateRagAnswer(state.answer, state.evidence);
    const llmValidation = await validateAnswerWithLocalLlm({ query: state.query, evidence: state.evidence, answer });
    const validation: AnswerValidation = llmValidation ?? {
      valid: true,
      failureKind: "none",
      reason: "Programmatic citation validation passed",
      unsupportedClaims: [],
      badCitations: [],
      rewriteInstruction: null,
      rewriteQuery: null
    };
    ragLog("graph.validate", {
      valid: validation.valid,
      failureKind: validation.failureKind,
      citationCount: answer.citations.length,
      retrievalAttempt: state.retrievalAttempt,
      answerRewriteCount: state.answerRewriteCount
    });
    return { answer, validation };
  } catch (error) {
    const hasEvidence = state.evidence.length > 0;
    const failureKind: AnswerValidation["failureKind"] = hasEvidence ? "bad_citation" : "not_enough_evidence";
    const validation: AnswerValidation = {
      valid: false,
      failureKind,
      reason: error instanceof Error ? error.message : String(error),
      unsupportedClaims: [],
      badCitations: state.answer.citations.map((citation) => citation.index),
      rewriteInstruction: hasEvidence ? "Use only the provided evidence and include valid citation indexes." : null,
      rewriteQuery: state.route?.topic || state.query
    };
    ragLog("graph.validate", {
      valid: false,
      failureKind,
      reason: validation.reason,
      retrievalAttempt: state.retrievalAttempt,
      answerRewriteCount: state.answerRewriteCount
    });
    return { validation };
  }
}

async function rewriteAnswerNode(state: RagGraphState): Promise<RagGraphUpdate> {
  ragLog("graph.rewrite_answer", {
    answerRewriteCount: state.answerRewriteCount + 1,
    reason: state.validation?.reason
  });
  const answer = await new ExtractiveAnswerProvider().generateAnswer({ query: state.query, evidence: state.evidence });
  return {
    answer,
    answerRewriteCount: state.answerRewriteCount + 1
  };
}

function afterGrade(state: RagGraphState) {
  if (!state.grade?.sufficient && state.retrievalAttempt < 1) return "rewrite_query";
  return "answer";
}

function afterValidate(state: RagGraphState) {
  if (!state.validation || state.validation.valid) return END;
  if ((state.validation.failureKind === "wrong_evidence" || state.validation.failureKind === "not_enough_evidence") && state.retrievalAttempt < 1) {
    return "rewrite_query";
  }
  if ((state.validation.failureKind === "bad_citation" || state.validation.failureKind === "unsupported_claim") && state.answerRewriteCount < 1) {
    return "rewrite_answer";
  }
  return END;
}

const ragGraph = new StateGraph(RagGraphAnnotation)
  .addNode("node_route", routeNode)
  .addNode("node_retrieve", retrieveNode)
  .addNode("node_grade", gradeNode)
  .addNode("node_rewrite_query", rewriteQueryNode)
  .addNode("node_answer", answerNode)
  .addNode("node_validate", validateNode)
  .addNode("node_rewrite_answer", rewriteAnswerNode)
  .addEdge(START, "node_route")
  .addEdge("node_route", "node_retrieve")
  .addEdge("node_retrieve", "node_grade")
  .addConditionalEdges("node_grade", afterGrade, {
    rewrite_query: "node_rewrite_query",
    answer: "node_answer"
  })
  .addEdge("node_rewrite_query", "node_route")
  .addEdge("node_answer", "node_validate")
  .addConditionalEdges("node_validate", afterValidate, {
    rewrite_query: "node_rewrite_query",
    rewrite_answer: "node_rewrite_answer",
    [END]: END
  })
  .addEdge("node_rewrite_answer", "node_validate")
  .compile();

export async function runRagGraph(input: RagQueryInput) {
  const startedAt = Date.now();
  const finalState = await ragGraph.invoke(initialRagGraphState(input), {
    recursionLimit: 20
  });
  ragLog("graph.done", {
    queryId: finalState.queryId,
    route: finalState.route?.strategy,
    evidenceCount: finalState.evidence.length,
    answerLength: finalState.answer?.text.length ?? 0,
    durationMs: elapsedMs(startedAt)
  });
  return graphStateToResponse(finalState);
}
