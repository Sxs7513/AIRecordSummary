import { z } from "zod";

const legacyStrategyMap: Record<string, "scope_summary" | "chunk_search"> = {
  recent_recording_summary: "scope_summary",
  date_range_summary: "scope_summary",
  vector_search: "chunk_search",
  scoped_vector_search: "chunk_search"
};

export const retrievalStrategySchema = z.preprocess((value) => {
  if (typeof value === "string" && value in legacyStrategyMap) return legacyStrategyMap[value];
  return value;
}, z.enum(["scope_summary", "chunk_search"]));

const nullableString = z.string().trim().min(1).nullable().default(null);

const nullablePositiveInt = z.preprocess((value) => {
  if (typeof value === "string" && /^\d+$/.test(value.trim())) return Number(value);
  return value;
}, z.number().int().min(1).max(366).nullable().default(null));

const timeRangeUnitSchema = z.preprocess((value) => {
  if (typeof value !== "string") return value;
  const normalized = value.trim().toLowerCase();
  const unitMap: Record<string, "day" | "week" | "month" | "year"> = {
    day: "day",
    days: "day",
    天: "day",
    日: "day",
    week: "week",
    weeks: "week",
    周: "week",
    星期: "week",
    month: "month",
    months: "month",
    月: "month",
    个月: "month",
    year: "year",
    years: "year",
    年: "year"
  };
  return unitMap[normalized] ?? value;
}, z.enum(["day", "week", "month", "year"]).nullable().default(null));

const timeRangeDirectionSchema = z.preprocess((value) => {
  if (typeof value !== "string") return value;
  const normalized = value.trim().toLowerCase();
  const directionMap: Record<string, "past" | "current" | "previous"> = {
    past: "past",
    recent: "past",
    rolling: "past",
    最近: "past",
    近: "past",
    过去: "past",
    current: "current",
    this: "current",
    本: "current",
    这: "current",
    previous: "previous",
    last: "previous",
    上: "previous",
    昨: "previous",
    前: "previous"
  };
  return directionMap[normalized] ?? value;
}, z.enum(["past", "current", "previous"]).nullable().default(null));

export const timeRangeSchema = z
  .preprocess((value) => {
    if (typeof value === "string") return { text: value };
    if (value && typeof value === "object" && !Array.isArray(value)) {
      const input = value as Record<string, unknown>;
      if (typeof input.kind === "string") {
        const legacyMap: Record<string, Partial<Record<"text" | "type" | "unit" | "direction", string>>> = {
          today: { text: "今天", type: "relative", unit: "day", direction: "current" },
          yesterday: { text: "昨天", type: "relative", unit: "day", direction: "previous" },
          day_before_yesterday: { text: "前天", type: "relative", unit: "day", direction: "previous" },
          last_n_days: { text: `最近${typeof input.amount === "number" || typeof input.amount === "string" ? input.amount : ""}天`, type: "relative", unit: "day", direction: "past" },
          this_week: { text: "本周", type: "relative", unit: "week", direction: "current" },
          last_week: { text: "上周", type: "relative", unit: "week", direction: "previous" },
          absolute: { type: "absolute" }
        };
        return {
          ...input,
          ...legacyMap[input.kind],
          text: input.text ?? legacyMap[input.kind]?.text ?? null
        };
      }
    }
    return value;
  }, z
    .object({
      text: nullableString,
      type: z.enum(["relative", "absolute"]).nullable().default(null),
      amount: nullablePositiveInt,
      unit: timeRangeUnitSchema,
      direction: timeRangeDirectionSchema,
      from: z.string().nullable().default(null),
      to: z.string().nullable().default(null)
    })
    .nullable()
    .default(null));

export const ragRouteSchema = z.object({
  intent: z.string().min(1).default("topic_search"),
  strategy: retrievalStrategySchema,
  topic: z.string().trim().nullable().default(null),
  recordingLimit: z.number().int().min(1).max(5).nullable().default(null),
  recordingRank: z.number().int().min(1).max(10).nullable().default(null),
  timeRange: timeRangeSchema,
  dateRange: z
    .preprocess((value) => {
      if (typeof value === "string") return { from: null, to: null };
      return value;
    }, z.object({
      from: z.string().nullable().default(null),
      to: z.string().nullable().default(null)
    }))
    .nullable()
    .default(null),
  filters: z
    .object({
      recordingIds: z.array(z.string()).default([]),
      speakerProfileIds: z.array(z.string()).default([]),
      personNames: z.array(z.string().trim().min(1)).default([]),
      locations: z.array(z.string().trim().min(1)).default([]),
      targetPersonOnly: z.boolean().default(false)
    })
    .default({ recordingIds: [], speakerProfileIds: [], personNames: [], locations: [], targetPersonOnly: false }),
  needsAnswer: z.boolean().default(true),
  reason: z.string().default("")
});

export type RetrievalStrategy = z.infer<typeof retrievalStrategySchema>;
export type RagRoute = z.infer<typeof ragRouteSchema>;

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function inferStrategy(raw: Record<string, unknown>, query: string): RetrievalStrategy {
  const topic = typeof raw.topic === "string" ? raw.topic.trim() : "";
  const intent = typeof raw.intent === "string" ? raw.intent : "";
  const asksForSummary = /(录音|音频|会议).*(说了什么|讲了什么|总结|主要内容|内容|重点)|分别|逐条|每个录音|每条录音|详细总结/.test(query);
  if (/summary|summar/i.test(intent) || asksForSummary) return "scope_summary";
  return topic ? "chunk_search" : "scope_summary";
}

function normalizeStringArray(value: unknown) {
  if (Array.isArray(value)) return value.map((item) => String(item).trim()).filter(Boolean);
  if (typeof value === "string") return value.split(/[、,，]/).map((item) => item.trim()).filter(Boolean);
  return [];
}

function repairRawRoute(raw: unknown, query: string) {
  const route = isRecord(raw) ? { ...raw } : {};
  if (!route.strategy) route.strategy = inferStrategy(route, query);
  if (!("intent" in route) || !route.intent) route.intent = route.strategy === "scope_summary" ? "scope_summary" : "topic_search";
  if (route.strategy === "scope_summary") {
    route.topic = null;
  } else if (!("topic" in route)) {
    route.topic = query;
  }
  if (!("recordingLimit" in route)) route.recordingLimit = null;
  if (!("recordingRank" in route)) route.recordingRank = null;
  if (!("timeRange" in route)) route.timeRange = null;
  if (!("dateRange" in route)) route.dateRange = null;
  const filters = isRecord(route.filters) ? route.filters : {};
  route.filters = {
    recordingIds: normalizeStringArray(filters.recordingIds ?? route.recordingIds),
    speakerProfileIds: normalizeStringArray(filters.speakerProfileIds ?? route.speakerProfileIds),
    personNames: normalizeStringArray(filters.personNames ?? filters.personName ?? route.personNames ?? route.personName),
    locations: normalizeStringArray(filters.locations ?? filters.location ?? route.locations ?? route.location),
    targetPersonOnly: Boolean(filters.targetPersonOnly ?? route.targetPersonOnly ?? false)
  };
  if (!("needsAnswer" in route)) route.needsAnswer = true;
  if (!("reason" in route)) route.reason = "router output repaired before schema validation";
  return route;
}

export function parseRagRoute(raw: unknown, query: string): RagRoute {
  return ragRouteSchema.parse(repairRawRoute(raw, query));
}

export function fallbackRoute(query: string): RagRoute {
  return {
    intent: "topic_search",
    strategy: "chunk_search",
    topic: query,
    recordingLimit: null,
    recordingRank: null,
    timeRange: null,
    dateRange: null,
    filters: {
      recordingIds: [],
      speakerProfileIds: [],
      personNames: [],
      locations: [],
      targetPersonOnly: false
    },
    needsAnswer: true,
    reason: "router fallback"
  };
}
