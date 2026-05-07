import { fallbackRoute, ragRouteSchema, type RagRoute } from "./route-schema";
import { resolveRelativeDateRange } from "./date-range";

function parseChineseCount(query: string) {
  const digitMatch = query.match(/最近\s*(\d+)\s*(个|条|段)?\s*(音频|录音|会议)/);
  if (digitMatch) return Math.max(1, Math.min(5, Number(digitMatch[1])));
  if (/最近\s*(两|二)\s*(个|条|段)?\s*(音频|录音|会议)/.test(query)) return 2;
  if (/最近\s*(三)\s*(个|条|段)?\s*(音频|录音|会议)/.test(query)) return 3;
  if (/最近\s*(四)\s*(个|条|段)?\s*(音频|录音|会议)/.test(query)) return 4;
  if (/最近\s*(五)\s*(个|条|段)?\s*(音频|录音|会议)/.test(query)) return 5;
  if (/最近.*(音频|录音|会议)/.test(query)) return 2;
  return null;
}

function parseRecordingRank(query: string) {
  if (/(倒数)?第\s*2\s*(个|条|段)?\s*(音频|录音|会议)/.test(query)) return 2;
  if (/(倒数)?第二\s*(个|条|段)?\s*(音频|录音|会议)/.test(query)) return 2;
  if (/(倒数)?第\s*3\s*(个|条|段)?\s*(音频|录音|会议)/.test(query)) return 3;
  if (/(倒数)?第三\s*(个|条|段)?\s*(音频|录音|会议)/.test(query)) return 3;
  if (/(上一个|上一条|前一个|前一条)\s*(音频|录音|会议)/.test(query)) return 2;
  if (/(最新|最近|最后)\s*(一个|一条)?\s*(音频|录音|会议)/.test(query)) return 1;
  return null;
}

export async function routeQueryWithRules(query: string): Promise<RagRoute> {
  const recordingRank = parseRecordingRank(query);
  if (recordingRank !== null && /(说了什么|讲了什么|总结|主要内容|内容|重点)/.test(query)) {
    return ragRouteSchema.parse({
      intent: "ordinal_recording_summary",
      strategy: "scope_summary",
      topic: null,
      recordingLimit: null,
      recordingRank,
      timeRange: null,
      dateRange: null,
      filters: { recordingIds: [], speakerProfileIds: [], personNames: [], locations: [], targetPersonOnly: false },
      needsAnswer: true,
      reason: "query asks for one completed recording by created_at rank"
    });
  }

  const recentCount = parseChineseCount(query);
  if (recentCount !== null && /(说了什么|讲了什么|总结|主要内容|内容|重点)/.test(query)) {
    return ragRouteSchema.parse({
      intent: "scope_summary",
      strategy: "scope_summary",
      topic: null,
      recordingLimit: recentCount,
      recordingRank: null,
      timeRange: null,
      dateRange: null,
      filters: { recordingIds: [], speakerProfileIds: [], personNames: [], locations: [], targetPersonOnly: false },
      needsAnswer: true,
      reason: "query asks for recent recording summary"
    });
  }

  const dateRange = resolveRelativeDateRange(query);
  if (dateRange && /(关于|有关|围绕|提到)/.test(query)) {
    const topic = query
      .replace(/前天|昨天|今天|上周|本周|这周/g, " ")
      .replace(/(关于|有关|围绕|提到)/g, " ")
      .replace(/(都)?(说了什么|讲了什么|说了哪些|讲了哪些|说啥了|总结|主要内容|重点)/g, " ")
      .replace(/[，。！？,.!?]/g, " ")
      .replace(/\s+/g, " ")
      .trim();
    if (topic) {
      return ragRouteSchema.parse({
        intent: "scoped_topic_search",
        strategy: "chunk_search",
        topic,
        recordingLimit: null,
        recordingRank: null,
        timeRange: null,
        dateRange,
        filters: { recordingIds: [], speakerProfileIds: [], personNames: [], locations: [], targetPersonOnly: false },
        needsAnswer: true,
        reason: "query combines a relative date range with a topic"
      });
    }
  }

  if (dateRange && /(录音|音频|会议).*(总结|说了什么|主要内容|重点)/.test(query)) {
    return ragRouteSchema.parse({
      intent: "scope_summary",
      strategy: "scope_summary",
      topic: null,
      recordingLimit: null,
      recordingRank: null,
      timeRange: null,
      dateRange,
      filters: { recordingIds: [], speakerProfileIds: [], personNames: [], locations: [], targetPersonOnly: false },
      needsAnswer: true,
      reason: "query asks for date range recording summary"
    });
  }

  return fallbackRoute(query);
}
