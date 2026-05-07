import type { RagRoute } from "./route-schema";

const TIME_ZONE = "Asia/Shanghai";

function zonedDateParts(date: Date) {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: TIME_ZONE,
    year: "numeric",
    month: "2-digit",
    day: "2-digit"
  }).formatToParts(date);
  const value = (type: string) => parts.find((part) => part.type === type)?.value ?? "";
  return {
    year: Number(value("year")),
    month: Number(value("month")),
    day: Number(value("day"))
  };
}

function zonedDateTimeParts(date: Date) {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: TIME_ZONE,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false
  }).formatToParts(date);
  const value = (type: string) => parts.find((part) => part.type === type)?.value ?? "";
  return {
    year: value("year"),
    month: value("month"),
    day: value("day"),
    hour: value("hour") === "24" ? "00" : value("hour"),
    minute: value("minute"),
    second: value("second")
  };
}

function toShanghaiIso(date: Date) {
  const parts = zonedDateTimeParts(date);
  return `${parts.year}-${parts.month}-${parts.day}T${parts.hour}:${parts.minute}:${parts.second}.000+08:00`;
}

function utcInstantForShanghaiLocal(year: number, month: number, day: number, hour = 0, minute = 0, second = 0, millisecond = 0) {
  return new Date(Date.UTC(year, month - 1, day, hour - 8, minute, second, millisecond));
}

function addDays(date: Date, days: number) {
  const next = new Date(date);
  next.setUTCDate(next.getUTCDate() + days);
  return next;
}

function todayLocalMidnight() {
  const today = zonedDateParts(new Date());
  return utcInstantForShanghaiLocal(today.year, today.month, today.day);
}

function dayRange(offsetDays: number) {
  return dayRangeSpan(offsetDays, 1);
}

function dayRangeSpan(offsetDays: number, days: number) {
  const from = addDays(todayLocalMidnight(), offsetDays);
  return { from: toShanghaiIso(from), to: toShanghaiIso(addDays(from, Math.max(1, days))) };
}

function rollingDaysRange(days: number) {
  const tomorrow = addDays(todayLocalMidnight(), 1);
  return {
    from: toShanghaiIso(addDays(tomorrow, -Math.max(1, days))),
    to: toShanghaiIso(tomorrow)
  };
}

function weekRange(offsetWeeks: number) {
  return weekRangeSpan(offsetWeeks, 1);
}

function weekRangeSpan(offsetWeeks: number, weeks: number) {
  const localMidnight = todayLocalMidnight();
  const localDay = new Intl.DateTimeFormat("en-US", { timeZone: TIME_ZONE, weekday: "short" }).format(new Date());
  const weekdayOffset: Record<string, number> = { Mon: 0, Tue: 1, Wed: 2, Thu: 3, Fri: 4, Sat: 5, Sun: 6 };
  const daysFromMonday = weekdayOffset[localDay] ?? 0;
  const monday = addDays(localMidnight, -daysFromMonday + offsetWeeks * 7);
  return { from: toShanghaiIso(monday), to: toShanghaiIso(addDays(monday, Math.max(1, weeks) * 7)) };
}

function addMonthsParts(year: number, month: number, deltaMonths: number) {
  const absoluteMonth = year * 12 + (month - 1) + deltaMonths;
  return {
    year: Math.floor(absoluteMonth / 12),
    month: (absoluteMonth % 12) + 1
  };
}

function monthRange(offsetMonths: number) {
  return monthRangeSpan(offsetMonths, 1);
}

function monthRangeSpan(offsetMonths: number, months: number) {
  const today = zonedDateParts(new Date());
  const fromParts = addMonthsParts(today.year, today.month, offsetMonths);
  const toParts = addMonthsParts(fromParts.year, fromParts.month, Math.max(1, months));
  const from = utcInstantForShanghaiLocal(fromParts.year, fromParts.month, 1);
  const to = utcInstantForShanghaiLocal(toParts.year, toParts.month, 1);
  return { from: toShanghaiIso(from), to: toShanghaiIso(to) };
}

function yearRange(offsetYears: number) {
  return yearRangeSpan(offsetYears, 1);
}

function yearRangeSpan(offsetYears: number, years: number) {
  const today = zonedDateParts(new Date());
  const from = utcInstantForShanghaiLocal(today.year + offsetYears, 1, 1);
  const to = utcInstantForShanghaiLocal(today.year + offsetYears + Math.max(1, years), 1, 1);
  return { from: toShanghaiIso(from), to: toShanghaiIso(to) };
}

function rollingCalendarRange(unit: "month" | "year", amount: number) {
  const today = zonedDateParts(new Date());
  const to = addDays(todayLocalMidnight(), 1);
  if (unit === "month") {
    const fromParts = addMonthsParts(today.year, today.month, 1 - Math.max(1, amount));
    return { from: toShanghaiIso(utcInstantForShanghaiLocal(fromParts.year, fromParts.month, 1)), to: toShanghaiIso(to) };
  }
  return { from: toShanghaiIso(utcInstantForShanghaiLocal(today.year + 1 - Math.max(1, amount), 1, 1)), to: toShanghaiIso(to) };
}

function parseChineseNumber(text: string) {
  if (/^\d+$/.test(text)) return Number(text);
  const map: Record<string, number> = {
    一: 1,
    二: 2,
    两: 2,
    三: 3,
    四: 4,
    五: 5,
    六: 6,
    七: 7,
    八: 8,
    九: 9,
    十: 10
  };
  return map[text] ?? null;
}

function unitFromChinese(text: string): "day" | "week" | "month" | "year" | null {
  if (text === "天" || text === "日") return "day";
  if (text === "周" || text === "星期") return "week";
  if (text === "月" || text === "个月") return "month";
  if (text === "年") return "year";
  return null;
}

function rollingRangeByUnit(unit: "day" | "week" | "month" | "year", amount: number) {
  if (unit === "day") return rollingDaysRange(amount);
  if (unit === "week") return rollingDaysRange(amount * 7);
  if (unit === "month") return rollingCalendarRange("month", amount);
  return rollingCalendarRange("year", amount);
}

function currentRangeByUnit(unit: "day" | "week" | "month" | "year") {
  if (unit === "day") return dayRange(0);
  if (unit === "week") return weekRange(0);
  if (unit === "month") return monthRange(0);
  return yearRange(0);
}

function previousRangeByUnit(unit: "day" | "week" | "month" | "year", amount: number) {
  if (unit === "day") return dayRangeSpan(-amount, amount);
  if (unit === "week") return weekRangeSpan(-amount, amount);
  if (unit === "month") return monthRangeSpan(-amount, amount);
  return yearRangeSpan(-amount, amount);
}

export function resolveRelativeDateRange(query: string) {
  const rollingMatch = query.match(/(?:最近|近|过去|这)\s*(\d+|一|二|两|三|四|五|六|七|八|九|十)\s*(天|日|周|星期|个月|月|年)/);
  if (rollingMatch) {
    const amount = parseChineseNumber(rollingMatch[1]);
    const unit = unitFromChinese(rollingMatch[2]);
    if (amount && unit) return rollingRangeByUnit(unit, amount);
  }
  if (/这两天|近两天|最近两天|过去两天/.test(query)) return rollingDaysRange(2);
  if (/这三天|近三天|最近三天|过去三天/.test(query)) return rollingDaysRange(3);
  if (/前天/.test(query)) return dayRange(-2);
  if (/昨天/.test(query)) return dayRange(-1);
  if (/今天/.test(query)) return dayRange(0);
  if (/上个月|上月/.test(query)) return monthRange(-1);
  if (/本月|这个月|这月/.test(query)) return monthRange(0);
  if (/去年/.test(query)) return yearRange(-1);
  if (/今年/.test(query)) return yearRange(0);
  if (/上周/.test(query)) return weekRange(-1);
  if (/本周|这周/.test(query)) return weekRange(0);
  return null;
}

function resolveStructuredTimeRange(route: RagRoute) {
  const timeRange = route.timeRange;
  if (!timeRange) return null;
  if (timeRange.from && timeRange.to) return { from: timeRange.from, to: timeRange.to };
  const textRange = timeRange.text ? resolveRelativeDateRange(timeRange.text) : null;
  if (textRange) return textRange;
  if (!timeRange.unit || !timeRange.direction) return null;
  const amount = timeRange.amount ?? 1;
  if (timeRange.direction === "current") return currentRangeByUnit(timeRange.unit);
  if (timeRange.direction === "previous") return previousRangeByUnit(timeRange.unit, amount);
  if (timeRange.direction === "past") return rollingRangeByUnit(timeRange.unit, amount);
  return null;
}

export function normalizeRouteDateRange(route: RagRoute, query: string): RagRoute {
  const structuredRange = resolveStructuredTimeRange(route);
  const queryRelativeRange = resolveRelativeDateRange(query);
  const routeDateRange = route.dateRange?.from && route.dateRange?.to ? route.dateRange : null;
  const resolved = structuredRange ?? queryRelativeRange ?? routeDateRange;
  if (!resolved) return route;
  return {
    ...route,
    dateRange: resolved
  };
}
