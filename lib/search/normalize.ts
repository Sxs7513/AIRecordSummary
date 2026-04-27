export function normalizeSearchText(text: string) {
  return text.trim().replace(/\s+/g, " ").toLowerCase();
}

export function vectorLiteral(values: number[]) {
  return `[${values.map((value) => {
    if (!Number.isFinite(value)) return "0";
    return String(value);
  }).join(",")}]`;
}
