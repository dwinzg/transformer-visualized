const DECIMAL = /^-?(\d+([.,]\d*)?|[.,]\d+)$/;

function parseDecimal(text: string): number | null {
  if (!DECIMAL.test(text)) return null;
  const value = Number(text.replace(',', '.'));
  return Number.isFinite(value) ? value : null;
}

/** Reads a typed answer. Accepts decimals (either decimal mark), percentages and a/b fractions. */
export function parseNumber(text: string): number | null {
  const trimmed = text.trim();
  if (trimmed.endsWith('%')) {
    const value = parseDecimal(trimmed.slice(0, -1).trim());
    return value === null ? null : value / 100;
  }
  const parts = trimmed.split('/');
  if (parts.length === 2) {
    const [numerator, denominator] = parts.map((part) => parseDecimal(part.trim()));
    if (numerator === null || denominator === null || denominator === 0) return null;
    return numerator / denominator;
  }
  return parseDecimal(trimmed);
}

export function isCorrect(given: number, answer: number, tolerance: number): boolean {
  return Math.abs(given - answer) <= tolerance;
}
