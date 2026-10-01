/**
 * The typography the tiny model's training data was cleaned of, with its ASCII equivalent.
 * Kept in step with `_ASCII_EQUIVALENTS` in model/tv_model/data.py by a test.
 */
export const ASCII_EQUIVALENTS: Readonly<Record<string, string>> = {
  '\u2018': "'",
  '\u2019': "'",
  '\u201c': '"',
  '\u201d': '"',
  '\u2013': '-',
  '\u2014': '-',
  '\u2026': '...',
  '\u00a0': ' ',
};

const PATTERN = new RegExp(`[${Object.keys(ASCII_EQUIVALENTS).join('')}]`, 'g');

/** Maps curly quotes, dashes, the ellipsis and no-break spaces to ASCII. Does not trim. */
export function normalizeText(text: string): string {
  return text.replace(PATTERN, (char) => ASCII_EQUIVALENTS[char]);
}

/** Distinct characters outside ASCII that remain after `normalizeText`, in first-seen order. */
export function unsupportedCharacters(text: string): string[] {
  const found = new Set<string>();
  for (const char of normalizeText(text)) {
    if ((char.codePointAt(0) ?? 0) > 0x7f) found.add(char);
  }
  return [...found];
}
