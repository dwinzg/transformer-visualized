/** Characters with no look of their own, drawn and spoken as a mark instead. */
const MARKS: Record<string, [shown: string, spoken: string]> = {
  ' ': ['\u00b7', 'space '],
  '\n': ['\u21b5', 'new line '],
  '\t': ['\u21e5', 'tab '],
};

/** A token's text with spaces, new lines and tabs drawn as marks, since they are part of it. */
export const shownToken = (text: string) => text.replace(/[ \n\t]/g, (char) => MARKS[char][0]);

/** A token's text for a screen reader, so " Lily" is "space Lily" and a byte is "byte C3". */
export const spokenToken = (text: string) =>
  /^<0x[0-9A-F]{2}>$/.test(text)
    ? `byte ${text.slice(3, 5)}`
    : text.replace(/[ \n\t]/g, (char) => MARKS[char][1]).trim();
