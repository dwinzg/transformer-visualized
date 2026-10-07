import { stepId } from './stepId';

/** One thing search can find: a chapter, a step, a term, a map part, a question or a source. */
export interface SearchEntry {
  kind: 'Chapter' | 'Step' | 'Glossary' | 'Map' | 'Answer' | 'Source' | 'Page';
  title: string;
  /** Where it lives, such as the chapter of a step. */
  context?: string;
  /** Site-relative link, without the base. */
  href: string;
  text: string;
}

const ENTITIES: Record<string, string> = { quot: '"', amp: '&', lt: '<', gt: '>', apos: "'" };

/** Drops each `{...}`, counting nested braces, so JSX expressions and their arrows go too. */
function dropBraces(text: string): string {
  let out = '';
  let depth = 0;
  for (const ch of text) {
    if (ch === '{') depth++;
    else if (ch === '}') depth = Math.max(0, depth - 1);
    else if (depth === 0) out += ch;
  }
  return out;
}

/** The words a reader sees in a piece of MDX: no code, math, tags or Markdown marks. */
export function plainText(mdx: string): string {
  let text = mdx
    .replace(/^```[\s\S]*?^```/gm, ' ')
    // Inline math keeps its letters and numbers, so "n = 6" still reads. Display math goes.
    .replace(/<Tex\b(?![^>]*\bdisplay\b)[^>]*?\btex="([^"]*)"[^>]*\/>/g, (_, tex: string) =>
      tex.replace(/\\[a-zA-Z]+|[{}\\]/g, ' '),
    )
    .replace(/<Tex\b[^]*?\/>/g, ' ')
    .replace(/<Tex\b[^]*?<\/Tex>/g, ' ');
  text = dropBraces(text)
    .replace(/<\/?[A-Za-z][^>]*>/g, ' ')
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/^#+\s*/gm, '')
    .replace(/^\s*\|?[\s|:-]+\|?\s*$/gm, ' ')
    .replace(/[|*_`]/g, ' ')
    .replace(/&(quot|amp|lt|gt|apos);/g, (_, name: string) => ENTITIES[name]);
  return (
    text
      .replace(/\s+/g, ' ')
      // A dropped citation or tag can leave "( )" or a space before a comma behind.
      .replace(/\(\s*\)/g, '')
      .replace(/\s+([,.;:!?)])/g, '$1')
      .replace(/\s+/g, ' ')
      .trim()
  );
}

/** Splits a chapter body into its steps, with the same ids the page gives them. */
export function stepsOf(body: string): { id: string; title: string; text: string }[] {
  const taken = new Set<string>();
  // Code samples never hold a real step, so they are blanked first, keeping every position.
  const prose = body.replace(/^```[\s\S]*?^```/gm, (code) => ' '.repeat(code.length));
  const starts = [...prose.matchAll(/<Step\b[^>]*?\btitle="([^"]*)"/g)];
  return starts.map((match, i) => {
    const title = match[1].replace(/&(quot|amp|lt|gt|apos);/g, (_, n: string) => ENTITIES[n]);
    const id = stepId(title, taken);
    taken.add(id);
    const end = starts[i + 1]?.index ?? body.length;
    return { id, title, text: plainText(body.slice(match.index, end)) };
  });
}
