export type Author = { family: string; given: string } | { literal: string };

export interface ReferenceData {
  type: 'paper' | 'book' | 'report' | 'article' | 'video' | 'website' | 'software' | 'dataset';
  authors: Author[];
  title: string;
  year: number;
  venue?: string;
  url: string;
  arxiv?: string;
  doi?: string;
  note?: string;
  topics: string[];
}

/** Section headings on the references page, in display order. */
export const TOPIC_LABELS: Record<string, string> = {
  architecture: 'The transformer',
  tokens: 'Tokens and embeddings',
  attention: 'Attention',
  prediction: 'Prediction and sampling',
  training: 'Training and data',
  today: "Today's language models",
  interpretability: 'Looking inside models',
  learning: 'How people learn',
  tools: 'Other explainers and tools',
};

function familyName(author: Author): string {
  return 'literal' in author ? author.literal : author.family;
}

function initials(given: string): string {
  return given
    .split(/\s+/)
    .filter(Boolean)
    .map((part) =>
      part
        .split('-')
        .map((piece) => `${piece.replace(/\.$/, '').charAt(0)}.`)
        .join('-'),
    )
    .join(' ');
}

/** Short in-text label such as "Vaswani et al., 2017". */
export function citationLabel(ref: Pick<ReferenceData, 'authors' | 'year'>): string {
  const names = ref.authors.map(familyName);
  const who =
    names.length === 1
      ? names[0]
      : names.length === 2
        ? `${names[0]} and ${names[1]}`
        : `${names[0]} et al.`;
  return `${who}, ${ref.year}`;
}

/** Full author list such as "Vaswani, A., Shazeer, N., and Parmar, N.". */
export function formatAuthors(authors: readonly Author[]): string {
  const names = authors.map((author) =>
    'literal' in author ? author.literal : `${author.family}, ${initials(author.given)}`,
  );
  if (names.length <= 1) return names.join('');
  if (names.length === 2) return `${names[0]} and ${names[1]}`;
  return `${names.slice(0, -1).join(', ')}, and ${names[names.length - 1]}`;
}
