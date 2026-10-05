/**
 * Every part of the transformer, written once. The architecture map, the chapters, the glossary
 * and the introduction's pipeline figure all build their links from this list.
 */
export type View = 'original' | 'gpt2';
export const VIEWS: readonly View[] = ['original', 'gpt2'];

export type PartId =
  | 'input'
  | 'embedding'
  | 'position'
  | 'masked-attn'
  | 'enc-attn'
  | 'add-norm'
  | 'cross-attn'
  | 'ffn'
  | 'stack'
  | 'final-norm'
  | 'linear'
  | 'softmax'
  | 'output';

export type ChapterId = 'tokens' | 'embeddings' | 'attention' | 'prediction';

export interface Part {
  id: PartId;
  title: string;
  color: 'neutral' | 'position' | 'attention' | 'residual' | 'output';
  /** One or two plain sentences, shown at every level. */
  story: string;
  /** Extra sentence for one view, where the paper and GPT-2 differ. */
  notes?: Partial<Record<View, string>>;
  /** Shapes in the tiny model, shown from the Numbers level up. */
  numbers: string;
  /** TeX, shown from the Formula level up. One string, or one per view. */
  formula?: string | Record<View, string>;
  /** One line of engine code, shown at the Code level. */
  code?: string;
  glossary?: string;
  chapter?: ChapterId;
  /** Step anchor inside the chapter, once the chapter exists. */
  step?: string;
}

const ATTENTION = String.raw`\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V`;

export const PARTS: Record<PartId, Part> = {
  input: {
    id: 'input',
    title: 'Inputs',
    color: 'neutral',
    story: 'Your text, split into tokens. Each token is an id number from the vocabulary.',
    notes: {
      original:
        'In the paper the right side read the translation so far, shifted one place right so each position predicts the next word.',
    },
    numbers: 'The sentence "Lily wanted to play with her" is 6 token ids, each from 0 to 4,095.',
    code: "const ids = tokenizer.encode('Lily wanted to play with her');",
    glossary: 'token',
    chapter: 'tokens',
    step: 'step-text-becomes-tokens',
  },
  embedding: {
    id: 'embedding',
    title: 'Embedding',
    color: 'neutral',
    story:
      "Each id picks one row from a table of learned numbers. That row is the token's starting list of numbers.",
    numbers: 'The table has 4,096 rows of 128 numbers. The sentence becomes 6 rows of 128.',
    formula: String.raw`\mathbf{x}_i = W_E[t_i] \in \mathbb{R}^{128}`,
    code: 'trace.tokenEmbeddings // 6 × 128',
    glossary: 'embedding',
    chapter: 'embeddings',
    step: 'step-each-token-becomes-a-list-of-numbers',
  },
  position: {
    id: 'position',
    title: 'Position',
    color: 'position',
    story:
      "Each position also gets a list of numbers, added to the token's list. That is how the model knows the order.",
    notes: {
      original: 'The paper used fixed sine and cosine waves.',
      gpt2: 'GPT-2 learns a table of positions instead.',
    },
    numbers:
      'A table of 128 positions by 128 numbers. Row 0 goes to the first token, row 1 to the second.',
    formula: {
      original: String.raw`PE_{(pos,\,2i)} = \sin\!\left(pos / 10000^{2i/d}\right)`,
      gpt2: String.raw`\mathbf{h}_i = W_E[t_i] + W_P[i-1]`,
    },
    code: 'trace.embeddings // tokenEmbeddings + positionEmbeddings',
    glossary: 'position-embedding',
    chapter: 'embeddings',
    step: 'step-each-place-gets-numbers-too',
  },
  'masked-attn': {
    id: 'masked-attn',
    title: 'Masked attention',
    color: 'attention',
    story:
      'Each token looks at itself and the tokens before it, and pulls in what it needs. Masked means it cannot look ahead.',
    numbers:
      'Our model has 4 heads with 32 numbers each. Each head makes a 6 by 6 grid of weights.',
    formula: String.raw`\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V, \quad M_{ij} = -\infty \text{ when } j > i`,
    code: 'trace.layers[0].heads[0].weights // 6 × 6, each row sums to 1',
    glossary: 'attention',
    chapter: 'attention',
    step: 'step-words-need-context',
  },
  'enc-attn': {
    id: 'enc-attn',
    title: 'Encoder attention',
    color: 'attention',
    story:
      'In the encoder, each word of the source sentence can look at every other word, before or after it.',
    numbers: 'The paper used 8 heads of 64 numbers each. Our model has no encoder.',
    formula: ATTENTION,
    glossary: 'attention',
    chapter: 'attention',
  },
  'add-norm': {
    id: 'add-norm',
    title: 'Add and norm',
    color: 'residual',
    story:
      "Add keeps what came in and puts the new result on top, so nothing is lost. Norm rescales each token's numbers to a steady range.",
    notes: {
      gpt2: 'GPT-2 moves the norm to the start of each step, before attention and before the feed-forward layer.',
    },
    numbers: "Each token's 128 numbers are handled on their own.",
    formula: {
      original: String.raw`\mathrm{LayerNorm}\big(x + \mathrm{Sublayer}(x)\big)`,
      gpt2: String.raw`x + \mathrm{Sublayer}\big(\mathrm{LayerNorm}(x)\big)`,
    },
    code: 'trace.layers[0].ln1.out // 6 × 128, normed before attention',
  },
  'cross-attn': {
    id: 'cross-attn',
    title: 'Cross-attention',
    color: 'attention',
    story:
      'Here the translation side looks at the encoded source sentence. GPT-2 has no source sentence, so it drops this step.',
    numbers: 'Not in our model.',
    formula: String.raw`\mathrm{softmax}\!\left(\frac{Q_{\text{dec}} K_{\text{enc}}^\top}{\sqrt{d_k}}\right) V_{\text{enc}}`,
    chapter: 'attention',
  },
  ffn: {
    id: 'ffn',
    title: 'Feed forward',
    color: 'neutral',
    story:
      "Each token's numbers go through a small network on their own, without looking at other tokens.",
    notes: { original: 'The paper used ReLU and grew 512 numbers to 2,048.' },
    numbers: '128 numbers grow to 512, pass through GELU, and shrink back to 128.',
    formula: {
      original: String.raw`\max(0,\, xW_1 + b_1)\,W_2 + b_2`,
      gpt2: String.raw`\mathrm{GELU}(xW_1 + b_1)\,W_2 + b_2`,
    },
    code: 'trace.layers[0].mlpOut // 6 × 128',
  },
  stack: {
    id: 'stack',
    title: 'Repeated blocks',
    color: 'residual',
    story:
      'Everything inside the frame is one block. The model repeats it, and each block refines the numbers further.',
    numbers: 'Our model stacks 4 blocks. The paper used 6, and GPT-2 small uses 12.',
    formula: String.raw`h^{(\ell+1)} = \mathrm{Block}_\ell\big(h^{(\ell)}\big)`,
    code: 'trace.layers.length // 4',
  },
  'final-norm': {
    id: 'final-norm',
    title: 'Final norm',
    color: 'residual',
    story: 'GPT-2 adds one more norm after the last block, before the scores are made.',
    numbers: 'The 6 rows of 128 numbers, rescaled.',
    formula: String.raw`\mathbf{u} = \mathrm{LayerNorm}\big(h^{(L)}\big)`,
    code: 'trace.lnFinal.out // 6 × 128',
  },
  linear: {
    id: 'linear',
    title: 'Linear',
    color: 'output',
    story:
      "Turns each token's numbers into one score for every token in the vocabulary. It reuses the embedding table.",
    numbers: '4,096 scores per position. Only the last position is used to guess the next token.',
    formula: String.raw`\mathbf{z} = \mathbf{u}\,W_E^\top`,
    code: 'trace.logits // 6 × 4,096',
    glossary: 'logit',
    chapter: 'prediction',
    step: 'step-one-score-for-every-token',
  },
  softmax: {
    id: 'softmax',
    title: 'Softmax',
    color: 'output',
    story: 'Turns the scores into chances between 0 and 1 that add up to 1.',
    numbers: '4,096 chances, one for each possible next token.',
    formula: String.raw`p_t = \frac{e^{z_t}}{\sum_s e^{z_s}}`,
    code: 'probabilities(rowView(trace.logits, ids.length - 1), 1)',
    glossary: 'softmax',
    chapter: 'prediction',
    step: 'step-scores-become-chances',
  },
  output: {
    id: 'output',
    title: 'Output',
    color: 'output',
    story:
      "The model's guess, a chance for every possible next token. Pick one, add it to the text, and run again.",
    numbers: 'One chance for each of the 4,096 tokens.',
    formula: String.raw`P(t_{T+1} \mid t_1, \dots, t_T)`,
    code: 'sample(probs, createRng(1)) // the next token id',
    glossary: 'sampling',
    chapter: 'prediction',
    step: 'step-greedy-top-k-and-top-p',
  },
};

export function isPartId(value: unknown): value is PartId {
  return typeof value === 'string' && Object.hasOwn(PARTS, value);
}

export function formulaFor(part: Part, view: View): string | undefined {
  return typeof part.formula === 'string' ? part.formula : part.formula?.[view];
}

/**
 * The map page with one part picked, as a path without the site base. Links come from lessons
 * about our GPT-2 style model, so they open that view. Parts it lacks fall back to the paper's.
 */
export function mapPath(part?: PartId): string {
  return part ? `learn/architecture/?part=${part}&view=gpt2` : 'learn/architecture/';
}
