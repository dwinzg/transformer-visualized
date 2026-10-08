import type { PartId, View } from './concepts';

export interface Block {
  id: string;
  part: PartId;
  label: string | Record<View, string>;
  /** The name a screen reader hears, where the label alone repeats another block's. */
  name?: string | Record<View, string>;
  /** Replaces the part's note for this block, where the block differs from the part. */
  notes?: Partial<Record<View, string>>;
  views: readonly View[];
  x: number;
  /** One y for both views, or one per view where GPT-2 moves the block. */
  y: number | Record<View, number>;
  w: number;
  h: number;
}

export const BOX = { width: 640, height: 800 } as const;
const BOTH: readonly View[] = ['original', 'gpt2', 'llama'];
const ORIGINAL: readonly View[] = ['original'];
const GPT2: readonly View[] = ['gpt2', 'llama'];
// Llama has no position table, since position enters inside attention.
const NO_LLAMA: readonly View[] = ['original', 'gpt2'];
const D = { x: 340, w: 220 };
const E = { x: 40, w: 220 };

export const BLOCKS: readonly Block[] = [
  {
    id: 'enc-input',
    part: 'input',
    label: 'Inputs',
    notes: { original: 'In the paper this side read the sentence to translate.' },
    views: ORIGINAL,
    ...E,
    y: 746,
    h: 40,
  },
  {
    id: 'enc-embedding',
    part: 'embedding',
    label: 'Input embedding',
    views: ORIGINAL,
    ...E,
    y: 686,
    h: 44,
  },
  {
    id: 'enc-position',
    part: 'position',
    label: 'Positional encoding',
    name: 'Encoder positional encoding',
    views: ORIGINAL,
    ...E,
    y: 630,
    h: 40,
  },
  {
    id: 'enc-attn',
    part: 'enc-attn',
    label: 'Multi-head attention',
    name: 'Encoder multi-head attention',
    views: ORIGINAL,
    ...E,
    y: 546,
    h: 44,
  },
  {
    id: 'enc-add-norm-1',
    part: 'add-norm',
    label: 'Add and norm',
    name: 'Encoder add and norm, after attention',
    views: ORIGINAL,
    ...E,
    y: 500,
    h: 32,
  },
  {
    id: 'enc-ffn',
    part: 'ffn',
    label: 'Feed forward',
    name: 'Encoder feed forward',
    views: ORIGINAL,
    ...E,
    y: 436,
    h: 44,
  },
  {
    id: 'enc-add-norm-2',
    part: 'add-norm',
    label: 'Add and norm',
    name: 'Encoder add and norm, after feed forward',
    views: ORIGINAL,
    ...E,
    y: 396,
    h: 32,
  },
  {
    id: 'input',
    part: 'input',
    label: { original: 'Outputs (shifted right)', gpt2: 'Inputs', llama: 'Inputs' },
    views: BOTH,
    ...D,
    y: 746,
    h: 40,
  },
  {
    id: 'embedding',
    part: 'embedding',
    label: { original: 'Output embedding', gpt2: 'Token embedding', llama: 'Token embedding' },
    views: BOTH,
    ...D,
    y: 686,
    h: 44,
  },
  {
    id: 'position',
    part: 'position',
    label: {
      original: 'Positional encoding',
      gpt2: 'Position embedding',
      llama: 'Position embedding',
    },
    views: NO_LLAMA,
    ...D,
    y: 630,
    h: 40,
  },
  {
    id: 'masked-attn',
    part: 'masked-attn',
    label: {
      original: 'Masked multi-head attention',
      gpt2: 'Masked multi-head attention',
      llama: 'Grouped-query attention, RoPE',
    },
    views: BOTH,
    ...D,
    y: { original: 546, gpt2: 500, llama: 500 },
    h: 44,
  },
  {
    id: 'add-norm-1',
    part: 'add-norm',
    label: { original: 'Add and norm', gpt2: 'Norm', llama: 'RMSNorm' },
    name: {
      original: 'Add and norm, after masked attention',
      gpt2: 'Norm, before attention',
      llama: 'RMSNorm, before attention',
    },
    views: BOTH,
    ...D,
    y: { original: 500, gpt2: 558, llama: 558 },
    h: 32,
  },
  {
    id: 'cross-attn',
    part: 'cross-attn',
    label: 'Multi-head attention',
    views: ORIGINAL,
    ...D,
    y: 436,
    h: 44,
  },
  {
    id: 'add-norm-2',
    part: 'add-norm',
    label: 'Add and norm',
    name: 'Add and norm, after cross-attention',
    views: ORIGINAL,
    ...D,
    y: 396,
    h: 32,
  },
  {
    id: 'ffn',
    part: 'ffn',
    label: { original: 'Feed forward', gpt2: 'Feed forward', llama: 'SwiGLU feed forward' },
    views: BOTH,
    ...D,
    y: { original: 316, gpt2: 270, llama: 270 },
    h: 44,
  },
  {
    id: 'add-norm-3',
    part: 'add-norm',
    label: { original: 'Add and norm', gpt2: 'Norm', llama: 'RMSNorm' },
    name: {
      original: 'Add and norm, after feed forward',
      gpt2: 'Norm, before feed forward',
      llama: 'RMSNorm, before feed forward',
    },
    views: BOTH,
    ...D,
    y: { original: 270, gpt2: 328, llama: 328 },
    h: 32,
  },
  {
    id: 'stack',
    part: 'stack',
    label: 'N×',
    name: 'Repeated blocks, N×',
    views: BOTH,
    x: 576,
    y: 400,
    w: 48,
    h: 44,
  },
  {
    id: 'final-norm',
    part: 'final-norm',
    label: { original: 'Final norm', gpt2: 'Final norm', llama: 'Final RMSNorm' },
    views: GPT2,
    ...D,
    y: 206,
    h: 32,
  },
  { id: 'linear', part: 'linear', label: 'Linear', views: BOTH, ...D, y: 150, h: 36 },
  { id: 'softmax', part: 'softmax', label: 'Softmax', views: BOTH, ...D, y: 96, h: 36 },
  { id: 'output', part: 'output', label: 'Output probabilities', views: BOTH, ...D, y: 40, h: 36 },
];

/** The frames drawn around each repeated block, decoder then encoder. */
export const FRAMES = [
  { views: BOTH, x: 324, y: 256, w: 252, h: 346 },
  { views: ORIGINAL, x: 24, y: 384, w: 252, h: 218 },
] as const;

const DECODER_INPUT = ['input', 'embedding', 'position'];
export const TOUR: Record<View, readonly string[]> = {
  original: [
    'enc-input',
    'enc-embedding',
    'enc-position',
    'enc-attn',
    'enc-add-norm-1',
    'enc-ffn',
    'enc-add-norm-2',
    ...DECODER_INPUT,
    'masked-attn',
    'add-norm-1',
    'cross-attn',
    'add-norm-2',
    'ffn',
    'add-norm-3',
    'stack',
    'linear',
    'softmax',
    'output',
  ],
  // GPT-2 norms before each sublayer, so each norm comes first.
  gpt2: [
    ...DECODER_INPUT,
    'add-norm-1',
    'masked-attn',
    'add-norm-3',
    'ffn',
    'stack',
    'final-norm',
    'linear',
    'softmax',
    'output',
  ],
  // The same as GPT-2, without the position table.
  llama: [
    'input',
    'embedding',
    'add-norm-1',
    'masked-attn',
    'add-norm-3',
    'ffn',
    'stack',
    'final-norm',
    'linear',
    'softmax',
    'output',
  ],
};

export function blockLabel(block: Block, view: View): string {
  return typeof block.label === 'string' ? block.label : block.label[view];
}

/** The accessible name, which always contains the visible label. */
export function blockName(block: Block, view: View): string {
  const name = block.name ?? block.label;
  return typeof name === 'string' ? name : name[view];
}

export function blockY(block: Block, view: View): number {
  return typeof block.y === 'number' ? block.y : block.y[view];
}

/**
 * GPT-2's residual adds: the line leaves below each norm, skips around it and its sublayer, and
 * joins at a plus just above the sublayer.
 */
export const RESIDUALS = [
  { from: 610, to: 490 },
  { from: 375, to: 263 },
] as const;

export function visibleBlocks(view: View): Block[] {
  return BLOCKS.filter((b) => b.views.includes(view));
}

/** The block a link to this part lands on, the first one in the tour for that view. */
export function firstBlockFor(part: PartId, view: View): Block | undefined {
  const id = TOUR[view].find((blockId) => BLOCKS.find((b) => b.id === blockId)?.part === part);
  return BLOCKS.find((b) => b.id === id);
}
