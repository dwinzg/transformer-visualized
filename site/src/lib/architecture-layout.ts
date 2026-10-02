import type { PartId, View } from './concepts';

export interface Block {
  id: string;
  part: PartId;
  label: string | Record<View, string>;
  views: readonly View[];
  x: number;
  y: number;
  w: number;
  h: number;
}

export const BOX = { width: 640, height: 800 } as const;
const BOTH: readonly View[] = ['original', 'gpt2'];
const ORIGINAL: readonly View[] = ['original'];
const GPT2: readonly View[] = ['gpt2'];
const D = { x: 340, w: 220 };
const E = { x: 40, w: 220 };

export const BLOCKS: readonly Block[] = [
  { id: 'enc-input', part: 'input', label: 'Inputs', views: ORIGINAL, ...E, y: 746, h: 40 },
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
    views: ORIGINAL,
    ...E,
    y: 630,
    h: 40,
  },
  {
    id: 'enc-attn',
    part: 'enc-attn',
    label: 'Multi-head attention',
    views: ORIGINAL,
    ...E,
    y: 546,
    h: 44,
  },
  {
    id: 'enc-add-norm-1',
    part: 'add-norm',
    label: 'Add and norm',
    views: ORIGINAL,
    ...E,
    y: 500,
    h: 32,
  },
  { id: 'enc-ffn', part: 'ffn', label: 'Feed forward', views: ORIGINAL, ...E, y: 436, h: 44 },
  {
    id: 'enc-add-norm-2',
    part: 'add-norm',
    label: 'Add and norm',
    views: ORIGINAL,
    ...E,
    y: 396,
    h: 32,
  },
  {
    id: 'input',
    part: 'input',
    label: { original: 'Outputs (shifted right)', gpt2: 'Inputs' },
    views: BOTH,
    ...D,
    y: 746,
    h: 40,
  },
  {
    id: 'embedding',
    part: 'embedding',
    label: { original: 'Output embedding', gpt2: 'Token embedding' },
    views: BOTH,
    ...D,
    y: 686,
    h: 44,
  },
  {
    id: 'position',
    part: 'position',
    label: { original: 'Positional encoding', gpt2: 'Position embedding' },
    views: BOTH,
    ...D,
    y: 630,
    h: 40,
  },
  {
    id: 'masked-attn',
    part: 'masked-attn',
    label: 'Masked multi-head attention',
    views: BOTH,
    ...D,
    y: 546,
    h: 44,
  },
  {
    id: 'add-norm-1',
    part: 'add-norm',
    label: { original: 'Add and norm', gpt2: 'Norm, then add' },
    views: BOTH,
    ...D,
    y: 500,
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
    views: ORIGINAL,
    ...D,
    y: 396,
    h: 32,
  },
  { id: 'ffn', part: 'ffn', label: 'Feed forward', views: BOTH, ...D, y: 316, h: 44 },
  {
    id: 'add-norm-3',
    part: 'add-norm',
    label: { original: 'Add and norm', gpt2: 'Norm, then add' },
    views: BOTH,
    ...D,
    y: 270,
    h: 32,
  },
  { id: 'stack', part: 'stack', label: 'N×', views: BOTH, x: 576, y: 400, w: 48, h: 44 },
  { id: 'final-norm', part: 'final-norm', label: 'Final norm', views: GPT2, ...D, y: 206, h: 32 },
  { id: 'linear', part: 'linear', label: 'Linear', views: BOTH, ...D, y: 150, h: 36 },
  { id: 'softmax', part: 'softmax', label: 'Softmax', views: BOTH, ...D, y: 96, h: 36 },
  { id: 'output', part: 'output', label: 'Output probabilities', views: BOTH, ...D, y: 40, h: 36 },
];

/** The frames drawn around each repeated block, decoder then encoder. */
export const FRAMES = [
  { views: BOTH, x: 324, y: 256, w: 252, h: 346 },
  { views: ORIGINAL, x: 24, y: 384, w: 252, h: 218 },
] as const;

const DECODER_TOUR = ['input', 'embedding', 'position', 'masked-attn', 'add-norm-1'];
export const TOUR: Record<View, readonly string[]> = {
  original: [
    'enc-input',
    'enc-embedding',
    'enc-position',
    'enc-attn',
    'enc-add-norm-1',
    'enc-ffn',
    'enc-add-norm-2',
    ...DECODER_TOUR,
    'cross-attn',
    'add-norm-2',
    'ffn',
    'add-norm-3',
    'stack',
    'linear',
    'softmax',
    'output',
  ],
  gpt2: [
    ...DECODER_TOUR,
    'ffn',
    'add-norm-3',
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

export function visibleBlocks(view: View): Block[] {
  return BLOCKS.filter((b) => b.views.includes(view));
}

/** The block a link to this part lands on, the first one in the tour for that view. */
export function firstBlockFor(part: PartId, view: View): Block | undefined {
  const id = TOUR[view].find((blockId) => BLOCKS.find((b) => b.id === blockId)?.part === part);
  return BLOCKS.find((b) => b.id === id);
}
