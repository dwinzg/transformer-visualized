import { readFileSync, readdirSync } from 'node:fs';
import katex from 'katex';
import { describe, expect, it } from 'vitest';
import {
  BLOCKS,
  blockLabel,
  blockName,
  blockY,
  BOX,
  firstBlockFor,
  TOUR,
  visibleBlocks,
  type Block,
} from './architecture-layout';
import { formulaFor, isPartId, mapPath, PARTS, VIEWS, type View } from './concepts';
import { stepId } from './stepId';

const glossary = JSON.parse(
  readFileSync(new URL('../content/glossary.json', import.meta.url), 'utf8'),
) as { id: string }[];
const chapterDir = new URL('../content/chapters/', import.meta.url);
const chapterFiles = new Set(readdirSync(chapterDir).map((f) => f.replace(/\.mdx$/, '')));

describe('concepts', () => {
  it('links only to glossary entries that exist', () => {
    const ids = new Set(glossary.map((g) => g.id));
    for (const part of Object.values(PARTS))
      if (part.glossary) expect(ids, part.id).toContain(part.glossary);
  });
  it('points step anchors at steps that exist in their chapter', () => {
    for (const part of Object.values(PARTS)) {
      if (!part.step) continue;
      expect(part.chapter, part.id).toBeDefined();
      expect(chapterFiles, part.id).toContain(part.chapter);
      const source = readFileSync(new URL(`${part.chapter}.mdx`, chapterDir), 'utf8');
      const ids = [...source.matchAll(/<Step title="([^"]+)"/g)].map((m) =>
        stepId(m[1], new Set()),
      );
      expect(ids, part.id).toContain(part.step);
    }
  });
  it('keeps the story short and plain', () => {
    for (const part of Object.values(PARTS)) {
      for (const text of [part.story, ...Object.values(part.notes ?? {})]) {
        expect(text, part.id).not.toMatch(/[\u2014:]/);
      }
      expect(part.story.split(/\s+/).length, part.id).toBeLessThanOrEqual(40);
    }
  });
  it('renders every formula with KaTeX', () => {
    for (const part of Object.values(PARTS))
      for (const view of VIEWS) {
        const tex = formulaFor(part, view);
        if (tex)
          expect(() => katex.renderToString(tex, { throwOnError: true }), part.id).not.toThrow();
      }
  });
  it('recognizes part ids and builds map paths', () => {
    expect(isPartId('ffn')).toBe(true);
    expect(isPartId('toString')).toBe(false);
    expect(mapPath('ffn')).toBe('learn/architecture/?part=ffn&view=gpt2');
    expect(mapPath()).toBe('learn/architecture/');
  });
});

describe('architecture layout', () => {
  it('uses only known parts', () => {
    for (const block of BLOCKS) expect(isPartId(block.part), block.id).toBe(true);
  });
  it.each(VIEWS)('tours every visible block once in the %s view', (view: View) => {
    expect([...TOUR[view]].sort()).toEqual(
      visibleBlocks(view)
        .map((b) => b.id)
        .sort(),
    );
  });
  it('hides the encoder and cross-attention in the GPT-2 view, and the final norm in the original', () => {
    expect(TOUR.gpt2).not.toContain('cross-attn');
    expect(TOUR.gpt2.some((id) => id.startsWith('enc-'))).toBe(false);
    expect(TOUR.original).not.toContain('final-norm');
  });
  it.each(VIEWS)('keeps blocks inside the box and apart in the %s view', (view: View) => {
    const blocks = visibleBlocks(view);
    const y = (b: Block) => blockY(b, view);
    for (const b of blocks) {
      expect(
        b.x >= 0 && y(b) >= 0 && b.x + b.w <= BOX.width && y(b) + b.h <= BOX.height,
        b.id,
      ).toBe(true);
    }
    for (const a of blocks)
      for (const b of blocks) {
        if (a === b) continue;
        const apart =
          a.x + a.w <= b.x || b.x + b.w <= a.x || y(a) + a.h <= y(b) || y(b) + b.h <= y(a);
        expect(apart, `${a.id} and ${b.id}`).toBe(true);
      }
  });
  it('draws each GPT-2 norm below its sublayer, since the diagram reads upward', () => {
    const at = (id: string) =>
      blockY(
        BLOCKS.find((b) => b.id === id)!,
        'gpt2',
      );
    expect(at('add-norm-1')).toBeGreaterThan(at('masked-attn'));
    expect(at('add-norm-3')).toBeGreaterThan(at('ffn'));
    expect(TOUR.gpt2.indexOf('add-norm-1')).toBeLessThan(TOUR.gpt2.indexOf('masked-attn'));
  });
  it.each(VIEWS)(
    'gives every block a unique name that contains its label in the %s view',
    (view: View) => {
      const names = visibleBlocks(view).map((b) => blockName(b, view));
      expect(new Set(names).size).toBe(names.length);
      for (const b of visibleBlocks(view))
        expect(blockName(b, view).toLowerCase()).toContain(blockLabel(b, view).toLowerCase());
    },
  );
  it('lands links on the first block of a part', () => {
    expect(firstBlockFor('add-norm', 'gpt2')?.id).toBe('add-norm-1');
    expect(firstBlockFor('input', 'original')?.id).toBe('enc-input');
    expect(firstBlockFor('cross-attn', 'gpt2')).toBeUndefined();
  });
});

describe('the code lines', () => {
  it('only name values the engine really traces', async () => {
    const { flattenTrace, forward, loadModel } = await import('@transformer-visualized/engine');
    const bytes = readFileSync(new URL('../../../models/tiny/model.safetensors', import.meta.url));
    const buffer = new ArrayBuffer(bytes.byteLength);
    new Uint8Array(buffer).set(bytes);
    const trace = forward(loadModel(buffer), [1, 2, 3]);
    const paths = [...flattenTrace(trace).keys()];
    for (const part of Object.values(PARTS)) {
      for (const [, path] of (part.code ?? '').matchAll(/trace\.([\w.[\]]+)/g)) {
        const flat = path.replace(/\[(\d+)\]/g, '.$1').replace(/\.length$/, '');
        expect(
          paths.some((p) => p === flat || p.startsWith(`${flat}.`)),
          `${part.id}: trace.${path}`,
        ).toBe(true);
      }
    }
  });
});
