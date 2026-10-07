import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import ts from 'typescript';
import { expect, it } from 'vitest';

// The Code level of each chapter shows short TypeScript samples that use the engine. This checks
// them against the engine's real types, so a renamed function or field cannot leave a sample behind.
const CHAPTERS = join(import.meta.dirname, '../content/chapters');

// What the chapters take as already there. A sample can still declare its own of each.
const SETUP = `
import {
  argmax, createRng, forward, generate, loadModel, nextTokenDistribution, probabilities, rowView,
  sample, softmax, type Model, type Tokenizer, type Trace,
} from '@transformer-visualized/engine';
declare const model: Model;
declare const tokenizer: Tokenizer;
declare const trace: Trace;
declare const ids: number[];
declare const tokenIds: number[];
declare const promptIds: number[];
declare const modelUrl: string;
declare const scores: ReturnType<typeof rowView>;
`;

function samples() {
  const found: { name: string; code: string }[] = [];
  for (const file of readdirSync(CHAPTERS).filter((f) => f.endsWith('.mdx'))) {
    const blocks = readFileSync(join(CHAPTERS, file), 'utf8').matchAll(/^```ts\n([\s\S]*?)^```/gm);
    let n = 0;
    for (const [, code] of blocks) found.push({ name: `${file} sample ${++n}`, code });
  }
  return found;
}

it('every TypeScript sample in the chapters type-checks against the engine', () => {
  const found = samples();
  expect(found.length).toBeGreaterThan(15);

  // Each sample sits in its own function, so samples can reuse names without clashing.
  const files = new Map(
    found.map(({ name, code }, i) => [
      join(import.meta.dirname, `__sample${i}.ts`),
      { name, text: `${SETUP}\nexport async function run() {\n${code}}\n` },
    ]),
  );
  const options: ts.CompilerOptions = {
    strict: true,
    // The samples are teaching code, and some leave out types to stay short.
    noImplicitAny: false,
    noEmit: true,
    target: ts.ScriptTarget.ES2022,
    module: ts.ModuleKind.ESNext,
    moduleResolution: ts.ModuleResolutionKind.Bundler,
    lib: ['lib.es2022.d.ts', 'lib.dom.d.ts'],
    skipLibCheck: true,
  };
  const host = ts.createCompilerHost(options);
  const { getSourceFile, fileExists, readFile } = host;
  host.fileExists = (f) => files.has(f) || fileExists(f);
  host.readFile = (f) => files.get(f)?.text ?? readFile(f);
  host.getSourceFile = (f, lang, ...rest) => {
    const file = files.get(f);
    return file ? ts.createSourceFile(f, file.text, lang) : getSourceFile(f, lang, ...rest);
  };

  const program = ts.createProgram([...files.keys()], options, host);
  const errors = ts.getPreEmitDiagnostics(program).map((d) => {
    const name = d.file ? (files.get(d.file.fileName)?.name ?? d.file.fileName) : '';
    return `${name}: ${ts.flattenDiagnosticMessageText(d.messageText, '\n')}`;
  });
  expect(errors).toEqual([]);
});
