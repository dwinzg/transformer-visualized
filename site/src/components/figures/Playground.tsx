import { useEffect, useId, useRef, useState } from 'react';
import { unsupportedCharacters } from '@transformer-visualized/engine';
import {
  GRID_TOKENS,
  type Ask,
  type AttentionMath,
  type PlaygroundRun,
  type WorkerMessage,
} from '../../lib/playground-run';
import type { Arch } from '../../lib/model-loader';
import AttentionInspector from './AttentionInspector';
import AttentionGridFigure from './AttentionGridFigure';
import FeedForwardFigure from './FeedForwardFigure';
import NumberStripFigure from './NumberStripFigure';
import ResidualFigure from './ResidualFigure';
import SamplingFigure from './SamplingFigure';
import { TokenList } from './TokenRow';
import './figures.css';

const START = 'Lily wanted to play with her';
const MAX_LENGTH = 400;
/** How long typing must pause before the model runs again. */
const WAIT_MS = 250;

type Stage = 'embeddings' | 'attention' | 'ffn' | 'residual' | 'output';
/** The stages in the order a token passes through them, each with one line on what it does. */
const STAGES: Record<Stage, { label: string; guide: string }> = {
  embeddings: {
    label: 'Embeddings',
    guide: 'Each token becomes 128 numbers, its token row plus its position row.',
  },
  attention: {
    label: 'Attention',
    guide: 'Each token looks back at the tokens before it and mixes in what it needs.',
  },
  ffn: {
    label: 'Feed forward',
    guide: "Then each token's numbers go through a small network on their own.",
  },
  residual: {
    label: 'Residual stream',
    guide:
      "Every step adds its result to the token's numbers, so earlier numbers are kept. A norm rescales them at the end.",
  },
  output: {
    label: 'Next token',
    guide: "The last token's numbers become a chance for every possible next token.",
  },
};
const ORDER = Object.keys(STAGES) as Stage[];

const ARCHS: Record<Arch, string> = { gpt2: 'GPT-2 style', llama: 'Llama style' };
const ARCH_KEY = 'tv-model';
/** What changes in a stage's guide for the Llama-style model. */
const LLAMA_GUIDES: Partial<Record<Stage, string>> = {
  embeddings:
    'Each token becomes its 128 token numbers. This model adds no position here. It turns queries and keys by position inside attention instead.',
  ffn: "Then each token's numbers go through a small gated network on their own.",
  residual:
    "Every step adds its result to the token's numbers, so earlier numbers are kept. An RMSNorm rescales them at the end.",
};

/** Run the real tiny model on any text, in the browser, and look at each stage. */
export default function Playground() {
  const [text, setText] = useState(START);
  const [arch, setArch] = useState<Arch>('gpt2');
  const [stage, setStage] = useState<Stage>('output');
  const [mounted, setMounted] = useState(false);
  const [ready, setReady] = useState(false);
  const [run, setRun] = useState<PlaygroundRun | null>(null);
  const worker = useRef<Worker | null>(null);
  const seq = useRef(0);
  // The run on screen, so the worker can explain its numbers, and the explanation open now.
  const shownSeq = useRef(0);
  const [inspected, setInspected] = useState<{ ask: Ask; math: AttentionMath } | null>(null);
  /** Why the model is not showing anything new, if something went wrong. */
  const [failed, setFailed] = useState<'load' | 'run' | null>(null);
  const [attempt, setAttempt] = useState(0);
  const status = useRef<HTMLDivElement>(null);
  const id = useId();
  // The model runs once typing pauses, so every key press stays quick.
  const [input, setInput] = useState(text);
  useEffect(() => {
    const timer = setTimeout(() => setInput(text), WAIT_MS);
    return () => clearTimeout(timer);
  }, [text]);

  const [copied, setCopied] = useState(false);

  // A link like playground/?text=Once%20upon opens with that text.
  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const shared = params.get('text');
    if (shared !== null) {
      setText(shared.slice(0, MAX_LENGTH));
      setInput(shared.slice(0, MAX_LENGTH));
    }
    // A link's model wins, then the one this reader picked last time.
    let saved: string | null = null;
    try {
      saved = localStorage.getItem(ARCH_KEY);
    } catch {
      // Storage can be blocked. Then the default model is used.
    }
    const asked = params.get('model') ?? saved;
    if (asked === 'llama' || asked === 'gpt2') setArch(asked);
    setMounted(true);
  }, []);
  // The address keeps up with the text, so it can be shared or bookmarked.
  useEffect(() => {
    if (!mounted) return;
    const url = new URL(location.href);
    if (input === START) url.searchParams.delete('text');
    else url.searchParams.set('text', input);
    if (arch === 'gpt2') url.searchParams.delete('model');
    else url.searchParams.set('model', arch);
    history.replaceState(history.state, '', url);
    setCopied(false);
  }, [mounted, input, arch]);
  // The model loads and runs in a worker. Try again starts a fresh one.
  useEffect(() => {
    setFailed(null);
    const w = new Worker(new URL('../../lib/playground.worker.ts', import.meta.url), {
      type: 'module',
    });
    w.onmessage = (event: MessageEvent<WorkerMessage>) => {
      const message = event.data;
      if (message.type === 'explained') {
        if (message.seq === shownSeq.current) {
          setInspected(message.math && { ask: message.ask, math: message.math });
        }
        return;
      }
      if (message.type === 'ready') setReady(true);
      else if (message.type === 'failed') setFailed('load');
      // Only the newest text counts. An older run that finishes late is dropped.
      else if (message.seq !== seq.current) return;
      else if (message.type === 'run-failed') setFailed('run');
      else {
        setFailed(null);
        setRun(message.run);
        shownSeq.current = message.seq;
        // New text means new numbers, so an open explanation would no longer match.
        setInspected(null);
      }
    };
    w.onerror = () => setFailed('load');
    worker.current = w;
    return () => {
      w.terminate();
      worker.current = null;
      setReady(false);
    };
  }, [attempt]);
  useEffect(() => {
    if (!ready) return;
    seq.current += 1;
    worker.current?.postMessage({ seq: seq.current, text: input, arch });
  }, [ready, input, arch]);

  const pickArch = (next: Arch) => {
    setArch(next);
    try {
      localStorage.setItem(ARCH_KEY, next);
    } catch {
      // The choice still holds for this visit.
    }
  };
  const guide = (s: Stage) => (arch === 'llama' && LLAMA_GUIDES[s]) || STAGES[s].guide;

  const unseen = unsupportedCharacters(text);

  return (
    <div className="playground">
      <div className="tokenizer-head">
        <label htmlFor={`${id}-text`}>Your text</label>
        <button
          type="button"
          className="figure-button press"
          aria-disabled={text === START}
          onClick={() => setText(START)}
        >
          Start over
        </button>
        {mounted && navigator.clipboard && (
          <button
            type="button"
            className="figure-button press"
            onClick={() =>
              navigator.clipboard.writeText(location.href).then(
                () => setCopied(true),
                () => setCopied(false),
              )
            }
          >
            {copied ? 'Link copied' : 'Copy link'}
          </button>
        )}
      </div>
      <textarea
        id={`${id}-text`}
        className="tokenizer-input"
        rows={3}
        maxLength={MAX_LENGTH}
        spellCheck={false}
        // Typing before the page is ready would be lost, so wait until it is.
        readOnly={!mounted}
        value={text}
        onChange={(event) => setText(event.target.value)}
      />
      <fieldset className="segmented playground-model">
        <legend>Model</legend>
        {(Object.keys(ARCHS) as Arch[]).map((a) => (
          <label key={a}>
            <input
              type="radio"
              name={`${id}-model`}
              checked={arch === a}
              disabled={!mounted}
              onChange={() => pickArch(a)}
            />
            <span>{ARCHS[a]}</span>
          </label>
        ))}
      </fieldset>
      {unseen.length > 0 && (
        <p className="tokenizer-unseen">
          The model never saw {unseen.join(' ')}, so it reads them as rare byte pieces.
        </p>
      )}

      {(!run || failed) && (
        <div role="status" className="playground-status" tabIndex={-1} ref={status}>
          {failed ? (
            <>
              {failed === 'load'
                ? 'The model did not load. Check your connection, then'
                : 'The model could not run on this text.'}{' '}
              <button
                type="button"
                className="figure-button press"
                onClick={() => {
                  setAttempt((n) => n + 1);
                  // The button goes away while loading, so the message keeps the focus.
                  status.current?.focus();
                }}
              >
                Try again
              </button>
            </>
          ) : (
            `Loading the ${ARCHS[arch]} model, about 5 MB. This happens once.`
          )}
        </div>
      )}
      {!run ? null : !run.views ? (
        <p className="playground-status">Type something to see what the model does with it.</p>
      ) : (
        <>
          <TokenList tokens={run.tokens} />
          <p className="tokenizer-count">
            {run.tokens.length} {run.tokens.length === 1 ? 'token' : 'tokens'}.
            {run.cut > 0 &&
              ` The model reads at most ${run.limit}, so the first ${run.cut} are left out.`}
            {/* The old numbers stay up until the new ones arrive, so say they are on the way. */}
            {(run.text !== text || run.arch !== arch) && !failed && ' Updating.'}
          </p>

          <fieldset className="segmented playground-stages">
            <legend className="visually-hidden">Stage</legend>
            {ORDER.map((s) => (
              <label key={s}>
                <input
                  type="radio"
                  name={`${id}-stage`}
                  checked={stage === s}
                  onChange={() => setStage(s)}
                />
                <span>{STAGES[s].label}</span>
              </label>
            ))}
          </fieldset>
          <div className="playground-guide">
            <p aria-live="polite">
              Step {ORDER.indexOf(stage) + 1} of {ORDER.length}. {guide(stage)}
            </p>
            <div className="loop-controls">
              <button
                type="button"
                className="figure-button press"
                aria-disabled={stage === ORDER[0]}
                onClick={() => stage !== ORDER[0] && setStage(ORDER[ORDER.indexOf(stage) - 1])}
              >
                Previous step
              </button>
              <button
                type="button"
                className="figure-button press"
                aria-disabled={stage === ORDER.at(-1)}
                onClick={() => stage !== ORDER.at(-1) && setStage(ORDER[ORDER.indexOf(stage) + 1])}
              >
                Next step
              </button>
            </div>
          </div>

          <section className="playground-panel" aria-label={STAGES[stage].label}>
            {stage === 'embeddings' && (
              <NumberStripFigure
                // A new strip for the other model, since it has no position views.
                key={run.arch}
                tokens={run.views.embeddings}
                token={run.tokens.length - 1}
                positions={run.arch === 'gpt2'}
              />
            )}
            {stage === 'attention' && (
              <>
                {run.views.attention.from > 0 && (
                  <p className="playground-note">
                    The grid shows the last {GRID_TOKENS} tokens. Weights on earlier tokens are left
                    out, so a row can add up to less than 1.
                  </p>
                )}
                <AttentionGridFigure
                  tokens={run.views.attention.tokens}
                  weights={run.views.attention.weights}
                  from={run.views.attention.from}
                  scores={run.views.attention.scores}
                  onInspect={(ask) =>
                    worker.current?.postMessage({
                      seq: shownSeq.current,
                      // The grid shows the last tokens only, so count from the start of the text.
                      ask: {
                        ...ask,
                        row: ask.row + run.views!.attention.from,
                        column: ask.column + run.views!.attention.from,
                      },
                    })
                  }
                  inspected={
                    inspected && {
                      ...inspected.ask,
                      row: inspected.ask.row - run.views.attention.from,
                    }
                  }
                >
                  {inspected && (
                    <AttentionInspector
                      math={inspected.math}
                      from={run.tokens[inspected.ask.row]}
                      to={run.tokens[inspected.ask.column]}
                      onClose={() => {
                        const column = inspected.ask.column - run.views!.attention.from;
                        setInspected(null);
                        // Back to the token that opened it, so the reader keeps their place.
                        requestAnimationFrame(() =>
                          document
                            .querySelector<HTMLElement>(`[data-inspect-column="${column}"]`)
                            ?.focus(),
                        );
                      }}
                    />
                  )}
                </AttentionGridFigure>
              </>
            )}
            {stage === 'ffn' && (
              <FeedForwardFigure tokens={run.tokens} activations={run.views.activations} />
            )}
            {stage === 'residual' && (
              <ResidualFigure
                tokens={run.tokens}
                steps={run.views.residual}
                norms={run.views.norm}
              />
            )}
            {stage === 'output' && (
              <SamplingFigure
                data={run.views.scores}
                inspect
                // No Add while the model catches up with the text, or once the box is full.
                onAdd={
                  text === run.text && arch === run.arch && text.length < MAX_LENGTH
                    ? (token) => setText((t) => (t + token.text).slice(0, MAX_LENGTH))
                    : undefined
                }
              />
            )}
          </section>
        </>
      )}
    </div>
  );
}
