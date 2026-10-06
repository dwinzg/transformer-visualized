import { useEffect, useId, useRef, useState } from 'react';
import { unsupportedCharacters } from '@transformer-visualized/engine';
import { GRID_TOKENS, type PlaygroundRun, type WorkerMessage } from '../../lib/playground-run';
import { shownToken, spokenToken } from '../../lib/token-text';
import AttentionGridFigure from './AttentionGridFigure';
import FeedForwardFigure from './FeedForwardFigure';
import NumberStripFigure from './NumberStripFigure';
import ResidualFigure from './ResidualFigure';
import SamplingFigure from './SamplingFigure';
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
    guide: "Every step adds its result to the token's numbers, so nothing is lost on the way.",
  },
  output: {
    label: 'Next token',
    guide: "The last token's numbers become a chance for every possible next token.",
  },
};
const ORDER = Object.keys(STAGES) as Stage[];

/** Run the real tiny model on any text, in the browser, and look at each stage. */
export default function Playground() {
  const [text, setText] = useState(START);
  const [stage, setStage] = useState<Stage>('output');
  const [mounted, setMounted] = useState(false);
  const [ready, setReady] = useState(false);
  const [run, setRun] = useState<PlaygroundRun | null>(null);
  const worker = useRef<Worker | null>(null);
  const seq = useRef(0);
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
    const shared = new URLSearchParams(location.search).get('text');
    if (shared !== null) {
      setText(shared.slice(0, MAX_LENGTH));
      setInput(shared.slice(0, MAX_LENGTH));
    }
    setMounted(true);
  }, []);
  // The address keeps up with the text, so it can be shared or bookmarked.
  useEffect(() => {
    if (!mounted) return;
    const url = new URL(location.href);
    if (input === START) url.searchParams.delete('text');
    else url.searchParams.set('text', input);
    history.replaceState(history.state, '', url);
    setCopied(false);
  }, [mounted, input]);
  // The model loads and runs in a worker. Try again starts a fresh one.
  useEffect(() => {
    setFailed(null);
    const w = new Worker(new URL('../../lib/playground.worker.ts', import.meta.url), {
      type: 'module',
    });
    w.onmessage = (event: MessageEvent<WorkerMessage>) => {
      const message = event.data;
      if (message.type === 'ready') setReady(true);
      else if (message.type === 'failed') setFailed('load');
      // Only the newest text counts. An older run that finishes late is dropped.
      else if (message.seq !== seq.current) return;
      else if (message.type === 'run-failed') setFailed('run');
      else {
        setFailed(null);
        setRun(message.run);
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
    worker.current?.postMessage({ seq: seq.current, text: input });
  }, [ready, input]);

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
            'Loading the model, about 5 MB. This happens once.'
          )}
        </div>
      )}
      {!run ? null : !run.views ? (
        <p className="playground-status">Type something to see what the model does with it.</p>
      ) : (
        <>
          <ol className="token-row tokenizer-tokens" aria-label="Tokens">
            {run.tokens.map((token, i) => (
              <li
                key={`${i}-${token.id}`}
                className={`token-chip${token.special ? ' is-special' : ''}`}
              >
                <span className="visually-hidden">{`${spokenToken(token.text)}, id ${token.id}`}</span>
                <span className="token-text" aria-hidden="true">
                  {token.special ? token.text : shownToken(token.text)}
                </span>
                <span className="token-id" aria-hidden="true">
                  {token.id}
                </span>
              </li>
            ))}
          </ol>
          <p className="tokenizer-count">
            {run.tokens.length} {run.tokens.length === 1 ? 'token' : 'tokens'}.
            {run.cut > 0 &&
              ` The model reads at most ${run.limit}, so the first ${run.cut} are left out.`}
            {/* The old numbers stay up until the new ones arrive, so say they are on the way. */}
            {run.text !== text && !failed && ' Updating.'}
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
              Step {ORDER.indexOf(stage) + 1} of {ORDER.length}. {STAGES[stage].guide}
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
              <NumberStripFigure tokens={run.views.embeddings} token={run.tokens.length - 1} />
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
                />
              </>
            )}
            {stage === 'ffn' && (
              <FeedForwardFigure tokens={run.tokens} activations={run.views.activations} />
            )}
            {stage === 'residual' && (
              <ResidualFigure tokens={run.tokens} steps={run.views.residual} />
            )}
            {stage === 'output' && (
              <SamplingFigure
                data={run.views.scores}
                // No Add while the model catches up with the text, or once the box is full.
                onAdd={
                  text === run.text && text.length < MAX_LENGTH
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
