import { useEffect, useId, useMemo, useRef, useState } from 'react';
import {
  forward,
  normalizeText,
  rowView,
  unsupportedCharacters,
  type Tokenizer,
} from '@transformer-visualized/engine';
import { displayToken, loadTinyModel, loadTokenizer, type Model } from '../../lib/model-loader';
import { attentionView, embeddingView, scoresView } from '../../lib/model-views';
import { round } from '../../lib/embeddings';
import { shownToken, spokenToken } from '../../lib/token-text';
import AttentionGridFigure from './AttentionGridFigure';
import FeedForwardFigure from './FeedForwardFigure';
import NumberStripFigure from './NumberStripFigure';
import SamplingFigure from './SamplingFigure';
import './figures.css';

const START = 'Lily wanted to play with her';
const MAX_LENGTH = 400;
/** How long typing must pause before the model runs again. */
const WAIT_MS = 250;
/** A grid for every token would be huge, so attention shows the last few. */
const GRID_TOKENS = 24;

type Stage = 'embeddings' | 'attention' | 'ffn' | 'output';
const STAGES: Record<Stage, string> = {
  embeddings: 'Embeddings',
  attention: 'Attention',
  ffn: 'Feed forward',
  output: 'Next token',
};

/** Run the real tiny model on any text, in the browser, and look at each stage. */
export default function Playground() {
  const [text, setText] = useState(START);
  const [stage, setStage] = useState<Stage>('output');
  const [mounted, setMounted] = useState(false);
  const [loaded, setLoaded] = useState<{ model: Model; tokenizer: Tokenizer } | null>(null);
  const [failed, setFailed] = useState(false);
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
  useEffect(() => {
    let live = true;
    setFailed(false);
    Promise.all([loadTinyModel(), loadTokenizer()]).then(
      ([model, tokenizer]) => live && setLoaded({ model, tokenizer }),
      () => live && setFailed(true),
    );
    return () => {
      live = false;
    };
  }, [attempt]);

  const run = useMemo(() => {
    if (!loaded) return null;
    const { model, tokenizer } = loaded;
    const all = tokenizer.encode(normalizeText(input));
    // The model reads at most contextLength tokens, so a long text keeps its end.
    const ids = all.slice(-model.config.contextLength);
    const tokens = ids.map((n) => displayToken(tokenizer, n));
    const limit = model.config.contextLength;
    if (ids.length === 0) return { tokens, cut: 0, limit, views: null };
    const trace = forward(model, ids);
    const attention = attentionView(tokens, trace);
    const from = Math.max(tokens.length - GRID_TOKENS, 0);
    return {
      tokens,
      cut: all.length - ids.length,
      limit,
      views: {
        embeddings: embeddingView(tokens, trace),
        attention: {
          from,
          tokens: attention.tokens.slice(from),
          weights: attention.weights.map((heads) =>
            heads.map((grid) => grid.slice(from).map((row) => row.slice(from))),
          ),
        },
        scores: scoresView(
          tokens,
          trace,
          (n) => displayToken(tokenizer, n),
          model.config.vocabSize,
        ),
        activations: trace.layers.map((layer) =>
          tokens.map((_, i) => Array.from(rowView(layer.mlpAct, i), (v) => round(v, 2))),
        ),
      },
    };
  }, [loaded, input]);

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

      {!run ? (
        <div role="status" className="playground-status" tabIndex={-1} ref={status}>
          {failed ? (
            <>
              The model did not load. Check your connection, then{' '}
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
      ) : !run.views ? (
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
          </p>

          <fieldset className="segmented playground-stages">
            <legend className="visually-hidden">Stage</legend>
            {(Object.keys(STAGES) as Stage[]).map((s) => (
              <label key={s}>
                <input
                  type="radio"
                  name={`${id}-stage`}
                  checked={stage === s}
                  onChange={() => setStage(s)}
                />
                <span>{STAGES[s]}</span>
              </label>
            ))}
          </fieldset>

          <section className="playground-panel" aria-label={STAGES[stage]}>
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
                />
              </>
            )}
            {stage === 'ffn' && (
              <FeedForwardFigure tokens={run.tokens} activations={run.views.activations} />
            )}
            {stage === 'output' && (
              <SamplingFigure
                data={run.views.scores}
                // No Add while the model catches up with the text, or once the box is full.
                onAdd={
                  text === input && text.length < MAX_LENGTH
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
