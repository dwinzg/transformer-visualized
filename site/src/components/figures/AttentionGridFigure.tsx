import { Fragment, useId, useState } from 'react';
import type { DisplayToken } from '../../lib/guess-tree';
import { shownToken, spokenToken } from '../../lib/token-text';
import { ChipPicker } from './ChipPicker';
import './figures.css';

interface Props {
  tokens: DisplayToken[];
  /** [layer][head][row][column] */
  weights: number[][][][];
  /** Starting layer, head and row, counting from 0. */
  layer?: number;
  head?: number;
  row?: number;
  /** How many earlier tokens are cut off. Then the rows no longer add up to 1. */
  from?: number;
  /** Scores before softmax, [layer][head][row][column up to row]. Shown when given. */
  scores?: number[][][][];
  /** When given, each token in the scores table is a button that asks how its weight is made. */
  onInspect?: (ask: { layer: number; head: number; row: number; column: number }) => void;
  /** What the open inspector explains, counting from the first token shown. */
  inspected?: { layer: number; head: number; row: number } | null;
  /** The open inspector. It shows only while the grid still shows its layer, head and token. */
  children?: React.ReactNode;
}

/** A token drawn with its space marks, and spoken with them too. */
const Shown = ({ token }: { token: DisplayToken }) => (
  <>
    <span aria-hidden="true">{shownToken(token)}</span>
    <span className="visually-hidden">{spokenToken(token)}</span>
  </>
);

/** The other tokens a row looked at, biggest weight first. */
function top(row: number[], tokens: DisplayToken[], count: number) {
  return (
    row
      .map((w, j) => ({ w, token: tokens[j] }))
      .sort((a, b) => b.w - a.w)
      .slice(0, count)
      // A weight that rounds to 0.00 is not worth naming, but the top one always is.
      .filter(({ w }, k) => k === 0 || w >= 0.005)
  );
}

/** The top weights in a row, like "Lily 0.66, ·her 0.20". */
const Describe = ({ row, tokens }: { row: number[]; tokens: DisplayToken[] }) =>
  top(row, tokens, 3).map(({ w, token }, k) => (
    <Fragment key={k}>
      {k > 0 && ', '}
      <Shown token={token} /> {w.toFixed(2)}
    </Fragment>
  ));

/** The real model's attention weights for one sentence, one layer and head at a time. */
export default function AttentionGridFigure({
  tokens,
  weights,
  layer: startLayer = 0,
  head: startHead = 0,
  row: startRow = tokens.length - 1,
  from = 0,
  scores,
  onInspect,
  inspected,
  children,
}: Props) {
  const [layer, setLayer] = useState(startLayer);
  const [head, setHead] = useState(startHead);
  // A pick counts from the first token of the whole text, so it stays on the same token when the
  // cut moves. Until a token is picked, the start row is used.
  const [picked, setPicked] = useState<number | null>(null);
  const last = tokens.length - 1;
  const row = Math.min(Math.max(picked === null ? startRow : picked - from, 0), last);
  const id = useId();
  const grid = weights[layer][head];
  const changed = layer !== startLayer || head !== startHead || row !== startRow;
  const visible = grid[row].slice(0, row + 1);

  const picker = (label: string, value: number, count: number, set: (n: number) => void) => (
    <fieldset className="segmented">
      <legend className="attn-legend">{label}</legend>
      {Array.from({ length: count }, (_, n) => (
        <label key={n}>
          <input
            type="radio"
            name={`${id}-${label}`}
            checked={value === n}
            onChange={() => set(n)}
          />
          <span>{n + 1}</span>
        </label>
      ))}
    </fieldset>
  );

  return (
    <div className="attn-figure">
      <div className="attn-controls">
        {picker('Layer', layer, weights.length, setLayer)}
        {picker('Head', head, weights[0].length, setHead)}
        <button
          type="button"
          className="figure-button press"
          aria-disabled={!changed}
          onClick={() => {
            if (!changed) return;
            setLayer(startLayer);
            setHead(startHead);
            setPicked(null);
          }}
        >
          Start over
        </button>
      </div>
      <ChipPicker
        label="Pick the token that looks"
        tokens={tokens}
        selected={row}
        onSelect={(i) => setPicked(i + from)}
      />
      <p className="attn-summary" aria-live="polite">
        In layer {layer + 1}, head {head + 1},{' '}
        <strong>
          <Shown token={tokens[row]} />
        </strong>{' '}
        looks most at <Describe row={visible} tokens={tokens} />.
      </p>
      {/* Focusable, so the grid can be scrolled with keys where a phone is too narrow. */}
      <div className="attn-scroll" tabIndex={0} role="group" aria-label="Attention grid">
        <table className="attn-grid">
          <caption className="visually-hidden">
            Attention weights in layer {layer + 1}, head {head + 1}. Each row is a token, and each
            column is a token it can look at.{' '}
            {from > 0
              ? 'Weights on earlier tokens are left out, so a row can add up to less than 1.'
              : 'Each row adds up to 1.'}
          </caption>
          <thead>
            <tr>
              <td />
              {tokens.map((token, j) => (
                <th key={j} scope="col">
                  <Shown token={token} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tokens.map((token, i) => (
              <tr key={i} className={i === row ? 'is-picked' : undefined}>
                <th scope="row">
                  <Shown token={token} />
                </th>
                {grid[i].map((w, j) => (
                  <td
                    key={j}
                    className={j > i ? 'is-hidden' : undefined}
                    style={
                      j > i
                        ? undefined
                        : {
                            // Capped at half strength, so the number on top stays readable.
                            background: `color-mix(in srgb, var(--concept-query) ${Math.round(w * 50)}%, var(--color-surface))`,
                          }
                    }
                  >
                    {j > i ? (
                      <>
                        <span className="visually-hidden">hidden</span>
                      </>
                    ) : (
                      w.toFixed(2)
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {scores && (
        <details className="attn-scores">
          <summary>
            How <Shown token={tokens[row]} /> gets its weights
          </summary>
          <p>
            Each score is the query of <Shown token={tokens[row]} /> times the key of another token,
            divided by √32. Softmax turns the scores into weights, so a bigger score gets a bigger
            weight.
            {onInspect && ' Pick a token to see the math, number by number.'}
          </p>
          <div className="attn-score-scroll" tabIndex={0} role="group" aria-label="Scores">
            <table className="attn-score-table">
              <caption className="visually-hidden">
                Scores and weights for {spokenToken(tokens[row])} in layer {layer + 1}, head{' '}
                {head + 1}.
              </caption>
              <thead>
                <tr>
                  <th scope="col">Token</th>
                  <th scope="col">Score</th>
                  <th scope="col">Weight</th>
                </tr>
              </thead>
              <tbody>
                {scores[layer][head][row].map((score, j) => (
                  <tr key={j}>
                    <th scope="row">
                      {onInspect ? (
                        <button
                          type="button"
                          className="attn-inspect press"
                          aria-label={`${spokenToken(tokens[j])}, show the math`}
                          data-inspect-column={j}
                          onClick={() => onInspect({ layer, head, row, column: j })}
                        >
                          {shownToken(tokens[j])}
                        </button>
                      ) : (
                        <Shown token={tokens[j]} />
                      )}
                    </th>
                    <td>{score.toFixed(2)}</td>
                    <td>{grid[row][j].toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
      {inspected &&
        inspected.layer === layer &&
        inspected.head === head &&
        inspected.row === row &&
        children}
      <details className="attn-text">
        <summary>Top three for every token</summary>
        <ul>
          {tokens.map((token, i) => (
            <li key={i}>
              <Shown token={token} /> looks at{' '}
              <Describe row={grid[i].slice(0, i + 1)} tokens={tokens} />.
            </li>
          ))}
        </ul>
      </details>
    </div>
  );
}
