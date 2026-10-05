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
}

/** A token drawn with its space marks, and spoken with them too. */
const Shown = ({ token }: { token: DisplayToken }) => (
  <>
    <span aria-hidden="true">{shownToken(token.text)}</span>
    <span className="visually-hidden">{spokenToken(token.text)}</span>
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
}: Props) {
  const [layer, setLayer] = useState(startLayer);
  const [head, setHead] = useState(startHead);
  const [row, setRow] = useState(startRow);
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
            setRow(startRow);
          }}
        >
          Start over
        </button>
      </div>
      <ChipPicker
        label="Pick the token that looks"
        tokens={tokens}
        selected={row}
        onSelect={setRow}
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
            column is a token it can look at. Each row adds up to 1.
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
