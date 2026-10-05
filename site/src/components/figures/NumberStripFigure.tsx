import { useId, useState, type KeyboardEvent } from 'react';
import type { EmbeddedToken } from '../../lib/embeddings';
import { ChipPicker } from './ChipPicker';
import './figures.css';

export type StripView = 'token' | 'position' | 'sum';

const VIEWS: Record<StripView, string> = {
  token: 'Token',
  position: 'Position',
  sum: 'Added together',
};

const COLUMNS = 16;
/** Values at or beyond this size get the full color. Most numbers are much smaller. */
const FULL_COLOR = 0.25;

const fmt = (value: number) => value.toFixed(3);

function rowFor(token: EmbeddedToken, view: StripView): number[] {
  return view === 'token' ? token.tokenRow : view === 'position' ? token.positionRow : token.sum;
}

/** Where the biggest and smallest numbers are, so the strip never relies on color alone. */
function describe(row: number[]): string {
  let hi = 0;
  let lo = 0;
  row.forEach((v, i) => {
    if (v > row[hi]) hi = i;
    if (v < row[lo]) lo = i;
  });
  return `Biggest is number ${hi + 1} at ${fmt(row[hi])}. Smallest is number ${lo + 1} at ${fmt(row[lo])}.`;
}

/** One token's 128 numbers as a grid of colored cells, with its position's numbers and the sum. */
export default function NumberStripFigure({
  tokens,
  view: initialView = 'token',
  token: initialToken = 0,
}: {
  tokens: EmbeddedToken[];
  view?: StripView;
  token?: number;
}) {
  const [picked, setPicked] = useState(initialToken);
  const [view, setView] = useState<StripView>(initialView);
  const [active, setActive] = useState(0);
  const id = useId();
  const token = tokens[picked];
  const row = rowFor(token, view);
  const size = row.length;
  const changed = picked !== initialToken || view !== initialView || active !== 0;

  const move = (event: KeyboardEvent<HTMLDivElement>) => {
    const next =
      event.key === 'ArrowRight'
        ? active + 1
        : event.key === 'ArrowLeft'
          ? active - 1
          : event.key === 'ArrowDown'
            ? active + COLUMNS
            : event.key === 'ArrowUp'
              ? active - COLUMNS
              : event.key === 'Home'
                ? 0
                : event.key === 'End'
                  ? size - 1
                  : null;
    if (next === null) return;
    event.preventDefault();
    setActive(Math.min(Math.max(next, 0), size - 1));
  };

  return (
    <div className="strip-figure">
      <ChipPicker
        label="Pick a token"
        tokens={tokens.map((t) => t.token)}
        selected={picked}
        onSelect={setPicked}
      />
      <div className="strip-controls">
        <fieldset className="segmented">
          <legend className="visually-hidden">Show the numbers for</legend>
          {(Object.keys(VIEWS) as StripView[]).map((v) => (
            <label key={v}>
              <input
                type="radio"
                name={`${id}-view`}
                checked={view === v}
                onChange={() => setView(v)}
              />
              <span>{VIEWS[v]}</span>
            </label>
          ))}
        </fieldset>
        <button
          type="button"
          className="figure-button press"
          aria-disabled={!changed}
          onClick={() => {
            if (!changed) return;
            setPicked(initialToken);
            setView(initialView);
            setActive(0);
          }}
        >
          Start over
        </button>
      </div>
      <div
        className={`strip-grid strip-${view}`}
        role="listbox"
        tabIndex={0}
        aria-label={`The ${size} ${VIEWS[view].toLowerCase()} numbers for ${token.token.text.trim()}`}
        aria-activedescendant={`${id}-cell-${active}`}
        onKeyDown={move}
      >
        {row.map((value, i) => {
          const strength = Math.round(Math.min(Math.abs(value) / FULL_COLOR, 1) * 100);
          return (
            <div
              key={i}
              id={`${id}-cell-${i}`}
              role="option"
              aria-selected={i === active}
              aria-label={`Number ${i + 1}, ${fmt(value)}`}
              className={`strip-cell${i === active ? ' is-active' : ''}`}
              style={{
                background: `color-mix(in srgb, var(${value < 0 ? '--strip-negative' : '--strip-positive'}) ${strength}%, var(--color-surface))`,
              }}
              onMouseEnter={() => setActive(i)}
              onClick={() => setActive(i)}
            />
          );
        })}
      </div>
      <p className="strip-readout">
        Number {active + 1} of {size}.{' '}
        <span className={view === 'token' ? 'is-current' : undefined}>
          Token {fmt(token.tokenRow[active])}
        </span>{' '}
        +{' '}
        <span className={view === 'position' ? 'is-current' : undefined}>
          position {fmt(token.positionRow[active])}
        </span>{' '}
        ={' '}
        <span className={view === 'sum' ? 'is-current' : undefined}>{fmt(token.sum[active])}</span>
      </p>
      <p className="strip-summary">
        Blue is below zero and orange is above, and paler means closer to zero. {describe(row)}
      </p>
    </div>
  );
}
