import { useId, useState, type KeyboardEvent } from 'react';
import type { EmbeddedToken } from '../../lib/embeddings';
import { spokenToken } from '../../lib/token-text';
import { ChipPicker } from './ChipPicker';
import './figures.css';

export type StripView = 'token' | 'position' | 'sum';

const VIEWS: Record<StripView, string> = {
  token: 'Token',
  position: 'Position',
  sum: 'Added together',
};

/** How the grid's name says each view. */
const NAMES: Record<StripView, string> = {
  token: 'token',
  position: 'position',
  sum: 'token plus position',
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
  // Until a token is picked, the start token is used, so it follows the end as the playground text grows.
  const [picked, setPicked] = useState<number | null>(null);
  const [view, setView] = useState<StripView>(initialView);
  const [active, setActive] = useState(0);
  const id = useId();
  const at = Math.min(picked ?? initialToken, tokens.length - 1);
  const token = tokens[at];
  const row = rowFor(token, view);
  const size = row.length;
  const changed = at !== initialToken || view !== initialView;

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
    if (next >= 0 && next < size) setActive(next);
  };

  return (
    <div className="strip-figure">
      <ChipPicker
        label="Pick a token"
        tokens={tokens.map((t) => t.token)}
        selected={at}
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
            setPicked(null);
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
        aria-label={`The ${size} ${NAMES[view]} numbers for ${spokenToken(token.token)}`}
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
      <p className="strip-summary" aria-live="polite">
        Blue is below zero and orange is above, and paler means closer to zero. {describe(row)}
      </p>
    </div>
  );
}
