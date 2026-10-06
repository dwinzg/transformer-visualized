import { useState } from 'react';
import type { DisplayToken } from '../../lib/guess-tree';
import type { ResidualStep } from '../../lib/playground-run';
import { shownToken, spokenToken } from '../../lib/token-text';
import { ChipPicker } from './ChipPicker';
import './figures.css';

/** How each attention and feed-forward step adds to one token's numbers, the residual stream. */
export default function ResidualFigure({
  tokens,
  steps,
}: {
  tokens: DisplayToken[];
  /** [token][step], the first step being the embedding. */
  steps: ResidualStep[][];
}) {
  // Until a token is picked, the last one is shown, even as the text changes.
  const [picked, setPicked] = useState<number | null>(null);
  const row = Math.min(picked ?? tokens.length - 1, tokens.length - 1);
  const list = steps[row];
  const updates = list.slice(1);
  const biggest = updates.reduce((best, s) => (s.added > best.added ? s : best), updates[0]);
  const max = Math.max(...updates.map((s) => s.added), 1e-9);
  const token = tokens[row];

  return (
    <div className="residual-figure">
      <ChipPicker label="Pick a token" tokens={tokens} selected={row} onSelect={setPicked} />
      <p className="residual-summary" aria-live="polite">
        <span aria-hidden="true">{shownToken(token)}</span>
        <span className="visually-hidden">{spokenToken(token)}</span> starts with numbers of length{' '}
        {list[0].length.toFixed(2)}. {biggest.label} adds the most, {biggest.added.toFixed(2)}.
      </p>
      <ul className="prob-bars" aria-label="What each step adds">
        {updates.map((step) => (
          <li
            key={step.label}
            className="prob-bar"
            style={{ '--p': `${Math.max((step.added / max) * 100, 0.5)}%` } as React.CSSProperties}
          >
            <span className="prob-fill" aria-hidden="true" />
            <span className="prob-word">{step.label}</span>
            <span className="prob-value">
              +{step.added.toFixed(2)}
              <span className="residual-length"> to {step.length.toFixed(2)}</span>
            </span>
          </li>
        ))}
      </ul>
      <p className="residual-note">
        Each bar is the length of what that step adds. The step adds it on top of the numbers that
        came in, so nothing is thrown away. The second number is the length after the step.
      </p>
    </div>
  );
}
