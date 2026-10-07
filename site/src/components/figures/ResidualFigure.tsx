import { useState } from 'react';
import type { DisplayToken } from '../../lib/guess-tree';
import type { NormStep, ResidualStep } from '../../lib/playground-run';
import { shownToken, spokenToken } from '../../lib/token-text';
import { ChipPicker } from './ChipPicker';
import './figures.css';

/** How each attention and feed-forward step adds to one token's numbers, the residual stream. */
export default function ResidualFigure({
  tokens,
  steps,
  norms,
}: {
  tokens: DisplayToken[];
  /** [token][step], the first step being the embedding. */
  steps: ResidualStep[][];
  /** How the final norm rescales each token. */
  norms: NormStep[];
}) {
  // Until a token is picked, the last one is shown, even as the text changes.
  const [picked, setPicked] = useState<number | null>(null);
  const row = Math.min(picked ?? tokens.length - 1, tokens.length - 1);
  const list = steps[row];
  const updates = list.slice(1);
  const biggest = updates.reduce((best, s) => (s.added > best.added ? s : best), updates[0]);
  const max = Math.max(...updates.map((s) => s.added), 1e-9);
  const token = tokens[row];
  const n = norms[row];
  const name = (
    <>
      <span aria-hidden="true">{shownToken(token)}</span>
      <span className="visually-hidden">{spokenToken(token)}</span>
    </>
  );

  return (
    <div className="residual-figure">
      <ChipPicker label="Pick a token" tokens={tokens} selected={row} onSelect={setPicked} />
      <p className="residual-summary">
        {name} starts with numbers of length {list[0].length.toFixed(2)}. {biggest.label} adds the
        most, {biggest.added.toFixed(2)}.
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
      <details className="attn-scores norm-details">
        <summary>Then the final norm rescales {name}</summary>
        <p>
          Before the numbers become chances, they go through one last norm. It takes their length
          from {n.before.toFixed(2)} to {n.after.toFixed(2)}.
        </p>
        <ol className="inspector-steps">
          <li>
            Take the average of all the numbers. For {name} it is{' '}
            <strong>{n.mean.toFixed(4)}</strong>.
          </li>
          <li>
            Measure how spread out they are, add a tiny amount to their variance and take the square
            root. That is <strong>{n.spread.toFixed(4)}</strong>.
          </li>
          <li>
            Subtract the average and divide by the spread. Now every token&apos;s numbers sit on the
            same scale.
          </li>
          <li>Multiply each number by a learned scale and add a learned shift.</li>
        </ol>
        <div className="attn-score-scroll" tabIndex={0} role="group" aria-label="First numbers">
          <table className="attn-score-table">
            <caption className="visually-hidden">
              The first {n.values.length} numbers of {spokenToken(token)}, step by step.
            </caption>
            <thead>
              <tr>
                <th scope="col">Number</th>
                <th scope="col">In</th>
                <th scope="col">Normed</th>
                <th scope="col">Scale</th>
                <th scope="col">Shift</th>
                <th scope="col">Out</th>
              </tr>
            </thead>
            <tbody>
              {n.values.map((v, i) => (
                <tr key={i}>
                  <th scope="row">{i + 1}</th>
                  <td>{v.input.toFixed(4)}</td>
                  <td>{v.normalized.toFixed(4)}</td>
                  <td>{v.gamma.toFixed(4)}</td>
                  <td>{v.beta.toFixed(4)}</td>
                  <td>{v.output.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="residual-note">
          Each block uses the same recipe, with its own scale and shift, on the way into its
          attention and again into its feed forward. Inside the blocks the stream itself is never
          normed, only the copy each step reads.
        </p>
      </details>
    </div>
  );
}
