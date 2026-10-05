import { useState } from 'react';
import {
  TOY_K,
  TOY_Q,
  TOY_STAGES,
  TOY_TOKENS,
  TOY_V,
  toyAttention,
  type ToyStage,
} from '../../lib/toy-attention';
import './figures.css';

const toy = toyAttention();

const STAGE_TEXT: Record<ToyStage, { title: string; text: string }> = {
  scores: {
    title: 'Match questions with name tags',
    text: "Each token's question is compared with every name tag using a dot product. A big score means a good match.",
  },
  scaled: {
    title: 'Scale the scores down',
    text: 'Each score is divided by the square root of 2, because each list has 2 numbers. This keeps scores from growing too big when lists are long.',
  },
  masked: {
    title: 'Hide the future',
    text: 'A token may not look at tokens that come after it, so those scores are hidden.',
  },
  weights: {
    title: 'Turn scores into weights',
    text: 'Softmax turns each row into weights between 0 and 1 that add up to 1.',
  },
  output: {
    title: 'Blend the backpacks',
    text: 'Each token fills a new backpack with a mix of the backpacks it can see, its own included, using its weights.',
  },
};

const fmt = (value: number) => (Number.isInteger(value) ? String(value) : value.toFixed(2));
const list = (values: number[]) => `(${values.map(fmt).join(', ')})`;

function cell(stage: ToyStage, i: number, j: number): string {
  if (stage === 'scores') return fmt(toy.scores[i][j]);
  if (stage === 'scaled') return fmt(toy.scaled[i][j]);
  if (j > i) return 'hidden';
  return stage === 'masked' ? fmt(toy.scaled[i][j]) : fmt(toy.weights[i][j]);
}

/** Three tokens with two numbers each, stepped through attention one operation at a time. */
export default function ToyAttentionFigure() {
  const [index, setIndex] = useState(0);
  const stage = TOY_STAGES[index];
  const last = TOY_STAGES.length - 1;
  const showWeights = stage === 'weights' || stage === 'output';

  return (
    <div className="toy-figure">
      <p className="toy-label">A made-up example with small numbers.</p>
      <ul className="toy-cards" aria-label="The three tokens">
        {TOY_TOKENS.map((token, i) => (
          <li key={token} className="toy-card">
            <span className="token-chip">{token}</span>
            <span className="toy-q">question {list(TOY_Q[i])}</span>
            <span className="toy-k">name tag {list(TOY_K[i])}</span>
            <span className="toy-v">backpack {list(TOY_V[i])}</span>
          </li>
        ))}
      </ul>
      <div className="toy-controls">
        <button
          type="button"
          className="figure-button press"
          aria-disabled={index === 0}
          onClick={() => index > 0 && setIndex(index - 1)}
        >
          Previous
        </button>
        <button
          type="button"
          className="figure-button press"
          aria-disabled={index === last}
          onClick={() => index < last && setIndex(index + 1)}
        >
          Next
        </button>
        <button
          type="button"
          className="figure-button press"
          aria-disabled={index === 0}
          onClick={() => setIndex(0)}
        >
          Start over
        </button>
      </div>
      <div aria-live="polite">
        <h3 className="toy-title">
          Step {index + 1} of {TOY_STAGES.length}. {STAGE_TEXT[stage].title}
        </h3>
        <p className="toy-text">{STAGE_TEXT[stage].text}</p>
      </div>
      <table className="toy-table">
        <caption className="visually-hidden">
          Rows are the token asking. Columns are the token being looked at.
        </caption>
        <thead>
          <tr>
            <th scope="col">
              <span className="visually-hidden">Asking token</span>
            </th>
            {TOY_TOKENS.map((token) => (
              <th key={token} scope="col">
                {token}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {TOY_TOKENS.map((token, i) => (
            <tr key={token}>
              <th scope="row">{token}</th>
              {TOY_TOKENS.map((other, j) => (
                <td
                  key={other}
                  className={j > i && index >= 2 ? 'is-hidden' : undefined}
                  style={
                    showWeights && j <= i
                      ? {
                          background: `color-mix(in srgb, var(--concept-query) ${Math.round(toy.weights[i][j] * 50)}%, var(--color-surface))`,
                        }
                      : undefined
                  }
                >
                  {cell(stage, i, j)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {stage === 'output' && (
        <ul className="toy-output" aria-label="New backpacks">
          {TOY_TOKENS.map((token, i) => (
            <li key={token}>
              <strong>{token}</strong>{' '}
              {toy.weights[i]
                .map((w, j) => (j > i ? null : `${fmt(w)} × ${TOY_TOKENS[j]}`))
                .filter(Boolean)
                .join(' + ')}{' '}
              = {list(toy.output[i])}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
