import { useId, useState } from 'react';
import type { DisplayToken } from '../../lib/guess-tree';
import { shownToken, spokenToken } from '../../lib/token-text';
import { ChipPicker } from './ChipPicker';
import './figures.css';

/** How many of the strongest neurons to list. */
const SHOWN = 8;

/** The feed-forward layer's neurons after GELU, for one token and one layer at a time. */
export default function FeedForwardFigure({
  tokens,
  activations,
}: {
  tokens: DisplayToken[];
  /** [layer][token][neuron] */
  activations: number[][][];
}) {
  const [layer, setLayer] = useState(0);
  const [row, setRow] = useState(tokens.length - 1);
  const id = useId();
  const values = activations[layer][row];
  const strongest = values
    .map((value, neuron) => ({ value, neuron }))
    .sort((a, b) => b.value - a.value)
    .slice(0, SHOWN);
  const on = values.filter((v) => v > 0).length;
  const max = Math.max(strongest[0].value, 1e-9);

  return (
    <div className="ffn-figure">
      <fieldset className="segmented">
        <legend className="attn-legend">Layer</legend>
        {activations.map((_, n) => (
          <label key={n}>
            <input
              type="radio"
              name={`${id}-layer`}
              checked={layer === n}
              onChange={() => setLayer(n)}
            />
            <span>{n + 1}</span>
          </label>
        ))}
      </fieldset>
      <ChipPicker label="Pick a token" tokens={tokens} selected={row} onSelect={setRow} />
      <p className="ffn-summary" aria-live="polite">
        In layer {layer + 1}, <span aria-hidden="true">{shownToken(tokens[row].text)}</span>
        <span className="visually-hidden">{spokenToken(tokens[row].text)}</span> turns on {on} of{' '}
        {values.length} neurons. The strongest are below.
      </p>
      <ul className="prob-bars" aria-label="The strongest neurons">
        {strongest.map(({ value, neuron }) => (
          <li
            key={neuron}
            className="prob-bar"
            style={{ '--p': `${Math.max((value / max) * 100, 0.5)}%` } as React.CSSProperties}
          >
            <span className="prob-fill" aria-hidden="true" />
            <span className="prob-word">Neuron {neuron + 1}</span>
            <span className="prob-value">{value.toFixed(2)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
