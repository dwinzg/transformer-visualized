import { useId, useState } from 'react';
import { describeGuesses, type DisplayToken, type Guess } from '../../lib/guess-tree';
import { ProbabilityBars } from './ProbabilityBars';
import { TokenRow } from './TokenRow';
import './figures.css';

interface Item {
  label: string;
  tokens: DisplayToken[];
  guesses: Guess[];
}

/** Two or more texts, one at a time, each with the model's likeliest next tokens. */
export default function CompareGuessesFigure({ items }: { items: Item[] }) {
  const [shown, setShown] = useState(0);
  const [announcement, setAnnouncement] = useState('');
  const id = useId();
  const item = items[shown];
  return (
    <div className="compare-figure">
      <fieldset className="segmented">
        <legend className="visually-hidden">Which text</legend>
        {items.map((it, i) => (
          <label key={i}>
            <input
              type="radio"
              name={id}
              checked={shown === i}
              onChange={() => {
                setShown(i);
                setAnnouncement(`${it.label}. Top guesses: ${describeGuesses(it.guesses)}.`);
              }}
            />
            <span>{it.label}</span>
          </label>
        ))}
      </fieldset>
      <TokenRow tokens={item.tokens} caret />
      <ProbabilityBars guesses={item.guesses} label={`Next guesses, ${item.label.toLowerCase()}`} />
      <p className="visually-hidden" aria-live="polite">
        {announcement}
      </p>
    </div>
  );
}
