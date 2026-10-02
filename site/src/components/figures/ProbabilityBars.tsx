import { useEffect, useId, useState, type KeyboardEvent } from 'react';
import { percent, type Guess } from '../../lib/guess-tree';
import './figures.css';

interface Props {
  guesses: readonly Guess[];
  label: string;
  onPick?: (index: number) => void;
  exact?: boolean;
}

/** The model's top guesses as bars. With onPick, it is a listbox the reader can choose from. */
export function ProbabilityBars({ guesses, label, onPick, exact = false }: Props) {
  const id = useId();
  const [active, setActive] = useState(0);
  // guessesAt returns the same array reference for the same tree node, and a new one whenever
  // the path changes, so this fires exactly when the reader picks a new guess: without it, a
  // keyboard user who picked option 2 would land on option 2 of the next list instead of the
  // top guess.
  useEffect(() => setActive(0), [guesses]);
  const value = (p: number) => (exact ? `${(p * 100).toFixed(1)}%` : percent(p));
  const fill = (p: number) => ({ '--p': `${Math.max(p * 100, 1)}%` }) as React.CSSProperties;
  const wordClass = (g: Guess) => `prob-word${g.token.special ? ' is-special' : ''}`;

  if (guesses.length === 0) return null;

  if (!onPick) {
    return (
      <ul className="prob-bars" aria-label={label}>
        {guesses.map((g, i) => (
          // Keyed by rank, so each bar's width grows or shrinks from the last list's value.
          <li key={i} className="prob-bar" style={fill(g.p)}>
            <span className="prob-fill" aria-hidden="true" />
            <span className={wordClass(g)}>{g.token.text}</span>
            <span className="prob-value">{value(g.p)}</span>
          </li>
        ))}
      </ul>
    );
  }

  const current = Math.min(active, guesses.length - 1);
  const onKeyDown = (event: KeyboardEvent<HTMLUListElement>) => {
    const last = guesses.length - 1;
    const next =
      event.key === 'ArrowDown'
        ? Math.min(current + 1, last)
        : event.key === 'ArrowUp'
          ? Math.max(current - 1, 0)
          : event.key === 'Home'
            ? 0
            : event.key === 'End'
              ? last
              : -1;
    if (next !== -1) {
      event.preventDefault();
      setActive(next);
    } else if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onPick(current);
    }
  };

  return (
    <ul
      className="prob-bars"
      role="listbox"
      aria-label={label}
      tabIndex={0}
      aria-activedescendant={`${id}-${current}`}
      onKeyDown={onKeyDown}
    >
      {guesses.map((g, i) => (
        <li
          key={i}
          id={`${id}-${i}`}
          role="option"
          aria-selected={i === current}
          data-active={i === current}
          className="prob-bar press"
          style={fill(g.p)}
          onClick={() => {
            setActive(i);
            onPick(i);
          }}
        >
          <span className="prob-fill" aria-hidden="true" />
          <span className={wordClass(g)}>{g.token.text}</span>
          <span className="prob-value">{value(g.p)}</span>
        </li>
      ))}
    </ul>
  );
}
