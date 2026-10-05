import { useRef, type KeyboardEvent } from 'react';
import type { DisplayToken } from '../../lib/guess-tree';
import './figures.css';

interface Props {
  label: string;
  tokens: readonly DisplayToken[];
  selected: number;
  onSelect: (index: number) => void;
}

/** A space at the start of a token is part of it, so it is drawn as a dot. */
export const shownToken = (text: string) => text.replace(/ /g, '\u00b7');

/** Token chips that work as one radio group: arrow keys move, and the picked chip is checked. */
export function ChipPicker({ label, tokens, selected, onSelect }: Props) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);

  const move = (event: KeyboardEvent<HTMLDivElement>) => {
    const last = tokens.length - 1;
    const next =
      event.key === 'ArrowRight' || event.key === 'ArrowDown'
        ? Math.min(selected + 1, last)
        : event.key === 'ArrowLeft' || event.key === 'ArrowUp'
          ? Math.max(selected - 1, 0)
          : event.key === 'Home'
            ? 0
            : event.key === 'End'
              ? last
              : -1;
    if (next === -1) return;
    event.preventDefault();
    onSelect(next);
    refs.current[next]?.focus();
  };

  return (
    <div className="chip-picker" role="radiogroup" aria-label={label} onKeyDown={move}>
      {tokens.map((token, i) => (
        <button
          key={`${i}-${token.id}`}
          ref={(el) => {
            refs.current[i] = el;
          }}
          type="button"
          role="radio"
          aria-checked={i === selected}
          aria-label={token.text.trim()}
          tabIndex={i === selected ? 0 : -1}
          className="token-chip chip-option press"
          onClick={() => onSelect(i)}
        >
          {shownToken(token.text)}
        </button>
      ))}
    </div>
  );
}
