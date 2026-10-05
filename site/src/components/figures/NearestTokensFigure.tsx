import { useState } from 'react';
import type { NearestTokens } from '../../lib/embeddings';
import { shownToken, spokenToken } from '../../lib/token-text';
import { ChipPicker } from './ChipPicker';
import './figures.css';

/** Pick a word and see the tokens whose numbers are most like its own. */
export default function NearestTokensFigure({ words }: { words: NearestTokens[] }) {
  const [picked, setPicked] = useState(0);
  const { word, neighbors } = words[picked];

  return (
    <div className="nearest-figure">
      <div className="strip-controls">
        <ChipPicker
          label="Pick a word"
          tokens={words.map((w) => w.word)}
          selected={picked}
          onSelect={setPicked}
        />
        <button
          type="button"
          className="figure-button press"
          aria-disabled={picked === 0}
          onClick={() => setPicked(0)}
        >
          Start over
        </button>
      </div>
      <p className="nearest-title" aria-live="polite">
        Closest to <strong aria-hidden="true">{shownToken(word.text)}</strong>
        <span className="visually-hidden">
          {`${spokenToken(word.text)}. ${neighbors.map((n) => `${spokenToken(n.token.text)} ${n.score.toFixed(2)}`).join(', ')}`}
        </span>
      </p>
      <ol className="nearest-list" aria-label={`Tokens closest to ${spokenToken(word.text)}`}>
        {neighbors.map((n) => (
          <li key={n.token.id} className="nearest-row">
            <span className="token-chip">
              <span aria-hidden="true">{shownToken(n.token.text)}</span>
              <span className="visually-hidden">{spokenToken(n.token.text)}</span>
            </span>
            <span className="nearest-bar" aria-hidden="true">
              <span style={{ width: `${Math.max(n.score, 0) * 100}%` }} />
            </span>
            <span className="nearest-score">{n.score.toFixed(2)}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}
