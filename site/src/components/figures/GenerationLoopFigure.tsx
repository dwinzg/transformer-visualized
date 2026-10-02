import { useCallback, useEffect, useRef, useState } from 'react';
import { chosenTokens, guessesAt, spoken, type GuessTree } from '../../lib/guess-tree';
import { ProbabilityBars } from './ProbabilityBars';
import { TokenRow } from './TokenRow';
import { useTravel } from './useTravel';

const STEP_MS = 1200;

/** Shows the model writing a few words on its own, one guess at a time. */
export default function GenerationLoopFigure({ tree }: { tree: GuessTree }) {
  const [path, setPath] = useState<number[]>([]);
  const [playing, setPlaying] = useState(false);
  const [announcement, setAnnouncement] = useState('');
  const root = useRef<HTMLDivElement>(null);
  const remember = useTravel(root, path);

  const guesses = guessesAt(tree, path);
  const tokens = chosenTokens(tree, path);

  const addOne = useCallback(() => {
    remember(0);
    const word = guesses[0] ? spoken(guesses[0].token) : '';
    setPath((p) => [...p, 0]);
    setAnnouncement(`Added '${word}'.`);
  }, [guesses, remember]);

  // Play keeps stepping every STEP_MS until the tree runs out of guesses.
  useEffect(() => {
    if (!playing) return;
    if (guesses.length === 0) {
      setPlaying(false);
      return;
    }
    const timer = setTimeout(addOne, STEP_MS);
    return () => clearTimeout(timer);
  }, [playing, guesses, addOne]);

  return (
    <div ref={root} className="generation-loop-figure">
      <div className="loop-controls">
        <button type="button" className="press" disabled={guesses.length === 0} onClick={addOne}>
          Next word
        </button>
        <button
          type="button"
          className="press"
          disabled={guesses.length === 0 || playing}
          onClick={() => setPlaying(true)}
        >
          Play
        </button>
        <button
          type="button"
          className="press"
          onClick={() => {
            setPlaying(false);
            setPath([]);
          }}
        >
          Reset
        </button>
      </div>
      <TokenRow tokens={tokens} caret={guesses.length > 0} newFrom={tree.tokens.length} />
      <ProbabilityBars guesses={guesses} label="The model's next word guesses" />
      <p className="guess-count">Guesses made: {path.length}</p>
      <p className="visually-hidden" aria-live="polite">
        {announcement}
      </p>
    </div>
  );
}
