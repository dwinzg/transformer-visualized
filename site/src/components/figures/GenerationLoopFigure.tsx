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
  const done = guesses.length === 0;

  const addOne = useCallback(() => {
    if (guesses.length === 0) return;
    remember(0);
    setPath((p) => [...p, 0]);
    setAnnouncement(`Added '${spoken(guesses[0].token)}'.`);
  }, [guesses, remember]);

  // Play keeps stepping every STEP_MS until the tree runs out of guesses.
  useEffect(() => {
    if (!playing) return;
    if (done) {
      setPlaying(false);
      return;
    }
    const timer = setTimeout(addOne, STEP_MS);
    return () => clearTimeout(timer);
  }, [playing, done, addOne]);

  // aria-disabled rather than disabled, so a focused button keeps focus when it runs out.
  return (
    <div ref={root} className="generation-loop-figure">
      <div className="loop-controls">
        <button type="button" className="figure-button press" aria-disabled={done} onClick={addOne}>
          Add a guess
        </button>
        <button
          type="button"
          className="figure-button press"
          aria-disabled={done && !playing}
          onClick={() => {
            if (!done || playing) setPlaying((p) => !p);
          }}
        >
          {playing ? 'Pause' : 'Play'}
        </button>
        <button
          type="button"
          className="figure-button press"
          aria-disabled={path.length === 0}
          onClick={() => {
            if (path.length === 0) return;
            setPlaying(false);
            setPath([]);
            setAnnouncement('Back to the start.');
          }}
        >
          Start over
        </button>
      </div>
      <TokenRow tokens={tokens} caret={!done} newFrom={tree.tokens.length} />
      <ProbabilityBars guesses={guesses} label="The model's guesses for what comes next" />
      <p className="guess-count">Guesses made: {path.length}</p>
      <p className="visually-hidden" aria-live="polite">
        {announcement}
      </p>
    </div>
  );
}
