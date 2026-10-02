import { useEffect, useRef, useState } from 'react';
import {
  browserStorage,
  DEPTH_EVENT,
  DEPTHS,
  isDepth,
  readStoredDepth,
  type Depth,
} from '../../lib/depth';
import {
  chosenTokens,
  describeGuesses,
  guessesAt,
  spoken,
  type GuessTree,
} from '../../lib/guess-tree';
import { ProbabilityBars } from './ProbabilityBars';
import { TokenRow } from './TokenRow';
import { useTravel } from './useTravel';

/** A reader-driven version of the home demo: pick a guess, see it added, reset at any point. */
export default function NextWordFigure({ tree }: { tree: GuessTree }) {
  const [path, setPath] = useState<number[]>([]);
  const [announcement, setAnnouncement] = useState('');
  const [depth, setDepth] = useState<Depth>(() => readStoredDepth(browserStorage()));
  const root = useRef<HTMLDivElement>(null);
  const remember = useTravel(root, path);

  useEffect(() => {
    const onChange = (event: Event) => {
      const detail = (event as CustomEvent<unknown>).detail;
      if (isDepth(detail)) setDepth(detail);
    };
    document.addEventListener(DEPTH_EVENT, onChange);
    return () => document.removeEventListener(DEPTH_EVENT, onChange);
  }, []);

  const guesses = guessesAt(tree, path);
  const tokens = chosenTokens(tree, path);
  const exact = DEPTHS.indexOf(depth) >= DEPTHS.indexOf('numbers');
  const showFormula = depth === 'formula' || depth === 'code';

  const pick = (index: number) => {
    remember(index);
    const nextPath = [...path, index];
    setPath(nextPath);
    const word = guesses[index] ? spoken(guesses[index].token) : '';
    const after = guessesAt(tree, nextPath);
    setAnnouncement(
      after.length > 0
        ? `Added '${word}'. Next guesses: ${describeGuesses(after)}.`
        : `Added '${word}'. That is as far as this demo goes.`,
    );
  };

  return (
    <div ref={root} className="next-word-figure">
      {path.length > 0 && (
        <button type="button" className="figure-reset press" onClick={() => setPath([])}>
          {guesses.length > 0 ? 'Reset' : 'Start over'}
        </button>
      )}
      <TokenRow tokens={tokens} caret={guesses.length > 0} newFrom={tree.tokens.length} />
      {guesses.length > 0 && (
        <ProbabilityBars
          guesses={guesses}
          label="The model's next word guesses"
          onPick={pick}
          exact={exact}
        />
      )}
      {showFormula && <p className="prob-note">P(next token | tokens so far)</p>}
      <p className="visually-hidden" aria-live="polite">
        {announcement}
      </p>
    </div>
  );
}
