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

/** A reader-driven version of the home demo: pick a guess, see it added, start over at any point. */
export default function NextWordFigure({ tree }: { tree: GuessTree }) {
  const [path, setPath] = useState<number[]>([]);
  const [announcement, setAnnouncement] = useState('');
  // Starts at the server's value so hydration matches. The stored depth is read after mount.
  const [depth, setDepth] = useState<Depth>('story');
  const root = useRef<HTMLDivElement>(null);
  const startOver = useRef<HTMLButtonElement>(null);
  const focusStartOver = useRef(false);
  const remember = useTravel(root, path);

  useEffect(() => {
    setDepth(readStoredDepth(browserStorage()));
    const onChange = (event: Event) => {
      const detail = (event as CustomEvent<unknown>).detail;
      if (isDepth(detail)) setDepth(detail);
    };
    document.addEventListener(DEPTH_EVENT, onChange);
    return () => document.removeEventListener(DEPTH_EVENT, onChange);
  }, []);

  // The guess list goes away after the last pick, so keep keyboard focus in the figure.
  useEffect(() => {
    if (!focusStartOver.current) return;
    focusStartOver.current = false;
    startOver.current?.focus();
  });

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
    if (after.length === 0) focusStartOver.current = true;
    setAnnouncement(
      after.length > 0
        ? `Added '${word}'. Next guesses: ${describeGuesses(after)}.`
        : `Added '${word}'. That is as far as this demo goes.`,
    );
  };

  return (
    <div ref={root} className="next-word-figure">
      <button
        ref={startOver}
        type="button"
        className="figure-button press"
        aria-disabled={path.length === 0}
        onClick={() => {
          if (path.length === 0) return;
          setPath([]);
          setAnnouncement('Back to the start.');
        }}
      >
        Start over
      </button>
      <TokenRow tokens={tokens} caret={guesses.length > 0} newFrom={tree.tokens.length} />
      <ProbabilityBars
        guesses={guesses}
        label="The model's guesses for what comes next"
        onPick={pick}
        exact={exact}
      />
      {showFormula && <p className="prob-note">P(next token | tokens so far)</p>}
      <p className="visually-hidden" aria-live="polite">
        {announcement}
      </p>
    </div>
  );
}
