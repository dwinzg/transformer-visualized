import { useCallback, useEffect, useRef, useState } from 'react';
import {
  chosenTokens,
  describeGuesses,
  graft,
  guessesAt,
  MAX_NEW_TOKENS,
  sentenceDone,
  spoken,
  type GuessTree,
} from '../../lib/guess-tree';
import { prefersReducedMotion } from '../../lib/motion';
import { ProbabilityBars } from './ProbabilityBars';
import { TokenRow } from './TokenRow';
import { useTravel } from './useTravel';

const STEP_MS = 1800;
const END_MS = 2400;
export default function HomeDemo({ trees: builtTrees }: { trees: GuessTree[] }) {
  // The built trees end a few guesses in, except along the first guesses. Picks past that are
  // grown here with the real model, so any sentence can be finished.
  const [trees, setTrees] = useState(builtTrees);
  const [live, setLive] = useState<'idle' | 'loading' | 'failed'>('idle');
  const [sentence, setSentence] = useState(0);
  const [path, setPath] = useState<number[]>([]);
  const [playing, setPlaying] = useState(true);
  const [held, setHeld] = useState(false); // off screen or hidden tab
  const [hovered, setHovered] = useState(false);
  const [focused, setFocused] = useState(false);
  const [announcement, setAnnouncement] = useState('');
  const root = useRef<HTMLDivElement>(null);
  const startOver = useRef<HTMLButtonElement>(null);
  const focusStartOver = useRef(false);
  const remember = useTravel(root, path);

  const tree = trees[sentence];
  const tokens = chosenTokens(tree, path);
  const done = path.length >= MAX_NEW_TOKENS || (path.length > 0 && sentenceDone(tokens));
  const guesses = done ? [] : guessesAt(tree, path);
  const needMore = !done && guesses.length === 0;

  // Fetch the next guesses when a pick goes past the built ones.
  useEffect(() => {
    if (!needMore) return;
    let current = true;
    setLive('loading');
    // Loaded on first need, so the home page stays light for readers who only watch.
    import('../../lib/live-guesses')
      .then(({ liveGuesses }) => liveGuesses(tokens.map((t) => t.id)))
      .then(
        (next) => {
          if (!current) return;
          setTrees((all) => all.map((t, i) => (i === sentence ? graft(t, path, next) : t)));
          setLive('idle');
          setAnnouncement(`Next guesses: ${describeGuesses(next)}.`);
        },
        () => current && setLive('failed'),
      );
    return () => {
      current = false;
    };
    // tokens follows from sentence and path.
  }, [needMore, sentence, path]);

  useEffect(() => {
    if (prefersReducedMotion()) setPlaying(false);
  }, []);

  // Remember where the picked bar was, so its word can travel into the sentence.
  const pick = useCallback(
    (index: number, byReader: boolean) => {
      remember(index);
      const nextPath = [...path, index];
      setPath(nextPath);
      if (byReader) {
        setPlaying(false);
        const word = guesses[index] ? spoken(guesses[index].token) : '';
        const ended =
          nextPath.length >= MAX_NEW_TOKENS || sentenceDone([...tokens, guesses[index].token]);
        const after = guessesAt(tree, nextPath);
        if (ended) focusStartOver.current = true;
        setAnnouncement(
          ended
            ? `Added '${word}'. The sentence is done.`
            : after.length > 0
              ? `Added '${word}'. Next guesses: ${describeGuesses(after)}.`
              : `Added '${word}'.`,
        );
      }
    },
    [guesses, path, remember, tree, tokens],
  );

  // Autoplay. Hover, focus, off-screen and a hidden tab all hold the timer.
  useEffect(() => {
    // While the model works out new guesses, there is nothing to play yet.
    if (!playing || held || hovered || focused || needMore) return;
    const timer = setTimeout(
      () => {
        if (guesses.length > 0) {
          pick(0, false);
        } else {
          setPath([]);
          setSentence((s) => (s + 1) % trees.length);
        }
      },
      guesses.length > 0 ? STEP_MS : END_MS,
    );
    return () => clearTimeout(timer);
  }, [playing, held, hovered, focused, needMore, guesses.length, pick, trees.length]);

  // Hold autoplay while the demo is off screen or the tab is hidden.
  useEffect(() => {
    const el = root.current;
    if (!el) return;
    let visible = true;
    const update = () => setHeld(!visible || document.hidden);
    const observer = new IntersectionObserver(
      ([entry]) => {
        visible = entry.intersectionRatio >= 0.3;
        update();
      },
      { threshold: [0, 0.3, 1] },
    );
    observer.observe(el);
    document.addEventListener('visibilitychange', update);
    return () => {
      observer.disconnect();
      document.removeEventListener('visibilitychange', update);
    };
  }, []);

  // The guess list goes away after the reader's last pick, so keep keyboard focus in the demo.
  useEffect(() => {
    if (!focusStartOver.current) return;
    focusStartOver.current = false;
    startOver.current?.focus();
  });

  const restart = () => {
    if (path.length === 0) return;
    setPath([]);
    setAnnouncement('Back to the start.');
  };

  return (
    <div
      ref={root}
      className="home-demo"
      data-home-demo
      onMouseEnter={(e) => {
        // The play/pause toggle controls autoplay directly, so the pointer landing there (as it
        // does right after a click) should not itself hold the timer back.
        if (!(e.target as HTMLElement).closest('.demo-toggle')) setHovered(true);
      }}
      onMouseLeave={() => setHovered(false)}
      onFocus={(e) => {
        // Same reasoning as onMouseEnter above: focus on the toggle itself does not pause.
        if (!(e.target as HTMLElement).closest('.demo-toggle')) setFocused(true);
      }}
      onBlur={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setFocused(false);
      }}
    >
      <div className="demo-controls">
        <button
          ref={startOver}
          type="button"
          className="figure-button press"
          aria-disabled={path.length === 0}
          onClick={restart}
        >
          Start over
        </button>
        <button
          type="button"
          className="figure-button demo-toggle press"
          aria-label={playing ? 'Pause demo' : 'Play demo'}
          onClick={() => {
            // Pressing Play means go now, even though the pointer and focus are inside the demo.
            if (!playing) {
              setHovered(false);
              setFocused(false);
            }
            setPlaying((p) => !p);
          }}
        >
          <span aria-hidden="true">{playing ? 'Pause' : 'Play'}</span>
        </button>
      </div>
      <TokenRow tokens={tokens} caret={guesses.length > 0} newFrom={tree.tokens.length} />
      {guesses.length > 0 ? (
        <ProbabilityBars
          guesses={guesses}
          label="The model's guesses for what comes next"
          onPick={(i) => pick(i, true)}
        />
      ) : needMore && live !== 'failed' ? (
        <p className="demo-status">
          Loading the model to keep going, about 5 MB. This happens once.
        </p>
      ) : (
        <div className="demo-end">
          {needMore && (
            <p className="demo-status">The model did not load, so the demo stops here.</p>
          )}
          <button
            type="button"
            className="figure-button press"
            onClick={() => {
              setPath([]);
              setSentence((s) => (s + 1) % trees.length);
              setAnnouncement('');
              setLive('idle');
            }}
          >
            Next sentence
          </button>
        </div>
      )}
      <p className="visually-hidden" aria-live="polite">
        {announcement}
      </p>
    </div>
  );
}
