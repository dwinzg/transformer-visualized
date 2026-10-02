import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import {
  chosenTokens,
  describeGuesses,
  guessesAt,
  spoken,
  type GuessTree,
} from '../../lib/guess-tree';
import { prefersReducedMotion, travel, type Box } from '../../lib/motion';
import { ProbabilityBars } from './ProbabilityBars';
import { TokenRow } from './TokenRow';

const STEP_MS = 1800;
const END_MS = 2400;

export default function HomeDemo({ trees }: { trees: GuessTree[] }) {
  const [sentence, setSentence] = useState(0);
  const [path, setPath] = useState<number[]>([]);
  const [playing, setPlaying] = useState(true);
  const [held, setHeld] = useState(false); // off screen or hidden tab
  const [hovered, setHovered] = useState(false);
  const [focused, setFocused] = useState(false);
  const [announcement, setAnnouncement] = useState('');
  const root = useRef<HTMLDivElement>(null);
  const fromBox = useRef<Box | null>(null);

  const tree = trees[sentence];
  const guesses = guessesAt(tree, path);
  const tokens = chosenTokens(tree, path);

  useEffect(() => {
    if (prefersReducedMotion()) setPlaying(false);
  }, []);

  // Remember where the picked bar was, so its word can travel into the sentence.
  const pick = useCallback(
    (index: number, byReader: boolean) => {
      const bar = root.current?.querySelectorAll<HTMLElement>('.prob-bar .prob-word')[index];
      fromBox.current = bar ? bar.getBoundingClientRect() : null;
      const nextPath = [...path, index];
      setPath(nextPath);
      if (byReader) {
        setPlaying(false);
        const word = guesses[index] ? spoken(guesses[index].token) : '';
        const after = guessesAt(tree, nextPath);
        setAnnouncement(
          after.length > 0
            ? `Added '${word}'. Next guesses: ${describeGuesses(after)}.`
            : `Added '${word}'. That is as far as this demo goes.`,
        );
      }
    },
    [guesses, path, tree],
  );

  useLayoutEffect(() => {
    if (!fromBox.current || !root.current) return;
    const chips = root.current.querySelectorAll<HTMLElement>('.token-chip');
    const last = chips[chips.length - 1];
    if (last) travel(last, fromBox.current);
    fromBox.current = null;
  }, [path]);

  // Autoplay. Hover, focus, off-screen and a hidden tab all hold the timer.
  useEffect(() => {
    if (!playing || held || hovered || focused) return;
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
  }, [playing, held, hovered, focused, guesses.length, pick, trees.length]);

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

  const paused = !playing || held || hovered || focused;

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
        {path.length > 0 && guesses.length > 0 && (
          <button type="button" className="demo-reset press" onClick={() => setPath([])}>
            Start over
          </button>
        )}
        <button
          type="button"
          className="demo-toggle press"
          aria-label={playing ? 'Pause demo' : 'Play demo'}
          onClick={() => setPlaying((p) => !p)}
        >
          <span aria-hidden="true">{playing ? 'Pause' : 'Play'}</span>
        </button>
      </div>
      <TokenRow tokens={tokens} caret={guesses.length > 0} newFrom={tree.tokens.length} />
      {guesses.length > 0 ? (
        <ProbabilityBars
          guesses={guesses}
          label="The model's next word guesses"
          onPick={(i) => pick(i, true)}
        />
      ) : (
        <div className="demo-end">
          <button type="button" className="press" onClick={() => setPath([])}>
            Start over
          </button>
          <button
            type="button"
            className="press"
            onClick={() => {
              setPath([]);
              setSentence((s) => (s + 1) % trees.length);
            }}
          >
            Next sentence
          </button>
        </div>
      )}
      <p className="visually-hidden" aria-live="polite">
        {announcement}
      </p>
      <span className="visually-hidden">{paused ? 'Demo paused.' : ''}</span>
    </div>
  );
}
