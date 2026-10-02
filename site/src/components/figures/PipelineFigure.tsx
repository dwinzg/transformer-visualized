import { useEffect, useRef, useState, type KeyboardEvent } from 'react';
import { prefersReducedMotion } from '../../lib/motion';
import './figures.css';

interface Stage {
  id: string;
  label: string;
  /** One of the site's concept colors, see figures.css. */
  color: 'neutral' | 'position' | 'residual' | 'output';
  text: string;
  /** Key into `links`, for the stages whose own chapter may exist. */
  linkKey?: 'tokens' | 'embeddings' | 'output';
}

const STAGES: Stage[] = [
  {
    id: 'tokens',
    label: 'Tokens',
    color: 'neutral',
    text: 'The text is split into tokens. Each token has an id number.',
    linkKey: 'tokens',
  },
  {
    id: 'embeddings',
    label: 'Embeddings and position',
    color: 'position',
    text: 'Each id becomes a list of 128 numbers. Its position is mixed in.',
    linkKey: 'embeddings',
  },
  {
    id: 'block-1',
    label: 'Block 1',
    color: 'residual',
    text: 'Each token looks at the tokens before it and updates its numbers.',
  },
  {
    id: 'block-2',
    label: 'Block 2',
    color: 'residual',
    text: 'Each token looks at the tokens before it and updates its numbers.',
  },
  {
    id: 'block-3',
    label: 'Block 3',
    color: 'residual',
    text: 'Each token looks at the tokens before it and updates its numbers.',
  },
  {
    id: 'block-4',
    label: 'Block 4',
    color: 'residual',
    text: 'Each token looks at the tokens before it and updates its numbers.',
  },
  {
    id: 'output',
    label: 'Output',
    color: 'output',
    text: "The last token's numbers become a chance for every possible next token.",
    linkKey: 'output',
  },
];

interface Props {
  links?: Partial<Record<'tokens' | 'embeddings' | 'output', string>>;
}

/** The transformer's stages as a single-select list, from tokens in to a prediction out. */
export default function PipelineFigure({ links = {} }: Props) {
  const [selected, setSelected] = useState<number | null>(null);
  const [animate, setAnimate] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);
  const buttonRefs = useRef<(HTMLButtonElement | null)[]>([]);

  useEffect(() => {
    if (prefersReducedMotion()) return;
    const el = listRef.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) return;
        setAnimate(true);
        observer.disconnect();
      },
      { threshold: 0.3 },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const move = (event: KeyboardEvent<HTMLDivElement>) => {
    const current = selected ?? 0;
    const last = STAGES.length - 1;
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
    if (next === -1) return;
    event.preventDefault();
    setSelected(next);
    buttonRefs.current[next]?.focus();
  };

  const active = STAGES[selected ?? 0];
  const href = active.linkKey ? links[active.linkKey] : undefined;

  return (
    <div className="pipeline-figure">
      <div
        ref={listRef}
        className={`pipeline-stages${animate ? ' is-playing' : ''}`}
        role="radiogroup"
        aria-label="The transformer's stages"
        onKeyDown={move}
      >
        {STAGES.map((stage, i) => (
          <button
            key={stage.id}
            ref={(el) => {
              buttonRefs.current[i] = el;
            }}
            type="button"
            role="radio"
            aria-checked={i === selected}
            tabIndex={i === (selected ?? 0) ? 0 : -1}
            className={`pipeline-stage press concept-${stage.color}`}
            onClick={() => setSelected(i)}
          >
            {stage.label}
          </button>
        ))}
      </div>
      <p className="pipeline-explain" aria-live="polite">
        {selected !== null && (
          <>
            {active.text}
            {href && (
              <>
                {' '}
                <a href={href}>Read the {active.label} chapter.</a>
              </>
            )}
          </>
        )}
      </p>
      <button
        type="button"
        className="figure-button press"
        aria-disabled={selected === null}
        onClick={() => {
          if (selected === null) return;
          setSelected(null);
          buttonRefs.current[0]?.focus();
        }}
      >
        Start over
      </button>
    </div>
  );
}
