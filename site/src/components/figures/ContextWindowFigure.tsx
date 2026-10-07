import { useId, useState } from 'react';
import { describeGuesses, type Guess } from '../../lib/guess-tree';
import { ProbabilityBars } from './ProbabilityBars';
import './figures.css';

interface Step {
  gap: number;
  total: number;
  seen: number;
  note: { from: number; to: number };
  guesses: Guess[];
}

interface Props {
  window: number;
  steps: Step[];
  note: string;
  question: string;
}

const pct = (n: number, of: number) => `${(n / of) * 100}%`;

/** A note, more and more text after it, and the window of tokens the model can still read. */
export default function ContextWindowFigure({ window, steps, note, question }: Props) {
  const [gap, setGap] = useState(0);
  const id = useId();
  const step = steps[gap];
  const longest = steps[steps.length - 1].total;
  const inside = step.total - step.seen <= step.note.from;
  const reads = step.total <= window ? `all ${step.total}` : `only the last ${window}`;
  const summary = `${step.total} tokens in all. The model can read ${window}, so it reads ${reads}, and the note is ${inside ? 'inside' : 'outside'} its window.`;

  return (
    <div className="window-figure">
      <label className="window-slider" htmlFor={`${id}-gap`}>
        <span>
          Sentences in between: <output htmlFor={`${id}-gap`}>{gap}</output>
        </span>
        <input
          id={`${id}-gap`}
          type="range"
          min={0}
          max={steps.length - 1}
          value={gap}
          aria-valuetext={`${gap} sentences, ${step.total} tokens`}
          onChange={(e) => setGap(Number(e.target.value))}
        />
      </label>
      <p className="window-text">
        <mark className={inside ? 'is-seen' : 'is-lost'}>{note}</mark>
        {gap > 0 && (
          <span className="window-filler">
            {' '}
            (then {gap} {gap === 1 ? 'sentence' : 'sentences'} about the garden)
          </span>
        )}{' '}
        {question}
      </p>
      {/* The whole text as a bar, with the part the model can read marked. */}
      <div className="window-track" aria-hidden="true">
        <div className="window-text-bar" style={{ width: pct(step.total, longest) }}>
          <span
            className={`window-note ${inside ? 'is-seen' : 'is-lost'}`}
            style={{ width: pct(step.note.to - step.note.from, step.total) }}
          />
          <span
            className="window-seen"
            style={{
              left: pct(step.total - step.seen, step.total),
              width: pct(step.seen, step.total),
            }}
          />
        </div>
      </div>
      <p className="window-summary">
        {step.total} tokens in all. The model can read {window}, so it reads {reads}, and the note
        is <strong>{inside ? 'inside' : 'outside'}</strong> its window.
      </p>
      <ProbabilityBars guesses={step.guesses} label="The model's guesses for the next token" />
      <p className="visually-hidden" aria-live="polite">
        {gap > 0 ? `${summary} Top guesses: ${describeGuesses(step.guesses)}.` : ''}
      </p>
    </div>
  );
}
