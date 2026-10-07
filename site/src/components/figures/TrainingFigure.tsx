import { useId, useState } from 'react';
import type { Guess } from '../../lib/guess-tree';
import { logX, strongestHead, tokenCount, type TrainingData } from '../../lib/training';
import { ProbabilityBars } from './ProbabilityBars';
import './figures.css';

interface Props {
  data: TrainingData;
  /** The loss of a blind guess, for the dashed line on the loss chart. */
  chance: number;
}

const W = 320;
const H = 140;
const PAD = { left: 34, right: 10, top: 10, bottom: 22 };

/** A small line chart with the step on a log axis and a dot at the chosen snapshot. */
function Chart({
  steps,
  lines,
  at,
  max,
  label,
  guide,
  format,
}: {
  steps: number[];
  lines: { values: number[]; className: string }[];
  at: number;
  max: number;
  label: string;
  guide?: { value: number; text: string };
  format: (v: number) => string;
}) {
  const last = steps[steps.length - 1];
  const x = (step: number) => PAD.left + logX(step, last) * (W - PAD.left - PAD.right);
  const y = (v: number) => PAD.top + (1 - v / max) * (H - PAD.top - PAD.bottom);
  return (
    <svg className="train-chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
      <line className="axis" x1={PAD.left} x2={W - PAD.right} y1={y(0)} y2={y(0)} />
      <text className="tick" x={PAD.left - 4} y={y(max) + 4} textAnchor="end">
        {format(max)}
      </text>
      <text className="tick" x={PAD.left - 4} y={y(0)} textAnchor="end">
        0
      </text>
      {/* Steps are spaced by powers of ten, so the early changes get room. */}
      {[0, 100, 1000, 10000].map((tick) => (
        <text
          key={tick}
          className="tick"
          x={x(tick)}
          y={H - 4}
          textAnchor={tick ? 'middle' : 'start'}
        >
          {tick ? tick.toLocaleString('en-US') : 'step 0'}
        </text>
      ))}
      {guide && (
        <>
          <line
            className="guide"
            x1={PAD.left}
            x2={W - PAD.right}
            y1={y(guide.value)}
            y2={y(guide.value)}
          />
          <text className="tick" x={W - PAD.right} y={y(guide.value) - 4} textAnchor="end">
            {guide.text}
          </text>
        </>
      )}
      {lines.map((line, i) => (
        <g key={i} className={line.className}>
          <polyline
            fill="none"
            points={steps.map((s, k) => `${x(s)},${y(line.values[k])}`).join(' ')}
          />
          <circle cx={x(steps[at])} cy={y(line.values[at])} r={4.5} />
        </g>
      ))}
    </svg>
  );
}

/** Slide through a training run and watch the model's guesses, stories and skills appear. */
export default function TrainingFigure({ data, chance }: Props) {
  const { snapshots, prompts, tokensPerStep } = data;
  const [at, setAt] = useState(snapshots.length - 1);
  const [announcement, setAnnouncement] = useState('');
  const id = useId();
  const snap = snapshots[at];
  const steps = snapshots.map((s) => s.step);
  const tokens = tokenCount(snap.step * tokensPerStep);
  const where =
    snap.step === 0 ? 'Before any training' : `After ${snap.step.toLocaleString('en-US')} steps`;
  const guesses: Guess[] = snap.next.map((g) => ({
    token: { id: g.id, text: g.text },
    p: g.p,
    next: [],
  }));
  const best = strongestHead(snap.induction);
  // Shaded by how far a score rises above an even spread, against the highest of the whole run.
  const chanceScore = data.inductionChance;
  const top = Math.max(...snapshots.flatMap((s) => s.induction.flat())) - chanceScore;
  const shade = (score: number) => Math.round((Math.max(0, score - chanceScore) / top) * 50);
  const percent = (p: number) => `${Math.round(p * 100)}%`;

  return (
    <div className="train-figure">
      <div className="train-slider">
        <label htmlFor={`${id}-step`}>Training step</label>
        <p className="train-where" aria-hidden="true">
          <strong>{where}</strong>, {tokens} tokens read
        </p>
        <input
          id={`${id}-step`}
          type="range"
          min={0}
          max={snapshots.length - 1}
          value={at}
          aria-valuetext={`${where}, ${tokens} tokens read`}
          onChange={(e) => {
            const next = snapshots[Number(e.target.value)];
            setAt(Number(e.target.value));
            setAnnouncement(
              `Loss ${next.valLoss.toFixed(2)}. Top guess ${next.next[0].text.trim()}.`,
            );
          }}
        />
      </div>

      <section className="train-panel" aria-labelledby={`${id}-loss`}>
        <h2 id={`${id}-loss`}>Loss {snap.valLoss.toFixed(2)}</h2>
        <p>
          How surprised the model is by the next token, on text it never trained on. Lower is
          better.
        </p>
        <Chart
          steps={steps}
          lines={[{ values: snapshots.map((s) => s.valLoss), className: 'line-loss' }]}
          at={at}
          max={Math.ceil(chance)}
          guide={{ value: chance, text: 'guessing blindly' }}
          format={(v) => v.toFixed(0)}
          label={`Loss falls from ${snapshots[0].valLoss.toFixed(2)} at the start to ${snapshots[snapshots.length - 1].valLoss.toFixed(2)} at the end. Now ${snap.valLoss.toFixed(2)}.`}
        />
      </section>

      <section className="train-panel" aria-labelledby={`${id}-story`}>
        <h2 id={`${id}-story`}>A story it writes</h2>
        <p className="train-story">
          <strong>{prompts.story}</strong>
          {snap.story}
        </p>
      </section>

      <section className="train-panel" aria-labelledby={`${id}-next`}>
        <h2 id={`${id}-next`}>Its guesses after "{prompts.next}"</h2>
        <ProbabilityBars guesses={guesses} label={`Guesses after ${prompts.next}`} />
      </section>

      <section className="train-panel" aria-labelledby={`${id}-skills`}>
        <h2 id={`${id}-skills`}>Using what came earlier in the text</h2>
        <ul className="train-skills">
          <li>
            <span className="key line-recall" aria-hidden="true" /> Remembering{' '}
            <strong>{prompts.recall.answer.trim()}</strong> from an earlier sentence,{' '}
            {percent(snap.recall)}
          </li>
          <li>
            <span className="key line-copy" aria-hidden="true" /> Copying the made-up word{' '}
            <strong>{prompts.copy.answer.trim()}</strong>, {percent(snap.copy)}
          </li>
        </ul>
        <Chart
          steps={steps}
          lines={[
            {
              values: snapshots.map((s) => s.recall),
              className: 'line-recall',
            },
            {
              values: snapshots.map((s) => s.copy),
              className: 'line-copy',
            },
          ]}
          at={at}
          max={1}
          format={() => '100%'}
          label={`The chance of each answer over training. Remembering ${prompts.recall.answer.trim()} now ${percent(snap.recall)}, copying ${prompts.copy.answer.trim()} now ${percent(snap.copy)}.`}
        />
      </section>

      <section className="train-panel" aria-labelledby={`${id}-heads`}>
        <h2 id={`${id}-heads`}>Heads that look for what came next last time</h2>
        <p>
          A head that spreads its weight evenly scores about {data.inductionChance.toFixed(2)}.
          Shaded cells score above that.
        </p>
        <div
          className="attn-score-scroll"
          tabIndex={0}
          role="group"
          aria-label="Copying score per head"
        >
          <table className="train-heads">
            <caption className="visually-hidden">
              For each layer and head, how much weight a repeated token puts on the token that
              followed its first copy.
            </caption>
            <thead>
              <tr>
                <td />
                {snap.induction[0].map((_, h) => (
                  <th key={h} scope="col">
                    Head {h + 1}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {snap.induction.map((row, l) => (
                <tr key={l}>
                  <th scope="row">Layer {l + 1}</th>
                  {row.map((score, h) => (
                    <td
                      key={h}
                      className={l === best.layer && h === best.head ? 'is-best' : undefined}
                      style={{
                        background: `color-mix(in srgb, var(--concept-value) ${shade(score)}%, var(--color-surface))`,
                      }}
                    >
                      {score.toFixed(2)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <p className="visually-hidden" aria-live="polite">
        {announcement}
      </p>
    </div>
  );
}
