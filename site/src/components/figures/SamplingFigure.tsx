import { useId, useRef, useState } from 'react';
import type { DisplayToken } from '../../lib/guess-tree';
import {
  chances,
  draw,
  explainChance,
  TOP_K,
  TOP_P,
  type Keep,
  type NextScores,
} from '../../lib/prediction';
import { shownToken, spokenToken } from '../../lib/token-text';
import './figures.css';

const KEEPS: Record<Keep, string> = {
  all: 'All tokens',
  greedy: 'Greedy',
  'top-k': `Top-k, k = ${TOP_K}`,
  'top-p': `Top-p, p = ${TOP_P}`,
};
/** How many bars to draw. The rest of the top 50 still count toward every chance. */
const SHOWN = 8;
/** How many picks to list. Every pick is counted, so the count keeps changing. */
const PICKS = 8;

const pct = (p: number) => (p > 0 && p < 0.001 ? '<0.1%' : `${(p * 100).toFixed(1)}%`);
const OTHER = 'a rarer token';

type Token = NextScores['top'][number]['token'];
const show = shownToken;
const speak = spokenToken;

/** Byte pieces and the end-of-story marker have no plain text to add back to a sentence. */
const addable = (t: Token) => !t.special && !t.text.startsWith('<0x');

const f3 = (v: number) => v.toFixed(3);

/** The softmax behind one token's chance, step by step. */
function ChanceInspector({
  math,
  token,
  keep,
  shown,
  onClose,
}: {
  math: ReturnType<typeof explainChance>;
  token: Token;
  keep: Keep;
  /** The chance the bars show, after the keep rule. */
  shown: number;
  onClose: () => void;
}) {
  const Name = () => (
    <>
      <span aria-hidden="true">{show(token)}</span>
      <span className="visually-hidden">{speak(token)}</span>
    </>
  );
  return (
    <section className="inspector" aria-label="How this chance is worked out">
      <div className="inspector-head">
        <p className="inspector-title">
          How <Name /> gets {pct(math.chance)}
        </p>
        <button type="button" className="figure-button press" onClick={onClose}>
          Close
        </button>
      </div>
      <ol className="inspector-steps">
        <li>
          The model&apos;s score for <Name /> is <strong>{f3(math.logit)}</strong>.
        </li>
        <li>
          Divide by the temperature, {math.temperature.toFixed(1)}. That gives {f3(math.scaled)}.
        </li>
        <li>
          Take away the largest score after dividing, {f3(math.max)}, so the biggest becomes 0. That
          gives {f3(math.scaled - math.max)}.
        </li>
        <li>
          Raise e to that power. That gives <strong>{math.exp.toFixed(4)}</strong>.
        </li>
        <li>
          Do the same for all {math.count.toLocaleString('en')} tokens and add them up. That gives{' '}
          {math.sum.toFixed(4)}.
        </li>
        <li>
          Divide, {math.exp.toFixed(4)} ÷ {math.sum.toFixed(4)}, for a chance of{' '}
          <strong>{pct(math.chance)}</strong>.
          {keep !== 'all' &&
            ` Then the keep rule leaves only some tokens and scales their chances to add up to 1, which gives the ${shown > 0 ? pct(shown) : 'out'} in the bars.`}
        </li>
      </ol>
    </section>
  );
}

/** The real next-token chances after a sentence, reshaped by temperature and a keep rule. */
export default function SamplingFigure({
  data,
  temperature: startTemperature = 1,
  keep: startKeep = 'all',
  onAdd,
  inspect = false,
}: {
  data: NextScores;
  temperature?: number;
  keep?: Keep;
  /** When set, a button adds the latest pick to the text. */
  onAdd?: (token: DisplayToken) => void;
  /** When set, each shown token is a button that opens the math behind its chance. */
  inspect?: boolean;
}) {
  const [temperature, setTemperature] = useState(startTemperature);
  const [keep, setKeep] = useState<Keep>(startKeep);
  const [picks, setPicks] = useState<number[]>([]);
  // Picks point into the chances, so new chances start a new list. The settings stay.
  const [pickedFrom, setPickedFrom] = useState(data);
  const [inspected, setInspected] = useState<number | null>(null);
  if (pickedFrom !== data) {
    setPickedFrom(data);
    setPicks([]);
    setInspected(null);
  }
  const sample = useRef<HTMLButtonElement>(null);
  const id = useId();
  const c = chances(data, temperature, keep);
  const changed = temperature !== startTemperature || keep !== startKeep || picks.length > 0;
  const name = (i: number) => (i < 0 ? OTHER : show(data.top[i].token));
  const say = (i: number) => (i < 0 ? OTHER : speak(data.top[i].token));
  // Everything below the shown bars, from the rest of the top 50 and from the other tokens.
  const below = c.top.slice(SHOWN);
  const otherP = c.other + below.reduce((sum, p) => sum + p, 0);
  const otherN = c.otherCount + below.filter((p) => p > 0).length;
  // A token that is still in always shows a sliver, and one that is out shows none.
  const fill = (p: number) =>
    ({ '--p': `${p > 0 ? Math.max(p * 100, 0.5) : 0}%` }) as React.CSSProperties;

  return (
    <div className="sampling-figure">
      <div className="sampling-controls">
        <label className="sampling-temperature" htmlFor={`${id}-t`}>
          Temperature
          <input
            id={`${id}-t`}
            type="range"
            min={0.1}
            max={2}
            step={0.1}
            value={temperature}
            onChange={(event) => setTemperature(Number(event.target.value))}
          />
          {/* The slider already says its value, so the output is for the eyes only. */}
          <output htmlFor={`${id}-t`} aria-hidden="true">
            {temperature.toFixed(1)}
          </output>
        </label>
        <fieldset className="segmented">
          <legend className="visually-hidden">Which tokens stay in</legend>
          {(Object.keys(KEEPS) as Keep[]).map((k) => (
            <label key={k}>
              <input
                type="radio"
                name={`${id}-keep`}
                checked={keep === k}
                onChange={() => setKeep(k)}
              />
              <span>{KEEPS[k]}</span>
            </label>
          ))}
        </fieldset>
      </div>
      <p className="sampling-summary" aria-live="polite">
        <span aria-hidden="true">{name(0)}</span>
        <span className="visually-hidden">{say(0)}</span> gets {pct(c.top[0])}.
        {keep === 'greedy' && ' Greedy always takes the top token, so temperature changes nothing.'}
      </p>
      <ul className="prob-bars" aria-label="Chances for the next token">
        {data.top.slice(0, SHOWN).map((t, i) => (
          <li key={i} className={`prob-bar${c.top[i] > 0 ? '' : ' is-out'}`} style={fill(c.top[i])}>
            <span className="prob-fill" aria-hidden="true" />
            <span className="prob-word">
              {inspect ? (
                <button
                  type="button"
                  className="attn-inspect press"
                  aria-label={`${speak(t.token)}, show the math`}
                  aria-expanded={inspected === i}
                  onClick={() => setInspected(inspected === i ? null : i)}
                >
                  {show(t.token)}
                </button>
              ) : (
                <>
                  <span aria-hidden="true">{show(t.token)}</span>
                  <span className="visually-hidden">{speak(t.token)}</span>
                </>
              )}
            </span>
            <span className="prob-value">{c.top[i] > 0 ? pct(c.top[i]) : 'out'}</span>
          </li>
        ))}
        <li className={`prob-bar${otherN > 0 ? '' : ' is-out'}`} style={fill(otherP)}>
          <span className="prob-fill" aria-hidden="true" />
          <span className="prob-word sampling-other">
            {otherN > 0 ? `${otherN.toLocaleString('en')} other tokens` : 'Other tokens'}
          </span>
          <span className="prob-value">{otherN > 0 ? pct(otherP) : 'out'}</span>
        </li>
      </ul>
      {inspected !== null && (
        <ChanceInspector
          math={explainChance(data, inspected, temperature)}
          token={data.top[inspected].token}
          keep={keep}
          shown={c.top[inspected]}
          onClose={() => setInspected(null)}
        />
      )}
      <div className="loop-controls">
        <button
          ref={sample}
          type="button"
          className="figure-button press"
          onClick={() => setPicks([draw(c, Math.random()), ...picks])}
        >
          Sample
        </button>
        {onAdd && picks.length > 0 && picks[0] >= 0 && addable(data.top[picks[0]].token) && (
          <button
            type="button"
            className="figure-button press"
            onClick={() => {
              onAdd(data.top[picks[0]].token);
              // This button goes away with the new text, so keep the focus close by.
              sample.current?.focus();
            }}
          >
            Add <span aria-hidden="true">{name(picks[0])}</span>
            <span className="visually-hidden">{say(picks[0])}</span> to the text
          </button>
        )}
        <button
          type="button"
          className="figure-button press"
          aria-disabled={!changed}
          onClick={() => {
            if (!changed) return;
            setTemperature(startTemperature);
            setKeep(startKeep);
            setPicks([]);
          }}
        >
          Start over
        </button>
      </div>
      <p className="sampling-picks">
        <span aria-live="polite">
          {picks.length === 0 ? (
            'Press Sample to pick a token.'
          ) : (
            <>
              {/* The count changes on every press, so a repeat of the same token is announced too. */}
              Pick {picks.length} is <strong aria-hidden="true">{name(picks[0])}</strong>
              <span className="visually-hidden">{say(picks[0])}</span>.
            </>
          )}
        </span>
        {picks.length > 1 && (
          <span className="sampling-history">
            {' '}
            Before that,{' '}
            <span aria-hidden="true">{picks.slice(1, PICKS).map(name).join(', ')}</span>
            <span className="visually-hidden">{picks.slice(1, PICKS).map(say).join(', ')}</span>.
          </span>
        )}
      </p>
    </div>
  );
}
