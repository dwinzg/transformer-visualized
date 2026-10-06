import { useEffect, useRef } from 'react';
import type { DisplayToken } from '../../lib/guess-tree';
import type { AttentionMath } from '../../lib/playground-run';
import { shownToken, spokenToken } from '../../lib/token-text';
import './figures.css';

const Name = ({ token }: { token: DisplayToken }) => (
  <>
    <span aria-hidden="true">{shownToken(token.text)}</span>
    <span className="visually-hidden">{spokenToken(token.text)}</span>
  </>
);

const fmt = (v: number) => v.toFixed(4);

/** A list of numbers, wrapped to fit. */
const Vector = ({ values, label }: { values: number[]; label: string }) => (
  <p className="inspect-vector">
    <span className="visually-hidden">{`${label}, ${values.length} numbers. `}</span>
    {values.map(fmt).join('  ')}
  </p>
);

/** The math behind one attention weight, from the query and key to the softmax. */
export default function AttentionInspector({
  math,
  from,
  to,
  onClose,
}: {
  math: AttentionMath;
  /** The token that looks. */
  from: DisplayToken;
  /** The token it looks at. */
  to: DisplayToken;
  onClose: () => void;
}) {
  // It opens below the grid, so bring it into view and put the focus on it.
  const root = useRef<HTMLElement>(null);
  useEffect(() => {
    root.current?.scrollIntoView({ block: 'nearest' });
    root.current?.focus({ preventScroll: true });
  }, [math]);
  return (
    <section
      ref={root}
      tabIndex={-1}
      className="inspector"
      aria-label="How this weight is worked out"
    >
      <div className="inspector-head">
        <p className="inspector-title">
          How <Name token={from} /> gives <Name token={to} /> a weight of {math.weight.toFixed(2)}
        </p>
        <button type="button" className="figure-button press" onClick={onClose}>
          Close
        </button>
      </div>
      <ol className="inspector-steps">
        <li>
          The query of <Name token={from} />, what it is looking for.
          <Vector values={math.query} label="Query" />
        </li>
        <li>
          The key of <Name token={to} />, what it offers.
          <Vector values={math.key} label="Key" />
        </li>
        <li>
          Multiply them number by number and add up the {math.products.length} products. That gives{' '}
          <strong>{fmt(math.dot)}</strong>.
        </li>
        <li>
          Divide by √{math.query.length}, so the scores stay in a steady range. That gives a score
          of <strong>{fmt(math.score)}</strong>.
        </li>
        <li>
          Softmax turns the row of scores into weights. Each score becomes e to the power of the
          score minus the row&apos;s largest score. For <Name token={to} /> that is {fmt(math.exp)},
          and the whole row adds up to {fmt(math.expSum)}. Dividing gives the weight,{' '}
          <strong>{fmt(math.weight)}</strong>.
        </li>
        <li>
          The weight scales the value of <Name token={to} />, what it passes on. The head adds up
          the scaled values of every token it sees.
          <Vector values={math.value} label="Value" />
        </li>
      </ol>
    </section>
  );
}
