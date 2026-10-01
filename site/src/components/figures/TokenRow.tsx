import type { DisplayToken } from '../../lib/guess-tree';
import './figures.css';

interface Props {
  tokens: readonly DisplayToken[];
  showIds?: boolean;
  caret?: boolean;
  newFrom?: number;
}

/** A sentence shown as the token chips the model actually reads. */
export function TokenRow({ tokens, showIds = false, caret = false, newFrom = Infinity }: Props) {
  return (
    <div className="token-row">
      {tokens.map((token, i) => (
        <span
          key={`${i}-${token.id}`}
          className={`token-chip${i >= newFrom ? ' is-new' : ''}${token.special ? ' is-special' : ''}`}
          data-token-index={i}
        >
          <span className="token-text">{token.text}</span>
          {showIds && <span className="token-id">{token.id}</span>}
        </span>
      ))}
      {caret && <span className="token-caret" aria-hidden="true" />}
    </div>
  );
}
