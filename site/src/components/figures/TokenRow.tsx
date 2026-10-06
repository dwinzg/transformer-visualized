import type { DisplayToken } from '../../lib/guess-tree';
import { shownToken, spokenToken } from '../../lib/token-text';
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

/**
 * Tokens as a list a screen reader can step through, each with its id. The Tokens chapter and
 * the playground show their text this way.
 */
export function TokenList({ tokens }: { tokens: readonly DisplayToken[] }) {
  return (
    <ol className="token-row tokenizer-tokens" aria-label="Tokens">
      {tokens.map((token, i) => (
        <li key={`${i}-${token.id}`} className={`token-chip${token.special ? ' is-special' : ''}`}>
          <span className="visually-hidden">{`${spokenToken(token)}, id ${token.id}`}</span>
          <span className="token-text" aria-hidden="true">
            {shownToken(token)}
          </span>
          <span className="token-id" aria-hidden="true">
            {token.id}
          </span>
        </li>
      ))}
    </ol>
  );
}
