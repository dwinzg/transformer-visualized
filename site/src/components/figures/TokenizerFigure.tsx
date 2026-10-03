import { useEffect, useId, useState } from 'react';
import { normalizeText, Tokenizer, unsupportedCharacters } from '@transformer-visualized/engine';
import tokenizerUrl from '../../../../models/tiny/tokenizer.json?url';
import type { DisplayToken } from '../../lib/guess-tree';
import './figures.css';

const MAX_LENGTH = 200;

// One download for every tokenizer figure on the page. It starts when the first one mounts, which
// client:visible delays until the figure is near the screen.
let loading: Promise<Tokenizer> | undefined;
function loadTokenizer(): Promise<Tokenizer> {
  loading ??= fetch(tokenizerUrl)
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then((json: unknown) => Tokenizer.fromJSON(json))
    .catch((error: unknown) => {
      loading = undefined;
      throw error;
    });
  return loading;
}

function toTokens(tokenizer: Tokenizer, text: string): DisplayToken[] {
  return tokenizer.encode(normalizeText(text)).map((id) => ({ id, text: tokenizer.tokenText(id) }));
}

/** A space at the start of a token is part of it, so it is drawn as a dot. */
const shown = (text: string) => text.replace(/ /g, '·');
const spokenToken = (text: string) => text.replace(/ /g, 'space ');

/** Type any text and see the tokens and ids the tiny model would read. */
export default function TokenizerFigure({
  text: initialText,
  tokens: initialTokens,
}: {
  text: string;
  /** The starting text's tokens, worked out at build time so they show before the tokenizer loads. */
  tokens: DisplayToken[];
}) {
  const [text, setText] = useState(initialText);
  const [tokenizer, setTokenizer] = useState<Tokenizer | null>(null);
  const [failed, setFailed] = useState(false);
  const inputId = useId();
  const noteId = useId();

  useEffect(() => {
    let live = true;
    loadTokenizer().then(
      (loaded) => live && setTokenizer(loaded),
      () => live && setFailed(true),
    );
    return () => {
      live = false;
    };
  }, []);

  const tokens =
    text === initialText ? initialTokens : tokenizer ? toTokens(tokenizer, text) : null;
  const unseen = unsupportedCharacters(text);
  const cleaned = normalizeText(text) !== text;

  return (
    <div className="tokenizer-figure" data-ready={tokenizer ? '' : undefined}>
      <div className="tokenizer-head">
        <label htmlFor={inputId}>Your text</label>
        <button
          type="button"
          className="figure-button press"
          aria-disabled={text === initialText}
          onClick={() => setText(initialText)}
        >
          Start over
        </button>
      </div>
      <textarea
        id={inputId}
        className="tokenizer-input"
        rows={2}
        maxLength={MAX_LENGTH}
        spellCheck={false}
        value={text}
        aria-describedby={noteId}
        onChange={(event) => setText(event.target.value)}
      />
      <p className="tokenizer-count" aria-live="polite">
        {tokens
          ? `${text.length} characters, ${tokens.length} ${tokens.length === 1 ? 'token' : 'tokens'}`
          : failed
            ? 'The tokenizer did not load. Check your connection and reload the page.'
            : 'Loading the tokenizer...'}
      </p>
      {tokens && tokens.length > 0 && (
        <ol className="token-row tokenizer-tokens" aria-label="Tokens">
          {tokens.map((token, i) => (
            <li
              key={`${i}-${token.id}`}
              className="token-chip"
              aria-label={`${spokenToken(token.text)}, id ${token.id}`}
            >
              <span className="token-text">{shown(token.text)}</span>
              <span className="token-id">{token.id}</span>
            </li>
          ))}
        </ol>
      )}
      <div id={noteId} className="tokenizer-notes">
        <p>A dot marks a space. The small number is the token id.</p>
        {cleaned && (
          <p>Curly quotes and long dashes become plain ones first, as in the training text.</p>
        )}
        {unseen.length > 0 && (
          <p className="tokenizer-unseen">
            The model never saw {unseen.join(' ')} in training, so it reads{' '}
            {unseen.length === 1 ? 'it' : 'them'} as raw bytes.
          </p>
        )}
      </div>
    </div>
  );
}
