import { useEffect, useId, useState } from 'react';
import { normalizeText, Tokenizer, unsupportedCharacters } from '@transformer-visualized/engine';
import tokenizerUrl from '../../../../models/tiny/tokenizer.json?url';
import type { DisplayToken } from '../../lib/guess-tree';
import './figures.css';

const MAX_LENGTH = 200;
const END_OF_STORY = '<|endoftext|>';
/** How long typing has to pause before a screen reader hears the new count. */
const ANNOUNCE_DELAY_MS = 800;

// One download for every tokenizer figure on the page. It starts when the first one mounts, which
// client:visible delays until the figure is near the screen. A failed download can be tried again.
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
  const end = tokenizer.specialTokenId(END_OF_STORY);
  return tokenizer
    .encode(normalizeText(text))
    .map((id) =>
      id === end
        ? { id, text: '[end of story]', special: true }
        : { id, text: tokenizer.tokenText(id) },
    );
}

/** Characters with no look of their own, drawn and spoken as a mark instead. */
const MARKS: Record<string, [shown: string, spoken: string]> = {
  ' ': ['\u00b7', 'space '],
  '\n': ['\u21b5', 'new line '],
  '\t': ['\u21e5', 'tab '],
};
const shown = (text: string) => text.replace(/[ \n\t]/g, (char) => MARKS[char][0]);
const spoken = (text: string) =>
  /^<0x[0-9A-F]{2}>$/.test(text)
    ? `byte ${text.slice(3, 5)}`
    : text.replace(/[ \n\t]/g, (char) => MARKS[char][1]).trim();

/** Counts what a reader sees as one character, so an emoji counts once. */
const segmenter = new Intl.Segmenter();
const characterCount = (text: string) => [...segmenter.segment(text)].length;

/** Emoji parts that have no look of their own, left out when naming what the model never saw. */
const INVISIBLE = /[\ufe0f\u200d]/;

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
  const [mounted, setMounted] = useState(false);
  const [tokenizer, setTokenizer] = useState<Tokenizer | null>(null);
  const [failed, setFailed] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const [announcement, setAnnouncement] = useState('');
  const inputId = useId();
  const noteId = useId();

  useEffect(() => setMounted(true), []);

  useEffect(() => {
    let live = true;
    loadTokenizer().then(
      (loaded) => live && setTokenizer(loaded),
      () => live && setFailed(true),
    );
    return () => {
      live = false;
    };
  }, [attempt]);

  const tokens =
    text === initialText ? initialTokens : tokenizer ? toTokens(tokenizer, text) : null;
  const unseen = unsupportedCharacters(text).filter((char) => !INVISIBLE.test(char));
  const cleaned = normalizeText(text) !== text;
  const count = tokens
    ? `${characterCount(text)} characters, ${tokens.length} ${tokens.length === 1 ? 'token' : 'tokens'}`
    : failed
      ? 'The tokenizer did not load. Check your connection, then type again to retry.'
      : 'Loading the tokenizer...';

  // Announce the count once typing pauses, not on every key.
  useEffect(() => {
    if (!mounted) return;
    const timer = setTimeout(() => setAnnouncement(count), ANNOUNCE_DELAY_MS);
    return () => clearTimeout(timer);
  }, [count, mounted]);

  const edit = (next: string) => {
    setText(next);
    if (failed) {
      setFailed(false);
      setAttempt((n) => n + 1);
    }
  };

  return (
    <div className="tokenizer-figure" data-ready={tokenizer ? '' : undefined}>
      <div className="tokenizer-head">
        <label htmlFor={inputId}>Your text</label>
        <button
          type="button"
          className="figure-button press"
          aria-disabled={text === initialText}
          onClick={() => edit(initialText)}
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
        // Typing before the page is ready would be lost, so wait until it is.
        readOnly={!mounted}
        value={text}
        aria-describedby={noteId}
        onChange={(event) => edit(event.target.value)}
      />
      <p className="tokenizer-count">{count}</p>
      <p className="visually-hidden" aria-live="polite">
        {announcement}
      </p>
      {tokens && tokens.length > 0 && (
        <ol className="token-row tokenizer-tokens" aria-label="Tokens">
          {tokens.map((token, i) => (
            <li
              key={`${i}-${token.id}`}
              className={`token-chip${token.special ? ' is-special' : ''}`}
            >
              <span className="visually-hidden">{`${spoken(token.text)}, id ${token.id}`}</span>
              <span className="token-text" aria-hidden="true">
                {shown(token.text)}
              </span>
              <span className="token-id" aria-hidden="true">
                {token.id}
              </span>
            </li>
          ))}
        </ol>
      )}
      <div id={noteId} className="tokenizer-notes">
        <p>A dot marks a space. The small number is the token id.</p>
        {cleaned && (
          <p>
            Curly quotes, long dashes and other fancy punctuation become plain ones first, as in the
            training text.
          </p>
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
