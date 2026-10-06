import { useEffect, useId, useState } from 'react';
import {
  normalizeText,
  unsupportedCharacters,
  type Tokenizer,
} from '@transformer-visualized/engine';
import { loadTokenizer, toTokens } from '../../lib/model-loader';
import type { DisplayToken } from '../../lib/guess-tree';
import { TokenList } from './TokenRow';
import './figures.css';

const MAX_LENGTH = 200;
/** How long typing has to pause before a screen reader hears the new count. */
const ANNOUNCE_DELAY_MS = 800;

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
      {tokens && tokens.length > 0 && <TokenList tokens={tokens} />}
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
