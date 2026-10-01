import { useId, useState, type ReactNode, type SyntheticEvent } from 'react';
import { isCorrect, parseNumber } from '../../lib/answers';

interface Props {
  prompt: string;
  answer: number;
  tolerance?: number;
  children?: ReactNode;
}

type Status = 'idle' | 'invalid' | 'wrong' | 'right' | 'shown';

/** "2nd", "3rd", "11th" and so on. */
function ordinal(n: number): string {
  if (n % 100 >= 11 && n % 100 <= 13) return `${n}th`;
  return `${n}${['th', 'st', 'nd', 'rd'][n % 10] ?? 'th'}`;
}

/** A small calculation the reader does themselves, checked on the spot. */
export default function Exercise({ prompt, answer, tolerance = 0.01, children }: Props) {
  const [text, setText] = useState('');
  const [status, setStatus] = useState<Status>('idle');
  const [attempt, setAttempt] = useState(0);
  const id = useId();

  const check = (event: SyntheticEvent<HTMLFormElement>) => {
    event.preventDefault();
    const value = parseNumber(text);
    if (value === null) {
      setStatus('invalid');
      return;
    }
    if (isCorrect(value, answer, tolerance)) {
      setStatus('right');
      return;
    }
    setAttempt((n) => n + 1);
    setStatus('wrong');
  };

  const messages: Record<Status, string> = {
    idle: '',
    invalid: 'Type a number, such as 0.25, 25% or 1/4.',
    // The attempt count changes the wording on every retry, so the aria-live region has
    // something new to announce even when the reader gets it wrong again.
    wrong:
      attempt > 1
        ? `Not quite, try again (${ordinal(attempt)} try).`
        : 'Not quite. Try again, or show the answer.',
    right: 'Correct.',
    shown: `The answer is ${answer}.`,
  };

  return (
    <form className="exercise" onSubmit={check}>
      <label htmlFor={`${id}-input`} className="exercise-prompt">
        {prompt}
      </label>
      <div className="exercise-row">
        <input
          id={`${id}-input`}
          inputMode="decimal"
          autoComplete="off"
          value={text}
          onChange={(event) => setText(event.target.value)}
        />
        <button type="submit">Check</button>
        <button type="button" onClick={() => setStatus('shown')}>
          Show answer
        </button>
      </div>
      <p className="exercise-feedback" aria-live="polite" data-status={status}>
        {messages[status]}
      </p>
      {(status === 'right' || status === 'shown') && children}
    </form>
  );
}
