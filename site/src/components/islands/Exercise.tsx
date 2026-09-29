import { useId, useState, type FormEvent, type ReactNode } from 'react';
import { isCorrect, parseNumber } from '../../lib/answers';

interface Props {
  prompt: string;
  answer: number;
  tolerance?: number;
  children?: ReactNode;
}

type Status = 'idle' | 'invalid' | 'wrong' | 'right' | 'shown';

/** A small calculation the reader does themselves, checked on the spot. */
export default function Exercise({ prompt, answer, tolerance = 0.01, children }: Props) {
  const [text, setText] = useState('');
  const [status, setStatus] = useState<Status>('idle');
  const id = useId();

  const check = (event: FormEvent) => {
    event.preventDefault();
    const value = parseNumber(text);
    if (value === null) setStatus('invalid');
    else setStatus(isCorrect(value, answer, tolerance) ? 'right' : 'wrong');
  };

  const messages: Record<Status, string> = {
    idle: '',
    invalid: 'Type a number, such as 0.25, 25% or 1/4.',
    wrong: 'Not quite. Try again, or show the answer.',
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
