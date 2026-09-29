import { useId, useState, type ReactNode } from 'react';

interface Props {
  question: string;
  options: string[];
  answer: number;
  children?: ReactNode;
}

/** Ask the reader to commit to a guess before showing the explanation. */
export default function PredictReveal({ question, options, answer, children }: Props) {
  const [choice, setChoice] = useState<number | null>(null);
  const id = useId();
  const revealed = choice !== null;
  return (
    <div className="predict">
      <p className="predict-question" id={`${id}-q`}>
        {question}
      </p>
      <div className="predict-options" role="group" aria-labelledby={`${id}-q`}>
        {options.map((option, i) => (
          <button
            key={option}
            type="button"
            aria-pressed={choice === i}
            disabled={revealed}
            onClick={() => setChoice(i)}
          >
            {option}
          </button>
        ))}
      </div>
      <div aria-live="polite">
        {revealed && (
          <div className="predict-answer">
            <p className="predict-verdict">
              {choice === answer
                ? 'That is the likeliest answer.'
                : `The likeliest answer is "${options[answer]}".`}
            </p>
            {children}
          </div>
        )}
      </div>
    </div>
  );
}
