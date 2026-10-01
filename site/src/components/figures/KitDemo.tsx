import { useState } from 'react';
import { chosenTokens, guessesAt, type GuessTree } from '../../lib/guess-tree';
import { ProbabilityBars } from './ProbabilityBars';
import { TokenRow } from './TokenRow';

export default function KitDemo({ tree }: { tree: GuessTree }) {
  const [path, setPath] = useState<number[]>([]);
  return (
    <div>
      <TokenRow tokens={chosenTokens(tree, path)} showIds caret newFrom={tree.tokens.length} />
      <ProbabilityBars
        guesses={guessesAt(tree, path)}
        label="Next word guesses"
        onPick={(i) => setPath((p) => [...p, i])}
      />
    </div>
  );
}
