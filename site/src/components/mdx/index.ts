import AttentionGrid from './AttentionGrid.astro';
import Callout from './Callout.astro';
import Exercise from './Exercise.astro';
import Figure from '../Figure.astro';
import GenerationLoop from './GenerationLoop.astro';
import Level from './Level.astro';
import NearestTokens from './NearestTokens.astro';
import NextWord from './NextWord.astro';
import NumberStrip from './NumberStrip.astro';
import Pipeline from './Pipeline.astro';
import PredictReveal from './PredictReveal.astro';
import Ref from './Ref.astro';
import Sampling from './Sampling.astro';
import Step from './Step.astro';
import Term from './Term.astro';
import TiedScores from './TiedScores.astro';
import Tex from './Tex.astro';
import Tokenizer from './Tokenizer.astro';
import ToyAttention from './ToyAttention.astro';

/** Components every chapter can use without importing them. */
export const mdxComponents = {
  AttentionGrid,
  Callout,
  Exercise,
  Figure,
  GenerationLoop,
  Level,
  NearestTokens,
  NextWord,
  NumberStrip,
  Pipeline,
  PredictReveal,
  Ref,
  Sampling,
  Step,
  Tex,
  Term,
  TiedScores,
  Tokenizer,
  ToyAttention,
};
