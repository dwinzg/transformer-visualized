import Callout from './Callout.astro';
import Exercise from './Exercise.astro';
import Figure from '../Figure.astro';
import GenerationLoop from './GenerationLoop.astro';
import Level from './Level.astro';
import NextWord from './NextWord.astro';
import Pipeline from './Pipeline.astro';
import PredictReveal from './PredictReveal.astro';
import Ref from './Ref.astro';
import Step from './Step.astro';
import Term from './Term.astro';
import Tex from './Tex.astro';

/** Components every chapter can use without importing them. */
export const mdxComponents = {
  Callout,
  Exercise,
  Figure,
  GenerationLoop,
  Level,
  NextWord,
  Pipeline,
  PredictReveal,
  Ref,
  Step,
  Tex,
  Term,
};
