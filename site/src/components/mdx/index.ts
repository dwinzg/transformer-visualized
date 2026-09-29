import Callout from './Callout.astro';
import Level from './Level.astro';
import Step from './Step.astro';
import Tex from './Tex.astro';

/** Components every chapter can use without importing them. */
export const mdxComponents = { Callout, Level, Step, Tex };
