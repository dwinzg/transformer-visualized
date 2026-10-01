import { defineCollection, reference } from 'astro:content';
import { file, glob } from 'astro/loaders';
import { z } from 'astro/zod';

const author = z.union([
  z.object({ family: z.string().min(1), given: z.string().min(1) }),
  z.object({ literal: z.string().min(1) }),
]);

const references = defineCollection({
  loader: file('src/content/references.json'),
  schema: z.object({
    type: z.enum(['paper', 'book', 'report', 'article', 'video', 'website', 'software', 'dataset']),
    authors: z.array(author).min(1),
    title: z.string().min(1),
    year: z.number().int().min(1900).max(2100),
    venue: z.string().optional(),
    url: z.url(),
    arxiv: z.string().optional(),
    doi: z.string().optional(),
    note: z.string().optional(),
    topics: z.array(z.string()).min(1),
  }),
});

const glossary = defineCollection({
  loader: file('src/content/glossary.json'),
  schema: z.object({
    term: z.string().min(1),
    short: z.string().min(1),
    related: z.array(reference('glossary')).default([]),
  }),
});

const chapters = defineCollection({
  loader: glob({ pattern: '*.mdx', base: './src/content/chapters' }),
  schema: z.object({
    title: z.string().min(1),
    order: z.number().int().min(0),
    summary: z.string().min(1),
    goals: z.array(z.string().min(1)).min(2).max(4),
    minutes: z.number().int().positive(),
    recap: z.array(z.string().min(1)).min(2).max(5),
    references: z.array(reference('references')).min(1),
    draft: z.boolean().default(false),
  }),
});

export const collections = { references, glossary, chapters };
