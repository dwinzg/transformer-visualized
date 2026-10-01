/**
 * Turns a step title into a URL-safe id, unique among the ids already taken on the page.
 * Falls back to a numbered id when the title has no ASCII letters or digits to slugify.
 */
export function stepId(title: string, taken: ReadonlySet<string>): string {
  const slug = title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');

  if (!slug) {
    let n = 1;
    let id = `step-${n}`;
    while (taken.has(id)) id = `step-${++n}`;
    return id;
  }

  let id = `step-${slug}`;
  for (let n = 2; taken.has(id); n++) id = `step-${slug}-${n}`;
  return id;
}
