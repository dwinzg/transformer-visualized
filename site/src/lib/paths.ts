/** Joins a base path and a site-relative path with exactly one slash between them. */
export function joinBase(base: string, path: string): string {
  const trimmedBase = base.endsWith('/') ? base : `${base}/`;
  return trimmedBase + path.replace(/^\/+/, '');
}

/** Prefixes a site-relative path with the deployment base, e.g. 'learn/' -> '/transformer-visualized/learn/'. */
export function withBase(path: string): string {
  return joinBase(import.meta.env.BASE_URL, path);
}

export const REPO_URL = 'https://github.com/dwinzg/transformer-visualized';
