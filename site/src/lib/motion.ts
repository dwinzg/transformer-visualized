/**
 * The site's motion patterns. Only five kinds of motion are allowed (enter, grow, travel, press
 * and page), and each must still make sense with reduced motion, where it becomes instant.
 * Grow and press are pure CSS (see global.css); enter and travel need a little script.
 */

export interface Box {
  left: number;
  top: number;
  width: number;
  height: number;
}

export function prefersReducedMotion(): boolean {
  return typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/** The transform that puts an element at `to` back where it was at `from` (the "invert" in FLIP). */
export function flipDelta(
  from: Box,
  to: Box,
): { x: number; y: number; scaleX: number; scaleY: number } {
  return {
    x: from.left - to.left,
    y: from.top - to.top,
    scaleX: to.width === 0 ? 1 : from.width / to.width,
    scaleY: to.height === 0 ? 1 : from.height / to.height,
  };
}

/**
 * Travel: animate `el`, already in its new place, from the box it used to occupy. Returns null
 * when motion is reduced, so the change is instant.
 */
export function travel(
  el: HTMLElement,
  from: Box,
  options: { duration?: number } = {},
): Animation | null {
  if (prefersReducedMotion()) return null;
  const d = flipDelta(from, el.getBoundingClientRect());
  return el.animate(
    [
      {
        transformOrigin: 'top left',
        transform: `translate(${d.x}px, ${d.y}px) scale(${d.scaleX}, ${d.scaleY})`,
      },
      { transformOrigin: 'top left', transform: 'none' },
    ],
    { duration: options.duration ?? 350, easing: 'cubic-bezier(0.2, 0.8, 0.2, 1)' },
  );
}

/**
 * Enter: marks every `[data-enter]` element under `root` with `.is-visible` the first time it
 * scrolls into view. CSS does the fade and rise. Returns a cleanup function.
 */
export function enterOnView(root: ParentNode): () => void {
  const targets = Array.from(root.querySelectorAll<HTMLElement>('[data-enter]'));
  if (prefersReducedMotion() || typeof IntersectionObserver !== 'function') {
    for (const el of targets) el.classList.add('is-visible');
    return () => {};
  }
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      }
    },
    { rootMargin: '0px 0px -10% 0px' },
  );
  for (const el of targets) observer.observe(el);
  return () => observer.disconnect();
}
