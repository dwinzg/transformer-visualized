import { useCallback, useLayoutEffect, useRef, type RefObject } from 'react';
import { travel, type Box } from '../../lib/motion';

/** Call `remember(index)` before a pick; after the next render the newest chip travels from that bar. */
export function useTravel(root: RefObject<HTMLElement | null>, trigger: unknown) {
  const from = useRef<Box | null>(null);
  useLayoutEffect(() => {
    if (!from.current || !root.current) return;
    const chips = root.current.querySelectorAll<HTMLElement>('.token-chip');
    const last = chips[chips.length - 1];
    if (last) travel(last, from.current);
    from.current = null;
  }, [trigger, root]);
  return useCallback(
    (index: number) => {
      const bar = root.current?.querySelectorAll<HTMLElement>('.prob-bar .prob-word')[index];
      from.current = bar ? bar.getBoundingClientRect() : null;
    },
    [root],
  );
}
