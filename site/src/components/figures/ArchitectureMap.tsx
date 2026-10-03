import { useEffect, useRef, useState, type KeyboardEvent } from 'react';
import {
  BLOCKS,
  BOX,
  blockLabel,
  blockY,
  firstBlockFor,
  FRAMES,
  RESIDUALS,
  TOUR,
  visibleBlocks,
  type Block,
} from '../../lib/architecture-layout';
import { isPartId, PARTS, type PartId, type View } from '../../lib/concepts';
import { DEPTH_EVENT, DEPTHS, isDepth, pageDepth, type Depth } from '../../lib/depth';
import './figures.css';

interface Props {
  /** KaTeX HTML rendered at build time, per part and view. */
  formulas: Partial<Record<PartId, Partial<Record<View, string>>>>;
  /** Hrefs with the site base. chapter is set only when that chapter exists. */
  links: Partial<Record<PartId, { chapter?: string; glossary?: string }>>;
}

const VIEW_LABELS: Record<View, string> = {
  original: 'Original paper',
  gpt2: 'GPT-2 style (our model)',
};

const byId = new Map(BLOCKS.map((b) => [b.id, b]));
const pct = (value: number, of: number) => `${(value / of) * 100}%`;

/** Arrows from each block to the next one above it in the same column, in tour order. */
function arrows(view: View): [Block, Block][] {
  const pairs: [Block, Block][] = [];
  const columns = new Map<number, Block[]>();
  for (const id of TOUR[view]) {
    const block = byId.get(id)!;
    if (block.part === 'stack') continue;
    columns.set(block.x, [...(columns.get(block.x) ?? []), block]);
  }
  for (const column of columns.values()) {
    for (let i = 1; i < column.length; i++) pairs.push([column[i - 1], column[i]]);
  }
  return pairs;
}

/** A clickable redrawing of Figure 1 of Vaswani et al. 2017, with a guided tour. */
export default function ArchitectureMap({ formulas, links }: Props) {
  const [view, setView] = useState<View>('original');
  const [selected, setSelected] = useState<string | null>(null);
  const [depth, setDepth] = useState<Depth>('story');
  const buttons = useRef(new Map<string, HTMLButtonElement>());
  const scroller = useRef<HTMLDivElement>(null);
  const panel = useRef<HTMLDivElement>(null);
  const focusNext = useRef<string | null>(null);

  useEffect(() => {
    setDepth(pageDepth());
    const onChange = (event: Event) => {
      const detail = (event as CustomEvent<unknown>).detail;
      if (isDepth(detail)) setDepth(detail);
    };
    document.addEventListener(DEPTH_EVENT, onChange);

    // ?part= picks a part, switching to the view that shows it when the default one hides it.
    const params = new URL(location.href).searchParams;
    const part = params.get('part');
    const wanted = params.get('view') === 'gpt2' ? 'gpt2' : 'original';
    if (isPartId(part)) {
      const other: View = wanted === 'gpt2' ? 'original' : 'gpt2';
      const block = firstBlockFor(part, wanted) ?? firstBlockFor(part, other);
      if (block) {
        setView(block.views.includes(wanted) ? wanted : other);
        setSelected(block.id);
      }
    } else if (wanted === 'gpt2') {
      setView('gpt2');
    }
    return () => document.removeEventListener(DEPTH_EVENT, onChange);
  }, []);

  // On narrow screens the drawing scrolls sideways inside its frame. Center the picked block there,
  // without scrolling the page, so the tour buttons and the text stay where they are.
  useEffect(() => {
    const frame = scroller.current;
    const button = selected ? buttons.current.get(selected) : undefined;
    if (!frame || !button || frame.scrollWidth <= frame.clientWidth) return;
    frame.scrollLeft = button.offsetLeft + button.offsetWidth / 2 - frame.clientWidth / 2;
  }, [selected, view]);

  // Move focus after the render that creates the target button.
  useEffect(() => {
    if (!focusNext.current) return;
    buttons.current.get(focusNext.current)?.focus();
    focusNext.current = null;
  });

  const tour = TOUR[view];
  const index = selected ? tour.indexOf(selected) : -1;
  const block = selected ? byId.get(selected) : undefined;
  const part = block ? PARTS[block.part] : undefined;
  const level = DEPTHS.indexOf(depth);

  const pick = (id: string, focus = false) => {
    setSelected(id);
    if (focus) focusNext.current = id;
  };

  // Focus stays on the view toggle, so arrow keys can flip back and forth.
  const changeView = (next: View) => {
    setView(next);
    if (selected && !byId.get(selected)!.views.includes(next)) setSelected(null);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const last = tour.length - 1;
    const current = Math.max(index, 0);
    const next =
      event.key === 'ArrowUp' || event.key === 'ArrowRight'
        ? Math.min(current + 1, last)
        : event.key === 'ArrowDown' || event.key === 'ArrowLeft'
          ? Math.max(current - 1, 0)
          : event.key === 'Home'
            ? 0
            : event.key === 'End'
              ? last
              : -1;
    if (next === -1) return;
    event.preventDefault();
    pick(tour[next], true);
  };

  const rovingId = selected ?? tour[0];
  const visible = visibleBlocks(view);
  // Faded outlines mark the paper's parts that GPT-2 drops, so the shape of Figure 1 stays.
  const hidden = view === 'gpt2' ? BLOCKS.filter((b) => !b.views.includes('gpt2')) : [];
  const formula = part ? formulas[part.id]?.[view] : undefined;
  const link = part ? links[part.id] : undefined;

  return (
    <div className="arch-map">
      <div className="arch-side">
        <div className="arch-controls">
          <fieldset className="arch-views">
            <legend className="visually-hidden">Show</legend>
            {(['original', 'gpt2'] as const).map((v) => (
              <label key={v}>
                <input
                  type="radio"
                  name="arch-view"
                  checked={view === v}
                  onChange={() => changeView(v)}
                />
                <span>{VIEW_LABELS[v]}</span>
              </label>
            ))}
          </fieldset>
          <div className="arch-tour">
            <button
              type="button"
              className="figure-button press"
              aria-disabled={index <= 0}
              onClick={() => index > 0 && pick(tour[index - 1])}
            >
              Previous
            </button>
            <span className="arch-count" aria-live="off">
              {index >= 0 ? `${index + 1} of ${tour.length}` : `${tour.length} parts`}
            </span>
            <button
              type="button"
              className="figure-button press"
              aria-disabled={index === tour.length - 1}
              onClick={() => index < tour.length - 1 && pick(tour[index + 1])}
            >
              Next
            </button>
            <button
              type="button"
              className="figure-button press"
              aria-disabled={selected === null && view === 'original'}
              onClick={() => {
                if (selected === null && view === 'original') return;
                setSelected(null);
                setView('original');
                focusNext.current = TOUR.original[0];
              }}
            >
              Start over
            </button>
          </div>
        </div>

        <div className="arch-panel" ref={panel}>
          {part ? (
            <>
              <h2 className="arch-title">{part.title}</h2>
              <p>{part.story}</p>
              {part.notes?.[view] && <p>{part.notes[view]}</p>}
              {level >= 1 && <p className="arch-numbers">{part.numbers}</p>}
              {level >= 2 && formula && (
                <div className="arch-formula" dangerouslySetInnerHTML={{ __html: formula }} />
              )}
              {level >= 3 && part.code && (
                <pre className="arch-code">
                  <code>{part.code}</code>
                </pre>
              )}
              <p className="arch-links">
                {link?.chapter ? (
                  <a href={link.chapter}>Read the chapter</a>
                ) : (
                  <span>Chapter coming soon</span>
                )}
                {link?.glossary && <a href={link.glossary}>What the word means</a>}
              </p>
            </>
          ) : (
            <p>Tap a part, or press Next for a tour.</p>
          )}
        </div>
        <p className="visually-hidden" aria-live="polite">
          {part ? `${index + 1} of ${tour.length}, ${part.title}` : ''}
        </p>
      </div>

      <div className="arch-scroll" ref={scroller}>
        <div className="arch-canvas">
          <svg viewBox={`0 0 ${BOX.width} ${BOX.height}`} aria-hidden="true" focusable="false">
            <defs>
              <marker
                id="arch-arrow"
                viewBox="0 0 10 10"
                refX="9"
                refY="5"
                markerWidth="6"
                markerHeight="6"
                orient="auto-start-reverse"
              >
                <path d="M0 0 L10 5 L0 10 z" className="arch-arrowhead" />
              </marker>
            </defs>
            {FRAMES.filter((f) => f.views.includes(view)).map((f) => (
              <rect
                key={`${f.x}`}
                className="arch-frame"
                x={f.x}
                y={f.y}
                width={f.w}
                height={f.h}
                rx="12"
              />
            ))}
            {view === 'original' && (
              <>
                <text className="arch-n" x={8} y={496}>
                  N×
                </text>
                {/* The encoder's output feeds the decoder's cross-attention. */}
                <path
                  className="arch-line"
                  d="M150 384 V364 H300 V458 H338"
                  markerEnd="url(#arch-arrow)"
                />
              </>
            )}
            {arrows(view).map(([from, to]) => (
              <line
                key={`${from.id}-${to.id}`}
                className="arch-line"
                x1={from.x + from.w / 2}
                y1={blockY(from, view)}
                x2={to.x + to.w / 2}
                y2={blockY(to, view) + to.h + 2}
                markerEnd="url(#arch-arrow)"
              />
            ))}
            {view === 'gpt2' &&
              RESIDUALS.map(({ from, to }) => (
                <g key={from}>
                  <path
                    className="arch-line"
                    d={`M450 ${from} H332 V${to} H442`}
                    markerEnd="url(#arch-arrow)"
                  />
                  <circle className="arch-plus" cx="450" cy={to} r="6" />
                  <path className="arch-line" d={`M446 ${to} H454 M450 ${to - 4} V${to + 4}`} />
                </g>
              ))}
            {hidden.map((b) => (
              <g key={b.id} className="arch-ghost">
                <rect x={b.x} y={blockY(b, 'original')} width={b.w} height={b.h} rx="8" />
                <text x={b.x + b.w / 2} y={blockY(b, 'original') + b.h / 2}>
                  {blockLabel(b, 'original')}
                </text>
              </g>
            ))}
          </svg>
          <div
            className="arch-blocks"
            role="radiogroup"
            aria-label="Parts of the transformer"
            onKeyDown={onKeyDown}
          >
            {visible.map((b) => (
              <button
                key={b.id}
                ref={(el) => {
                  if (el) buttons.current.set(b.id, el);
                  else buttons.current.delete(b.id);
                }}
                type="button"
                role="radio"
                aria-checked={b.id === selected}
                tabIndex={b.id === rovingId ? 0 : -1}
                className={`arch-block press arch-${PARTS[b.part].color}`}
                style={{
                  left: pct(b.x, BOX.width),
                  top: pct(blockY(b, view), BOX.height),
                  width: pct(b.w, BOX.width),
                  height: pct(b.h, BOX.height),
                }}
                onClick={() => {
                  pick(b.id);
                  // On phones the text sits above the drawing, so show it after a tap.
                  panel.current?.scrollIntoView({ block: 'nearest' });
                }}
              >
                {blockLabel(b, view)}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
