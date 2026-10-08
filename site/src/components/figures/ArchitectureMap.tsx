import { useEffect, useRef, useState, type KeyboardEvent } from 'react';
import { createPortal } from 'react-dom';
import {
  BLOCKS,
  BOX,
  blockLabel,
  blockName,
  blockY,
  firstBlockFor,
  FRAMES,
  RESIDUALS,
  TOUR,
  visibleBlocks,
  type Block,
} from '../../lib/architecture-layout';
import { isPartId, PARTS, VIEWS, type PartId, type View } from '../../lib/concepts';
import './figures.css';

interface Props {
  /** KaTeX HTML rendered at build time, per part and view. */
  formulas: Partial<Record<PartId, Partial<Record<View, string>>>>;
  /** Hrefs with the site base. chapter is set only when that chapter exists. */
  links: Partial<Record<PartId, { chapter?: string; glossary?: string }>>;
  /** PyTorch for each part, with what it prints for our sentence. */
  articles: Partial<Record<PartId, Article>>;
  /** GPT-2's second norm box, which runs ln_2 instead of ln_1. */
  secondNorm: Article;
  /** The PyTorch setup every snippet needs. */
  setup: string;
  /** The same for the Llama-style model, run on models/llama-tiny. */
  llama: { articles: Partial<Record<PartId, Article>>; secondNorm: Article; setup: string };
}

export interface Article {
  shape: string;
  code: string;
  prints?: string;
  paper?: { code: string; prints?: string; sketch?: boolean };
}

/** The page's spot for the long article, below the figure. */
const ARTICLE_ID = 'part-article';

/** A code block with a Copy button. */
function Code({ code, label, copy = true }: { code: string; label: string; copy?: boolean }) {
  const [copied, setCopied] = useState(false);
  // Some browsers block the clipboard, and then there is nothing to press.
  const canCopy = copy && typeof navigator !== 'undefined' && !!navigator.clipboard;
  return (
    <div className="arch-snippet">
      {canCopy && (
        <button
          type="button"
          className="figure-button press"
          onClick={() =>
            navigator.clipboard.writeText(code).then(
              () => setCopied(true),
              () => setCopied(false),
            )
          }
        >
          {copied ? 'Copied' : 'Copy'}
          <span className="visually-hidden"> {label}</span>
        </button>
      )}
      <pre tabIndex={0} aria-label={label}>
        <code>{code}</code>
      </pre>
    </div>
  );
}

const VIEW_LABELS: Record<View, string> = {
  original: 'Original paper',
  gpt2: 'GPT-2 style (our model)',
  llama: 'Llama style',
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
export default function ArchitectureMap({
  formulas,
  links,
  articles: gpt2Articles,
  secondNorm: gpt2SecondNorm,
  setup: gpt2Setup,
  llama,
}: Props) {
  const [view, setView] = useState<View>('original');
  const [selected, setSelected] = useState<string | null>(null);
  const [target, setTarget] = useState<HTMLElement | null>(null);
  const buttons = useRef(new Map<string, HTMLButtonElement>());
  const scroller = useRef<HTMLDivElement>(null);
  const panel = useRef<HTMLDivElement>(null);
  const title = useRef<HTMLHeadingElement>(null);
  const focusNext = useRef<string | null>(null);
  const focusTitle = useRef(false);

  useEffect(() => {
    // The page's placeholder is for readers without JavaScript. This island fills the spot now.
    const spot = document.getElementById(ARTICLE_ID);
    spot?.replaceChildren();
    setTarget(spot);

    // ?part= picks a part, switching to the view that shows it when the default one hides it.
    const params = new URL(location.href).searchParams;
    const part = params.get('part');
    const asked = params.get('view');
    const wanted: View = VIEWS.find((v) => v === asked) ?? 'original';
    if (isPartId(part)) {
      // The asked view first, then GPT-2, then the paper, the first that has the part.
      const order = [...new Set<View>([wanted, 'gpt2', 'original'])];
      const shown = order.find((v) => firstBlockFor(part, v));
      if (shown) {
        setView(shown);
        setSelected(firstBlockFor(part, shown)!.id);
      }
    } else if (wanted !== 'original') {
      setView(wanted);
    }
  }, []);

  // On narrow screens the drawing scrolls sideways inside its frame. Center the picked block there,
  // without scrolling the page, so the tour buttons and the text stay where they are.
  useEffect(() => {
    const frame = scroller.current;
    const button = selected ? buttons.current.get(selected) : undefined;
    if (!frame || !button || frame.scrollWidth <= frame.clientWidth) return;
    frame.scrollLeft = button.offsetLeft + button.offsetWidth / 2 - frame.clientWidth / 2;
  }, [selected, view]);

  // Move focus after the render that creates the target button or article.
  useEffect(() => {
    if (focusTitle.current) {
      focusTitle.current = false;
      title.current?.scrollIntoView({ block: 'start' });
      title.current?.focus({ preventScroll: true });
    }
    if (!focusNext.current) return;
    buttons.current.get(focusNext.current)?.focus();
    focusNext.current = null;
  });

  const tour = TOUR[view];
  const index = selected ? tour.indexOf(selected) : -1;
  const block = selected ? byId.get(selected) : undefined;
  const part = block ? PARTS[block.part] : undefined;

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
  // Faded outlines mark the paper's parts this view drops, so the shape of Figure 1 stays.
  const hidden = view === 'original' ? [] : BLOCKS.filter((b) => !b.views.includes(view));
  const { articles, secondNorm, setup } =
    view === 'llama'
      ? llama
      : { articles: gpt2Articles, secondNorm: gpt2SecondNorm, setup: gpt2Setup };
  const formula = part ? formulas[part.id]?.[view] : undefined;
  const note = (block?.notes ?? part?.notes)?.[view];
  const link = part ? links[part.id] : undefined;
  const article =
    view !== 'original' && block?.id === 'add-norm-3' ? secondNorm : part && articles[part.id];
  // Box names, since GPT-2's two norms share one part.
  const titleAt = (i: number) => blockName(byId.get(tour[i])!, view);
  // The article's own buttons move the reader to the top of the next article.
  const read = (i: number) => {
    pick(tour[i]);
    focusTitle.current = true;
  };

  const articleView = (
    <>
      <details className="arch-setup">
        <summary>Run the code yourself</summary>
        <p>
          Every part below has a few lines of PyTorch. Run this setup once, next to{' '}
          <code>model.safetensors</code> from{' '}
          <code>{view === 'llama' ? 'models/llama-tiny' : 'models/tiny'}</code> in the repository.
          Then run the parts in the order of the {view === 'llama' ? 'Llama' : 'GPT-2'} tour, and
          each one prints the same numbers as that model.
        </p>
        <Code code={setup} label="PyTorch setup" />
      </details>
      {part ? (
        // A new key for each box, so Copy buttons start fresh.
        <article
          key={`${view}-${selected}`}
          className="arch-article"
          aria-labelledby={`${ARTICLE_ID}-title`}
        >
          <h2 id={`${ARTICLE_ID}-title`} ref={title} tabIndex={-1}>
            {part.title}
          </h2>
          <p>{part.story}</p>
          {note && <p>{note}</p>}
          <h3>Sizes</h3>
          <p>{(view === 'llama' && part.llamaNumbers) || part.numbers}</p>
          {article && (
            <p>
              <code>{article.shape}</code>
            </p>
          )}
          {formula && (
            <>
              <h3>Formula</h3>
              {/* Focusable, so a wide formula can be scrolled with keys on a phone. */}
              <div
                className="arch-formula"
                tabIndex={0}
                role="group"
                aria-label={`Formula for ${part.title}`}
                dangerouslySetInnerHTML={{ __html: formula }}
              />
            </>
          )}
          {article && (
            <>
              <h3>In PyTorch</h3>
              {!article.prints && <p>Our model has no encoder, so this uses random numbers.</p>}
              <Code code={article.code} label={`PyTorch for ${part.title}`} />
              {article.prints && (
                <>
                  <p className="arch-prints-label">It prints</p>
                  <pre className="arch-prints" tabIndex={0} aria-label="What it prints">
                    <code>{article.prints}</code>
                  </pre>
                </>
              )}
              {view === 'original' && article.paper && (
                <>
                  <h3>The paper&apos;s version</h3>
                  <Code
                    code={article.paper.code}
                    label={`The paper's ${part.title}`}
                    copy={!article.paper.sketch}
                  />
                  {article.paper.prints && (
                    <pre className="arch-prints" tabIndex={0} aria-label="What it prints">
                      <code>{article.paper.prints}</code>
                    </pre>
                  )}
                </>
              )}
            </>
          )}
          {part.code && (
            <>
              <h3>In our engine</h3>
              <pre className="arch-code" tabIndex={0} aria-label="Engine code">
                <code>{part.code}</code>
              </pre>
            </>
          )}
          {(link?.chapter || link?.glossary) && (
            <p className="arch-links">
              {link.chapter && <a href={link.chapter}>Read the chapter</a>}
              {link.glossary && <a href={link.glossary}>What the word means</a>}
            </p>
          )}
          <nav className="arch-pager" aria-label="Previous and next part">
            {index > 0 && (
              <button type="button" className="press" onClick={() => read(index - 1)}>
                <span className="arch-pager-label">Previous</span> {titleAt(index - 1)}
              </button>
            )}
            {index < tour.length - 1 && (
              <button type="button" className="press" onClick={() => read(index + 1)}>
                <span className="arch-pager-label">Next</span> {titleAt(index + 1)}
              </button>
            )}
          </nav>
        </article>
      ) : (
        <p className="arch-article">
          Pick a part on the map to read about it here, or{' '}
          <button type="button" className="figure-button press" onClick={() => read(0)}>
            start with {titleAt(0)}
          </button>
        </p>
      )}
    </>
  );

  return (
    <div className="arch-map">
      {target && createPortal(articleView, target)}
      <div className="arch-side">
        <div className="arch-controls">
          <fieldset className="arch-views">
            <legend className="visually-hidden">Show</legend>
            {VIEWS.map((v) => (
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
              <p className="arch-title">{part.title}</p>
              <p>{part.story}</p>
              {note && <p>{note}</p>}
              <p>
                <a href={`#${ARTICLE_ID}`}>Sizes, formula and code below</a>
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
            {view !== 'original' &&
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
                aria-label={
                  blockName(b, view) === blockLabel(b, view) ? undefined : blockName(b, view)
                }
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
