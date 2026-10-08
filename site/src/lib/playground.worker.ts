import { loadLlamaModel, loadTinyModel, loadTokenizer, type Arch } from './model-loader';
import type { AttentionTraceLike } from '@transformer-visualized/engine';
import { attentionMath, runPlayground, type Ask, type WorkerMessage } from './playground-run';

// Runs the model off the main thread, so typing and scrolling never wait for it.
const send = (message: WorkerMessage) => postMessage(message);
// Each model downloads only once the page asks for it, so a reader who never switches never
// downloads the second one.
const MODELS = { gpt2: loadTinyModel, llama: loadLlamaModel } satisfies Record<Arch, unknown>;
loadTokenizer().then(
  () => send({ type: 'ready' }),
  () => send({ type: 'failed' }),
);

// Texts that arrive while a run is busy wait, and only the newest one runs next.
let latest: { seq: number; text: string; arch: Arch } | null = null;
let busy = false;
// The newest run's trace, kept to answer questions about its numbers.
let kept: { seq: number; trace: AttentionTraceLike } | null = null;

async function work() {
  if (busy) return;
  busy = true;
  while (latest) {
    const { seq, text, arch } = latest;
    latest = null;
    let model;
    let tokenizer;
    try {
      [model, tokenizer] = await Promise.all([MODELS[arch](), loadTokenizer()]);
    } catch {
      // The download failed. The page offers Try again, which starts a fresh worker.
      send({ type: 'failed' });
      continue;
    }
    try {
      let trace: AttentionTraceLike | undefined;
      const run = runPlayground(model, tokenizer, text, (t) => (trace = t));
      // Kept only once the whole run worked, so it always matches the run on screen.
      if (trace) kept = { seq, trace };
      send({ type: 'run', seq, run });
    } catch {
      // Say so, or the page would wait for this run forever.
      send({ type: 'run-failed', seq });
    }
    // Let messages that came in during the run update latest before the next one.
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  busy = false;
}

type Request = { seq: number; text: string; arch: Arch } | { seq: number; ask: Ask };

self.onmessage = (event: MessageEvent<Request>) => {
  const message = event.data;
  if ('ask' in message) {
    // Only the run the page is showing can be explained. An older one gets no numbers.
    let math = null;
    if (kept?.seq === message.seq) {
      try {
        math = attentionMath(kept.trace, message.ask);
      } catch {
        math = null;
      }
    }
    send({ type: 'explained', seq: message.seq, ask: message.ask, math });
    return;
  }
  latest = message;
  work().catch(() => (busy = false));
};
