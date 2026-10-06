import { loadTinyModel, loadTokenizer } from './model-loader';
import type { Trace } from '@transformer-visualized/engine';
import { attentionMath, runPlayground, type Ask, type WorkerMessage } from './playground-run';

// Runs the model off the main thread, so typing and scrolling never wait for it.
const send = (message: WorkerMessage) => postMessage(message);
const loaded = Promise.all([loadTinyModel(), loadTokenizer()]);
loaded.then(
  () => send({ type: 'ready' }),
  () => send({ type: 'failed' }),
);

// Texts that arrive while a run is busy wait, and only the newest one runs next.
let latest: { seq: number; text: string } | null = null;
let busy = false;
// The newest run's trace, kept to answer questions about its numbers.
let kept: { seq: number; trace: Trace } | null = null;

async function work() {
  if (busy) return;
  busy = true;
  const [model, tokenizer] = await loaded;
  while (latest) {
    const { seq, text } = latest;
    latest = null;
    try {
      const run = runPlayground(model, tokenizer, text, (trace) => (kept = { seq, trace }));
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

type Request = { seq: number; text: string } | { seq: number; ask: Ask };

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
