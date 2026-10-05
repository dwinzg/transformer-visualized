import { loadTinyModel, loadTokenizer } from './model-loader';
import { runPlayground, type WorkerMessage } from './playground-run';

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

async function work() {
  if (busy) return;
  busy = true;
  const [model, tokenizer] = await loaded;
  while (latest) {
    const { seq, text } = latest;
    latest = null;
    send({ type: 'run', seq, run: runPlayground(model, tokenizer, text) });
    // Let messages that came in during the run update latest before the next one.
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  busy = false;
}

self.onmessage = (event: MessageEvent<{ seq: number; text: string }>) => {
  latest = event.data;
  work().catch(() => (busy = false));
};
