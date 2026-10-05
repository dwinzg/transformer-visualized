import type { Trace } from '@transformer-visualized/engine';
import type { PartId } from './concepts';

/**
 * Short PyTorch versions of each part, for the map's articles. Run SETUP once, then the GPT-2
 * parts in tour order, and each one prints the same numbers our TypeScript engine computes for
 * "Lily wanted to play with her". The paper-only parts use random numbers, since our model has
 * no encoder.
 */

export const SETUP = `import math, torch
import torch.nn.functional as F
from safetensors.torch import load_file

torch.set_printoptions(precision=4, sci_mode=False)
w = load_file("model.safetensors")  # models/tiny in this repo
ids = torch.tensor([665, 408, 266, 324, 329, 336])  # "Lily wanted to play with her"
T, d, heads = len(ids), 128, 4
mask = torch.ones(T, T).tril().bool()  # a token sees itself and the tokens before it

def linear(x, name):
    return x @ w[name + ".weight"].T + w[name + ".bias"]

def layer_norm(x, name):
    return F.layer_norm(x, (d,), w[name + ".weight"], w[name + ".bias"], eps=1e-5)`;

export interface Snippet {
  /** Sizes in our model, in and out. */
  shape: string;
  code: string;
  /** What the last line prints, from the engine's trace. Missing when the numbers are random. */
  prints?: (trace: Trace) => string;
  /** The paper's version, shown in the original view, when it differs from GPT-2's. */
  paper?: { code: string; prints?: string };
}

const LAST = 5;
const fmt = (values: ArrayLike<number>) =>
  `tensor([${Array.from(values, (v) => v.toFixed(4)).join(', ')}])`;
/** The first four numbers of the last token's row. */
const head4 = (m: { data: Float32Array; cols: number }) =>
  fmt(m.data.subarray(LAST * m.cols, LAST * m.cols + 4));

/** The paper's sine and cosine position numbers, for position 1, first four of 128. */
function sinusoid(pos: number): string {
  const values = [0, 1, 2, 3].map((j) => {
    const i = j - (j % 2);
    const angle = pos / 10000 ** (i / 128);
    return j % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
  });
  return fmt(values);
}

export const PYTORCH: Partial<Record<PartId, Snippet>> = {
  input: {
    shape: '6 token ids',
    code: `print(ids)`,
    prints: (t) => `tensor([${t.tokenIds.join(', ')}])`,
  },
  embedding: {
    shape: '6 ids → 6 × 128',
    code: `tok = w["wte.weight"][ids]  # one row of 128 numbers per token
print(tok[-1, :4])  # the first four numbers of "her"`,
    prints: (t) => head4(t.tokenEmbeddings),
  },
  position: {
    shape: '6 × 128 plus 6 × 128 → 6 × 128',
    code: `pos = w["wpe.weight"][:T]  # one learned row per place
x = tok + pos
print(x[-1, :4])`,
    prints: (t) => head4(t.embeddings),
    paper: {
      code: `p = torch.arange(T)[:, None]
i = torch.arange(0, d, 2)
pe = torch.zeros(T, d)
pe[:, 0::2] = torch.sin(p / 10000 ** (i / d))
pe[:, 1::2] = torch.cos(p / 10000 ** (i / d))
print(pe[1, :4])  # position 1`,
      prints: sinusoid(1),
    },
  },
  'add-norm': {
    shape: '6 × 128 → 6 × 128',
    code: `a = layer_norm(x, "h.0.ln_1")  # norm first, then attention
print(a[-1, :4])`,
    prints: (t) => head4(t.layers[0].ln1.out),
    paper: {
      code: `# The paper adds first and norms after.
x = layer_norm(x + sublayer(x), "...")`,
    },
  },
  'masked-attn': {
    shape: '6 × 128 → 4 heads of 6 × 6 weights → 6 × 128',
    code: `q, k, v = linear(a, "h.0.attn.c_attn").split(d, dim=1)  # each 6 × 128
q, k, v = (m.view(T, heads, -1).transpose(0, 1) for m in (q, k, v))  # 4 × 6 × 32
scores = q @ k.transpose(1, 2) / math.sqrt(d // heads)  # 4 × 6 × 6
weights = scores.masked_fill(~mask, float("-inf")).softmax(-1)
out = (weights @ v).transpose(0, 1).reshape(T, d)  # heads side by side
x = x + linear(out, "h.0.attn.c_proj")  # add it back
print(weights[0, -1])  # where "her" looks in head 1`,
    prints: (t) => {
      const w = t.layers[0].heads[0].weights;
      return fmt(w.data.subarray(LAST * w.cols, (LAST + 1) * w.cols));
    },
  },
  'enc-attn': {
    shape: '6 × 64 → 6 × 6 → 6 × 64, for one of 8 heads',
    code: `q, k, v = torch.randn(3, T, 64)  # one head, with the paper's 64 numbers
weights = (q @ k.T / math.sqrt(64)).softmax(-1)  # 6 × 6, no mask
out = weights @ v`,
  },
  'cross-attn': {
    shape: '6 × 64 and 9 × 64 → 6 × 9 → 6 × 64',
    code: `src = torch.randn(9, 64)  # the encoded source sentence, 9 words
q = torch.randn(T, 64)  # the translation so far
weights = (q @ src.T / math.sqrt(64)).softmax(-1)  # 6 × 9
out = weights @ src`,
  },
  ffn: {
    shape: '6 × 128 → 6 × 512 → 6 × 128',
    code: `hidden = linear(layer_norm(x, "h.0.ln_2"), "h.0.mlp.c_fc")  # 6 × 512
hidden = F.gelu(hidden, approximate="tanh")
x = x + linear(hidden, "h.0.mlp.c_proj")  # back to 6 × 128, added
print(x[-1, :4])  # what leaves block 1`,
    prints: (t) => head4(t.layers[0].output),
  },
  stack: {
    shape: '6 × 128 → 6 × 128, 4 times',
    code: `def block(x, n):
    p = f"h.{n}."
    q, k, v = linear(layer_norm(x, p + "ln_1"), p + "attn.c_attn").split(d, dim=1)
    q, k, v = (m.view(T, heads, -1).transpose(0, 1) for m in (q, k, v))
    weights = (q @ k.transpose(1, 2) / math.sqrt(d // heads)).masked_fill(~mask, float("-inf")).softmax(-1)
    x = x + linear((weights @ v).transpose(0, 1).reshape(T, d), p + "attn.c_proj")
    hidden = F.gelu(linear(layer_norm(x, p + "ln_2"), p + "mlp.c_fc"), approximate="tanh")
    return x + linear(hidden, p + "mlp.c_proj")

x = tok + pos
for n in range(4):
    x = block(x, n)
print(x[-1, :4])`,
    prints: (t) => head4(t.layers[t.layers.length - 1].output),
  },
  'final-norm': {
    shape: '6 × 128 → 6 × 128',
    code: `u = layer_norm(x, "ln_f")
print(u[-1, :4])`,
    prints: (t) => head4(t.lnFinal.out),
  },
  linear: {
    shape: '6 × 128 → 6 × 4,096',
    code: `z = u @ w["wte.weight"].T  # reuses the embedding table
print(z[-1, :4])  # scores for tokens 0 to 3`,
    prints: (t) => head4(t.logits),
  },
  softmax: {
    shape: '4,096 scores → 4,096 chances',
    code: `p = z[-1].softmax(-1)  # adds up to 1
print(p.topk(3).indices)  # the three likeliest next tokens`,
    prints: (t) => {
      const row = Array.from(t.logits.data.subarray(LAST * t.logits.cols, (LAST + 1) * t.logits.cols));
      const top = row
        .map((v, id) => [v, id] as const)
        .sort((a, b) => b[0] - a[0])
        .slice(0, 3)
        .map(([, id]) => id);
      return `tensor([${top.join(', ')}])`;
    },
  },
  output: {
    shape: '4,096 chances → 1 token id',
    code: `greedy = p.argmax()  # always the top token
picked = torch.multinomial(p, 1)  # or pick by chance
print(greedy)`,
    prints: (t) => {
      const row = t.logits.data.subarray(LAST * t.logits.cols, (LAST + 1) * t.logits.cols);
      let best = 0;
      for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
      return `tensor(${best})`;
    },
  },
};
