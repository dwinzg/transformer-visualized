import type { LlamaTrace, Trace } from '@transformer-visualized/engine';
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
# "Lily wanted to play with her"
ids = torch.tensor([665, 408, 266, 324, 329, 336])
T, d, heads = len(ids), 128, 4
# A token sees itself and the tokens before it.
mask = torch.ones(T, T).tril().bool()

def linear(x, name):
    return x @ w[name + ".weight"].T + w[name + ".bias"]

def layer_norm(x, name):
    weight, bias = w[name + ".weight"], w[name + ".bias"]
    return F.layer_norm(x, (d,), weight, bias, eps=1e-5)`;

export interface Snippet {
  /** Sizes in our model, in and out. */
  shape: string;
  code: string;
  /** What the last line prints, from the engine's trace. Missing when the numbers are random. */
  prints?: (trace: Trace | LlamaTrace) => string;
  /** The paper's version, shown in the original view. A sketch has no Copy button. */
  paper?: { code: string; prints?: string; sketch?: boolean };
}

const LAST = 5;
/** Prints like PyTorch, which pads every number to the same width. */
function fmt(values: ArrayLike<number>): string {
  const text = Array.from(values, (v) => v.toFixed(4));
  const width = Math.max(...text.map((t) => t.length));
  return `tensor([${text.map((t) => t.padStart(width)).join(', ')}])`;
}
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

/**
 * GPT-2's second norm box, before the feed-forward layer. It is the same part as the first, but
 * by then attention has changed x, so it runs ln_2 and prints different numbers.
 */
export const SECOND_NORM: Snippet = {
  shape: '6 × 128 → 6 × 128',
  code: `b = layer_norm(x, "h.0.ln_2")  # norm again, before feed forward
print(b[-1, :4])`,
  prints: (t) => head4(t.layers[0].ln2.out),
};

export const PYTORCH: Partial<Record<PartId, Snippet>> = {
  input: {
    shape: '6 token ids',
    code: `print(ids)`,
    prints: (t) => `tensor([${t.tokenIds.join(', ')}])`,
  },
  embedding: {
    shape: '6 ids → 6 × 128',
    code: `# One row of 128 numbers per token.
tok = w["wte.weight"][ids]
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
      code: `# The paper adds first and norms after. A sketch, not runnable.
x = layer_norm(x + sublayer(x))`,
      sketch: true,
    },
  },
  'masked-attn': {
    shape: '6 × 128 → 4 heads of 6 × 6 weights → 6 × 128',
    code: `# Queries, keys and values, each 6 × 128.
q, k, v = linear(a, "h.0.attn.c_attn").split(d, dim=1)
# Split into 4 heads of 32 numbers, so 4 × 6 × 32.
q, k, v = (m.view(T, heads, -1).transpose(0, 1) for m in (q, k, v))
scores = q @ k.transpose(1, 2) / math.sqrt(d // heads)  # 4 × 6 × 6
scores = scores.masked_fill(~mask, float("-inf"))
weights = scores.softmax(-1)  # each row adds up to 1
# Mix the values, then put the heads side by side again.
out = (weights @ v).transpose(0, 1).reshape(T, d)
x = x + linear(out, "h.0.attn.c_proj")  # add it back
print(weights[0, -1])  # where "her" looks in head 1`,
    prints: (t) => {
      const w = t.layers[0].heads[0].weights;
      return fmt(w.data.subarray(LAST * w.cols, (LAST + 1) * w.cols));
    },
  },
  'enc-attn': {
    shape: '6 × 64 → 6 × 6 → 6 × 64, for one of 8 heads',
    code: `# One head, with the paper's 64 numbers.
q, k, v = torch.randn(3, T, 64)
# No mask, so each word sees every word. 6 × 6.
weights = (q @ k.T / math.sqrt(64)).softmax(-1)
out = weights @ v`,
  },
  'cross-attn': {
    shape: '6 × 64 and 9 × 64 → 6 × 9 → 6 × 64',
    code: `src = torch.randn(9, 64)  # the encoded source, 9 words
q = torch.randn(T, 64)  # the translation so far
# Each word of the translation looks at the source. 6 × 9.
weights = (q @ src.T / math.sqrt(64)).softmax(-1)
out = weights @ src`,
  },
  ffn: {
    shape: '6 × 128 → 6 × 512 → 6 × 128',
    code: `hidden = linear(b, "h.0.mlp.c_fc")  # 6 × 512, from the norm above
hidden = F.gelu(hidden, approximate="tanh")
# Back to 6 × 128, then add it.
x = x + linear(hidden, "h.0.mlp.c_proj")
print(x[-1, :4])  # what leaves block 1`,
    prints: (t) => head4(t.layers[0].output),
  },
  stack: {
    shape: '6 × 128 → 6 × 128, 4 times',
    code: `# The same steps as above, for block n.
def block(x, n):
    p = f"h.{n}."
    a = layer_norm(x, p + "ln_1")
    q, k, v = linear(a, p + "attn.c_attn").split(d, dim=1)
    q, k, v = (m.view(T, heads, -1).transpose(0, 1) for m in (q, k, v))
    scores = q @ k.transpose(1, 2) / math.sqrt(d // heads)
    weights = scores.masked_fill(~mask, float("-inf")).softmax(-1)
    out = (weights @ v).transpose(0, 1).reshape(T, d)
    x = x + linear(out, p + "attn.c_proj")
    b = layer_norm(x, p + "ln_2")
    hidden = F.gelu(linear(b, p + "mlp.c_fc"), approximate="tanh")
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
      const row = Array.from(
        t.logits.data.subarray(LAST * t.logits.cols, (LAST + 1) * t.logits.cols),
      );
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

/** The same tour for the Llama-style model in models/llama-tiny. Run LLAMA_SETUP first. */
export const LLAMA_SETUP = `import math, torch
import torch.nn.functional as F
from safetensors.torch import load_file

torch.set_printoptions(precision=4, sci_mode=False)
w = load_file("model.safetensors")  # models/llama-tiny in this repo
# "Lily wanted to play with her"
ids = torch.tensor([665, 408, 266, 324, 329, 336])
T, d, heads, kv_heads = len(ids), 128, 4, 2
dh = d // heads
mask = torch.ones(T, T).tril().bool()

def linear(x, name):
    return x @ w[name + ".weight"].T  # no biases

def rms_norm(x, name):
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w[name + ".weight"]

def rope(x):  # x is heads × T × 32
    i = torch.arange(0, dh, 2)
    angle = torch.arange(T)[:, None] * 10000 ** (-i / dh)  # T × 16
    cos, sin = angle.cos(), angle.sin()
    even, odd = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out`;

export const LLAMA_SECOND_NORM: Snippet = {
  shape: '6 × 128 → 6 × 128',
  code: `b = rms_norm(x, "h.0.norm_2")  # norm again, before feed forward
print(b[-1, :4])`,
  prints: (t) => head4(t.layers[0].ln2.out),
};

export const LLAMA_PYTORCH: Partial<Record<PartId, Snippet>> = {
  input: PYTORCH.input,
  embedding: {
    shape: '6 ids → 6 × 128',
    code: `# One row of 128 numbers per token, and no position table.
x = w["wte.weight"][ids]
print(x[-1, :4])  # the first four numbers of "her"`,
    prints: (t) => head4(t.tokenEmbeddings),
  },
  'add-norm': {
    shape: '6 × 128 → 6 × 128',
    code: `a = rms_norm(x, "h.0.norm_1")  # scale only, no mean, no shift
print(a[-1, :4])`,
    prints: (t) => head4(t.layers[0].ln1.out),
  },
  'masked-attn': {
    shape: '6 × 128 → 4 heads of 6 × 6 weights → 6 × 128',
    code: `# 4 query heads, but only 2 key and value heads, each 6 × 32.
q = linear(a, "h.0.attn.wq").view(T, heads, dh).transpose(0, 1)
k = linear(a, "h.0.attn.wk").view(T, kv_heads, dh).transpose(0, 1)
v = linear(a, "h.0.attn.wv").view(T, kv_heads, dh).transpose(0, 1)
q, k = rope(q), rope(k)  # turn each pair by its position
# Heads 1 and 2 share key and value head 1, heads 3 and 4 share head 2.
k, v = k.repeat_interleave(2, 0), v.repeat_interleave(2, 0)
scores = q @ k.transpose(1, 2) / math.sqrt(dh)
weights = scores.masked_fill(~mask, float("-inf")).softmax(-1)
out = (weights @ v).transpose(0, 1).reshape(T, d)
x = x + linear(out, "h.0.attn.wo")  # add it back
print(weights[0, -1])  # where "her" looks in head 1`,
    prints: (t) => {
      const w = t.layers[0].heads[0].weights;
      return fmt(w.data.subarray(LAST * w.cols, (LAST + 1) * w.cols));
    },
  },
  ffn: {
    shape: '6 × 128 → 2 × 6 × 344 → 6 × 128',
    code: `gate, up = linear(b, "h.0.mlp.w_gate"), linear(b, "h.0.mlp.w_up")
hidden = F.silu(gate) * up  # the gate opens or closes each number
x = x + linear(hidden, "h.0.mlp.w_down")  # back to 6 × 128, then add
print(x[-1, :4])  # what leaves block 1`,
    prints: (t) => head4(t.layers[0].output),
  },
  stack: {
    shape: '6 × 128 → 6 × 128, 4 times',
    code: `# The same steps as above, for block n.
def block(x, n):
    p = f"h.{n}."
    a = rms_norm(x, p + "norm_1")
    q = linear(a, p + "attn.wq").view(T, heads, dh).transpose(0, 1)
    k = linear(a, p + "attn.wk").view(T, kv_heads, dh).transpose(0, 1)
    v = linear(a, p + "attn.wv").view(T, kv_heads, dh).transpose(0, 1)
    q, k = rope(q), rope(k)
    k, v = k.repeat_interleave(2, 0), v.repeat_interleave(2, 0)
    scores = q @ k.transpose(1, 2) / math.sqrt(dh)
    weights = scores.masked_fill(~mask, float("-inf")).softmax(-1)
    x = x + linear((weights @ v).transpose(0, 1).reshape(T, d), p + "attn.wo")
    b = rms_norm(x, p + "norm_2")
    hidden = F.silu(linear(b, p + "mlp.w_gate")) * linear(b, p + "mlp.w_up")
    return x + linear(hidden, p + "mlp.w_down")

x = w["wte.weight"][ids]
for n in range(4):
    x = block(x, n)
print(x[-1, :4])`,
    prints: (t) => head4(t.layers[t.layers.length - 1].output),
  },
  'final-norm': {
    shape: '6 × 128 → 6 × 128',
    code: `u = rms_norm(x, "norm_f")
print(u[-1, :4])`,
    prints: (t) => head4(t.lnFinal.out),
  },
  linear: PYTORCH.linear,
  softmax: PYTORCH.softmax,
  output: PYTORCH.output,
};
