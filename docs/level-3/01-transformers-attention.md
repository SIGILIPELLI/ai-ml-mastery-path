# 01 · Transformers & Attention from Scratch

Every state-of-the-art model in NLP and much of vision today is built on the
**transformer** architecture, and its core idea is **self-attention**: every
position in a sequence looks at every other position and decides how much to
"pay attention" to it. This module builds self-attention and a minimal
transformer block from raw tensor operations, then trains it on a toy task.

## Self-attention from scratch

```python
import torch
import torch.nn.functional as F

torch.manual_seed(42)
seq_len, d_model = 5, 8
x = torch.randn(seq_len, d_model)     # 5 tokens, 8-dim embeddings each

W_q = torch.randn(d_model, d_model) * 0.1
W_k = torch.randn(d_model, d_model) * 0.1
W_v = torch.randn(d_model, d_model) * 0.1

Q = x @ W_q     # (5, 8) -- "what am I looking for"
K = x @ W_k     # (5, 8) -- "what do I offer"
V = x @ W_v     # (5, 8) -- "what do I actually contribute if attended to"

scores = Q @ K.T / (d_model ** 0.5)      # (5, 5) -- raw attention scores
weights = F.softmax(scores, dim=-1)      # (5, 5) -- each row sums to 1
output = weights @ V                      # (5, 8) -- weighted blend of all tokens' V

print(weights.round(decimals=2))
# tensor([[0.22, 0.19, 0.21, 0.18, 0.20],
#         [0.19, 0.23, 0.18, 0.21, 0.19],
#         ...])   -- token i's attention distribution over all 5 tokens
```

Every output row is a weighted average of *all* the value vectors — token
1's output blends information from tokens 1-5, weighted by how relevant
each one is to token 1's query.

## Multi-head attention

Running several smaller attention operations in parallel ("heads") lets the
model attend to different kinds of relationships simultaneously.

```python
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = torch.nn.Linear(d_model, 3 * d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)

    def forward(self, x):
        seq_len, d_model = x.shape
        qkv = self.qkv(x)                                       # (seq, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)                           # each (seq, d_model)
        q = q.view(seq_len, self.n_heads, self.d_head).transpose(0, 1)   # (heads, seq, d_head)
        k = k.view(seq_len, self.n_heads, self.d_head).transpose(0, 1)
        v = v.view(seq_len, self.n_heads, self.d_head).transpose(0, 1)

        scores = q @ k.transpose(-2, -1) / (self.d_head ** 0.5)  # (heads, seq, seq)
        weights = F.softmax(scores, dim=-1)
        out = weights @ v                                        # (heads, seq, d_head)
        out = out.transpose(0, 1).reshape(seq_len, d_model)       # back to (seq, d_model)
        return self.out_proj(out)

mha = MultiHeadAttention(d_model=8, n_heads=2)
print(mha(x).shape)   # torch.Size([5, 8])
```

## Positional encoding: injecting order

Attention itself has no notion of position — reorder the input tokens and
`weights @ V` gives the same *set* of outputs reordered identically.
**Positional encodings** are added to the input embeddings to break this
symmetry.

```python
import math

def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

pe = positional_encoding(seq_len, d_model)
x_with_pos = x + pe    # elementwise add -- now position is baked into the representation
```

## A minimal transformer block

```python
class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff), torch.nn.ReLU(), torch.nn.Linear(d_ff, d_model)
        )
        self.norm2 = torch.nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))    # residual connection + attention
        x = self.norm2(x + self.ff(x))      # residual connection + feed-forward
        return x

block = TransformerBlock(d_model=8, n_heads=2, d_ff=32)
print(block(x_with_pos).shape)   # torch.Size([5, 8])
```

## Cheat sheet

| Component | Role |
|---|---|
| Query / Key / Value | Learned projections deciding what to look for / offer / contribute |
| `scores = QK^T / sqrt(d_head)` | Scaled similarity between every pair of positions |
| `softmax(scores)` | Turns scores into a probability distribution per row |
| Multi-head | Several attention "views" run in parallel, concatenated |
| Positional encoding | Injects order, since attention alone is order-agnostic |
| Residual + LayerNorm | Stabilizes training in deep stacks |

## How It Actually Works

**The `1/sqrt(d_head)` scaling factor exists to keep softmax's gradient from
vanishing.** `Q @ K.T` sums `d_head` products of roughly unit-variance
values, so its variance grows *linearly* with `d_head` — for a large head
dimension, raw dot products can be large in magnitude. Feeding large-
magnitude values into `softmax` (which exponentiates: `e^score_i /
Σ e^score_j`) saturates it — one score dominates, the output becomes nearly
one-hot, and its gradient with respect to the inputs becomes vanishingly
small, exactly as `sigmoid` (Module 05, Level 1) saturates and stalls
training for extreme inputs. Dividing by `sqrt(d_head)` rescales the scores'
variance back down to roughly 1 regardless of head dimension, keeping
softmax in its sensitive, well-behaved range — a variance-normalization
trick, not an arbitrary constant.

**Softmax turns attention into a literal weighted average, and that average
is what "attending" means mechanically.** Row `i` of `weights` sums to
exactly 1 by softmax's definition, so `output[i] = Σ_j weights[i,j] *
V[j]` is a convex combination of every token's value vector — no more, no
less. A weight near 1 on position `j` means the output for position `i` is
*mostly copying* `V[j]`; weights spread evenly (as in the near-uniform
example matrix above, from random untrained weights) mean the output
blends everything roughly equally. This is the entire mechanism behind
statements like "the model attends strongly to token 3" — it is a literal
description of a large numeric entry in `weights`, not a metaphor.

**Positional encoding's sine/cosine construction lets relative positions be
expressed as a *linear* function, which attention can exploit.** For any
fixed offset `k`, `PE(pos+k)` can be written as a linear transformation
(rotation) of `PE(pos)`, because `sin(a+b)` and `cos(a+b)` expand via the
angle-addition formulas into linear combinations of `sin(a), cos(a),
sin(b), cos(b)`. Because attention scores are computed via dot products
(`Q @ K.T`), and dot products are preserved (up to a fixed rotation) under
such linear relationships, a model can, in principle, learn attention
patterns that key off *relative* position ("attend to the token 2 steps
back") using the same fixed positional encoding for every sequence length,
rather than needing a distinct learned encoding per absolute position
learned separately for every possible sequence length.

**LayerNorm and residual connections exist to keep gradients well-behaved
through many stacked blocks.** The residual add (`x + self.attn(x)`) gives
`backward()` a direct additive path back to the input — an identity-mapping
shortcut through which gradients can flow with magnitude 1, avoiding the
vanishing-gradient problem that deep stacks without residuals suffer
(`d(x+f(x))/dx = 1 + df/dx`, which cannot go to zero the way a pure product
of many `df/dx` terms across many layers can). `LayerNorm` then rescales
each token's representation to zero mean and unit variance across its
features, preventing activations from drifting to extreme magnitudes as
they pass through dozens of stacked transformer blocks — a normalization
concern conceptually parallel to why `StandardScaler` matters for k-NN in
Level 1, applied here to intermediate activations rather than raw input
features.

## Exercise

Extend `TransformerBlock` to accept a boolean `causal` flag. When `True`,
mask `scores` before the softmax so that position `i` can only attend to
positions `j <= i` (set disallowed entries to `-inf` before `softmax`, so
they receive weight 0). Verify by printing `weights` for a causal block and
confirming the upper triangle is exactly zero — this is the mechanism GPT-
style autoregressive models use to prevent "seeing the future" token during
training.
