# 09 · Neural Network Fundamentals with PyTorch

Neural networks power modern AI — vision, speech, language models — and
PyTorch is the dominant way to build them. Under the hype, a neural net is
just layers of linear regression with non-linear "squashes" between them,
trained by gradient descent. This module builds that understanding from the
ground up: tensors, autograd, a small multi-layer perceptron (MLP), and the
training loop — the four ideas that everything from here to GPT is built on.

## Tensors: NumPy arrays with superpowers

A `torch.Tensor` is an N-D array like `ndarray`, with two additions: it can
live on a GPU, and it can track gradients.

```python
import torch

t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(t.shape, t.dtype)     # torch.Size([2, 2]) torch.float32

print(t * 2)                # elementwise, just like NumPy
print(t @ t)                # matrix multiply
print(t.mean(dim=0))        # tensor([2., 3.])  -- dim ~ NumPy's axis

# NumPy interop is trivial:
import numpy as np
a = np.array([1.0, 2.0, 3.0])
t2 = torch.from_numpy(a)
back = t2.numpy()
```

Everything you learned about shapes, vectorization, and axis-wise reduction
transfers directly — `dim` instead of `axis`, `float32` (not 64) as the
default working dtype.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)               # "cpu" is fine for everything in this module
```

## Autograd: derivatives for free

Training means adjusting weights to reduce a loss, which requires the loss's
derivative with respect to every weight. PyTorch records the operations you
perform and computes all those derivatives automatically:

```python
w = torch.tensor(3.0, requires_grad=True)
loss = (w - 5.0) ** 2       # loss is smallest at w = 5
loss.backward()             # compute d(loss)/dw
print(w.grad)               # tensor(-4.)  -- indeed 2*(3-5) = -4
```

The gradient says "loss decreases if w increases" (negative slope), so
gradient descent nudges `w` a small step *against* the gradient:
`w ← w − lr · grad`. Repeat thousands of times over all weights at once and
that's deep learning — everything else is bookkeeping.

## A dataset and a network

We'll classify `make_moons` — the two interleaved crescents from Module 06
that defeat any straight-line classifier:

```python
import torch
from torch import nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_moons(n_samples=1000, noise=0.15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
scaler = StandardScaler().fit(X_train)          # same rules as always:
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

# to float32 tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
y_test_t  = torch.tensor(y_test, dtype=torch.long)
```

The model — an MLP with one hidden layer:

```python
torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(2, 16),   # 2 inputs -> 16 hidden units (a linear layer = Wx + b)
    nn.ReLU(),          # non-linearity: max(0, x)
    nn.Linear(16, 2),   # 16 hidden -> 2 output scores ("logits"), one per class
)
print(sum(p.numel() for p in model.parameters()), "parameters")   # 82
```

The `ReLU` between the linear layers is what makes this more than linear
regression: without it, two stacked linear layers collapse into one linear
map, and the crescents stay unseparable. With it, the network can bend its
decision boundary.

## The training loop: forward → loss → backward → step

This loop *is* deep learning. Every framework wraps it; PyTorch shows it to
you honestly:

```python
loss_fn   = nn.CrossEntropyLoss()                       # standard classification loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    model.train()
    logits = model(X_train_t)              # 1. forward: predictions
    loss   = loss_fn(logits, y_train_t)    # 2. loss: how wrong are we?
    optimizer.zero_grad()                  #    (clear old gradients)
    loss.backward()                        # 3. backward: gradients via autograd
    optimizer.step()                       # 4. step: nudge every weight

    if epoch % 50 == 0:
        print(f"epoch {epoch:3d}  loss {loss.item():.4f}")
# epoch   0  loss 0.7075
# epoch  50  loss 0.2242
# epoch 100  loss 0.1103
# epoch 150  loss 0.0674
```

Line by line: the **forward pass** computes predictions; **CrossEntropyLoss**
converts logits + true labels into a single wrongness number;
**`backward()`** fills `p.grad` for all 82 parameters; **`step()`** applies
the Adam update rule. `zero_grad()` matters because PyTorch *accumulates*
gradients by default — forget it and updates go haywire (a classic first
bug).

Evaluate:

```python
model.eval()
with torch.no_grad():                       # no gradient tracking needed
    test_logits = model(X_test_t)
    pred = test_logits.argmax(dim=1)
    acc = (pred == y_test_t).float().mean()
print(f"test accuracy: {acc.item():.3f}")   # test accuracy: ~0.99
```

~99% on data a linear model caps out around 85% on — the hidden layer
learned the curve. (`model.eval()`/`model.train()` toggle layers like
dropout; harmless here, a vital habit later.)

For datasets too big to feed at once, PyTorch batches with `DataLoader` —
same loop, wrapped in `for xb, yb in loader:`:

```python
from torch.utils.data import TensorDataset, DataLoader
loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                    batch_size=64, shuffle=True)
```

Each pass through the whole dataset is an **epoch**; each `xb` is a
**mini-batch**. Mini-batching is why deep learning scales to datasets that
don't fit in memory.

## When deep learning is — and isn't — the right tool

| Situation | Reach for |
|-----------|-----------|
| Tabular data, hundreds–100k rows | Gradient-boosted trees / random forests — they still routinely beat neural nets here. |
| Images, audio, text | Neural networks — nothing else is close. |
| Need interpretable coefficients | Linear/logistic regression. |
| Tiny dataset (< a few hundred rows) | Simple models; a neural net will memorize it. |
| Huge data + complex patterns + GPU budget | Neural networks shine. |

The honest summary for Level 1: for the tabular problems in this course,
scikit-learn models are usually equal or better with far less fuss. You learn
PyTorch now because the *concepts* (tensors, autograd, the loop) are the
foundation for Level 2's CNNs and Level 3's transformers.

## Cheat sheet

| Task | Code |
|------|------|
| Tensor from data | `torch.tensor(arr, dtype=torch.float32)` |
| Labels for CrossEntropy | dtype `torch.long` |
| Track gradients | `requires_grad=True`; `loss.backward()`; `.grad` |
| Define an MLP | `nn.Sequential(nn.Linear(i,h), nn.ReLU(), nn.Linear(h,o))` |
| Classification loss | `nn.CrossEntropyLoss()` (takes raw logits) |
| Regression loss | `nn.MSELoss()` |
| Optimizer | `torch.optim.Adam(model.parameters(), lr=0.01)` |
| The loop | forward → `loss` → `zero_grad()` → `backward()` → `step()` |
| Inference mode | `model.eval()` + `with torch.no_grad():` |
| Predicted class | `logits.argmax(dim=1)` |
| Reproducibility | `torch.manual_seed(42)` |
| Mini-batches | `DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)` |

## How It Actually Works

**Autograd builds a computation graph as operations run, then walks it
backward.** Every tensor operation you perform on a `requires_grad=True`
tensor doesn't just compute a value — it also records, on a `grad_fn`
attached to the result, which operation produced it and from which input
tensors. This builds up a directed acyclic graph (DAG) of the whole forward
pass, node by node, as it executes ("define-by-run"). `loss.backward()`
then walks that DAG from the loss node back to every leaf tensor, applying
the **chain rule of calculus** at each node: if `y = f(x)`, the gradient
flowing into `x` is `dL/dx = dL/dy · dy/dx`, where `dy/dx` is a small,
hand-derived local formula PyTorch knows for that specific operation (for
`w - 5`, `dy/dw = 1`; for squaring, `d(y²)/dy = 2y`). Composing these local
derivatives back through the graph is exactly how `loss.backward()` fills
`w.grad` with `2*(3-5) = -4` — no numerical approximation, no finite
differences, just mechanical application of the chain rule at each recorded
step. For a real network with 82 parameters, the same graph-walk computes
all 82 partial derivatives in one backward pass, which is what makes
training thousands of weights per step computationally tractable.

**Why gradient descent's update rule works.** The gradient `dL/dw` points in
the direction of *steepest increase* of the loss with respect to `w`;
moving `w` a small step in the *opposite* direction (`w ← w - lr·grad`) is
therefore guaranteed, for a small enough step, to decrease the loss —
that's the entire mathematical justification, a first-order Taylor
approximation: `L(w - lr·grad) ≈ L(w) - lr·grad²`, which is less than
`L(w)` whenever `lr` and `grad²` are positive. `lr` (learning rate) scales
how big that step is; too large and the linear approximation breaks down
(the update overshoots and the loss can increase instead — the
"divergence" the exercise asks you to watch for with `lr=1.0`). Adam, used
here instead of plain gradient descent, additionally tracks a running
average of past gradients and their squares per parameter, adapting the
effective step size per-weight — which is why it converges faster and more
reliably than a fixed-rate update, without changing the underlying
chain-rule math.

**Why ReLU is not a cosmetic detail — it's what makes depth meaningful.** A
linear layer computes `Wx + b`. Stacking two linear layers gives
`W2(W1x + b1) + b2 = (W2W1)x + (W2b1 + b2)`, which is algebraically just
*another* linear layer with weights `W2W1` — no matter how many linear
layers you stack, the composition collapses to one matrix multiply, capable
only of drawing a single straight decision boundary. `nn.ReLU()`
(`max(0, x)`, applied elementwise) breaks that collapse because it's not
linear: `f(a+b) ≠ f(a)+f(b)` in general. With a ReLU between them, the two
`Linear` layers cannot be algebraically merged into one, and the resulting
function is genuinely piecewise-linear — literally made of many flat linear
"facets" stitched together at the points where individual ReLU units switch
between passing their input through and outputting zero. With 16 hidden
units there are up to 16 such switching boundaries the network can place,
which is exactly the mechanism that lets it bend around the interleaved
moons instead of being stuck with one straight line — and exactly why
deleting `nn.ReLU()` caps accuracy at whatever a plain linear classifier
achieves, regardless of hidden-layer width.

**`CrossEntropyLoss` combines softmax and log-likelihood in one numerically
stable step.** The network's final `Linear(16, 2)` layer outputs raw,
unbounded "logits" — one real number per class. `CrossEntropyLoss`
internally applies **softmax** (`p_c = e^(z_c) / Σ_k e^(z_k)`, turning
logits into a probability distribution that sums to 1) and then computes
`-log(p_true_class)` — the negative log-probability the model assigned to
the correct class. A model that's confidently correct produces a loss near
0; one that's confidently wrong produces a loss that grows without bound,
which is precisely the pressure that drives `backward()`'s gradients:
weights are pushed to raise the logit of the true class and lower the
others, batch after batch, until the loss curve you see printed (0.71 →
0.07) flattens out.

## Exercise

Three experiments on the moons setup, changing one thing at a time: (1)
delete the `nn.ReLU()` — how good can the purely-linear network get, and why
does that ceiling exist? (2) restore ReLU but shrink the hidden layer to 2
units, then grow it to 64 — compare test accuracy and note diminishing
returns; (3) set `lr=1.0` and watch the loss curve — describe what
"divergence" looks like. Bonus: switch the data to
`make_circles(noise=0.1, factor=0.4)` and check the same 16-unit network
still works.
