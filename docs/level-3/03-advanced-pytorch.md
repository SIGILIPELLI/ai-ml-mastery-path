---
description: "Advanced PyTorch Patterns — Modules 09 (Level 1) and 05 (Level 2) used simple in-memory tensors and plain training loops. Real projects need custom…"
---

# 03 · Advanced PyTorch Patterns

Modules 09 (Level 1) and 05 (Level 2) used simple in-memory tensors and
plain training loops. Real projects need custom `Dataset`s for data that
doesn't fit in memory, learning rate schedules, and mixed precision for
speed. This module covers the engineering patterns production PyTorch code
actually uses.

## Custom Dataset and DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CSVWindowDataset(Dataset):
    """Loads rows lazily and returns (features, label) pairs."""
    def __init__(self, n_rows=10000, n_features=20):
        rng = np.random.default_rng(42)
        self.X = rng.normal(size=(n_rows, n_features)).astype("float32")
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype("int64")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), self.y[idx]

ds = CSVWindowDataset()
loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
xb, yb = next(iter(loader))
print(xb.shape, yb.shape)   # torch.Size([64, 20]) torch.Size([64])
```

`__len__` and `__getitem__` are the only two methods required — `DataLoader`
handles batching, shuffling, and (with `num_workers > 0`) parallel loading
in background worker processes so the GPU is never left waiting on I/O.

## Learning rate schedulers

A fixed learning rate is rarely optimal for the whole training run — large
early on to make fast progress, small later to settle into a minimum.

```python
model = torch.nn.Sequential(torch.nn.Linear(20, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    for xb, yb in loader:
        logits = model(xb)
        loss = loss_fn(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()               # called once per epoch, after all batches
    print(f"epoch {epoch}  lr={scheduler.get_last_lr()[0]:.5f}")
# epoch 0  lr=0.00976
# epoch 5  lr=0.00345
# epoch 9  lr=0.00001
```

`CosineAnnealingLR` smoothly decays the rate along a cosine curve from
`1e-2` toward 0 over `T_max` epochs — no manual step-function tuning
required.

## Mixed precision training

Training in 16-bit floats instead of 32-bit roughly halves memory use and
can significantly speed up GPU training, with `autocast` keeping
numerically sensitive operations (like loss computation) in 32-bit
automatically.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

for xb, yb in loader:
    xb, yb = xb.to(device), yb.to(device)
    optimizer.zero_grad()
    with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
        logits = model(xb)
        loss = loss_fn(logits, yb)
    scaler.scale(loss).backward()     # scales the loss to avoid underflow in fp16 gradients
    scaler.step(optimizer)
    scaler.update()
```

## Worked example: gradient clipping against exploding gradients

```python
model2 = torch.nn.Sequential(torch.nn.Linear(20, 128), torch.nn.ReLU(), torch.nn.Linear(128, 2))
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.5)   # deliberately too large

for xb, yb in loader:
    logits = model2(xb)
    loss = loss_fn(logits, yb)
    optimizer2.zero_grad()
    loss.backward()
    total_norm = torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1.0)
    optimizer2.step()
    break
print(f"gradient norm before clipping: {total_norm:.2f}")   # can be large, e.g. 14.7
```

`clip_grad_norm_` rescales all gradients in place if their combined norm
exceeds `max_norm`, preventing a single unstable batch from taking a
destructively large step.

## Cheat sheet

| Pattern | Code |
|---|---|
| Custom data source | Subclass `Dataset`, implement `__len__`/`__getitem__` |
| Parallel data loading | `DataLoader(..., num_workers=N, pin_memory=True)` |
| LR schedule | `torch.optim.lr_scheduler.CosineAnnealingLR/StepLR/OneCycleLR` |
| Mixed precision | `torch.amp.autocast()` + `GradScaler` |
| Prevent exploding gradients | `torch.nn.utils.clip_grad_norm_` |

## How It Actually Works

**`DataLoader` workers parallelize Python-level I/O and preprocessing, not
GPU compute.** With `num_workers=2`, PyTorch spawns separate worker
processes, each independently calling `__getitem__` on copies of the
dataset object and assembling batches, which are then handed to the main
process through inter-process queues. This matters because Python's GIL
would otherwise serialize any CPU-bound work (file reads, image decoding,
augmentation) with the main training loop; overlapping that work in
separate processes means the next batch can be ready and waiting by the
time the GPU finishes the current one, rather than the GPU sitting idle
while Python assembles the next batch — `pin_memory=True` additionally
allocates the batch in page-locked host memory, which the GPU can DMA-copy
faster than regular pageable memory.

**Cosine annealing is a literal cosine function mapped onto the learning
rate axis, and its shape is why it works better than a linear decay in
practice.** `CosineAnnealingLR` sets
`lr(t) = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(π * t / T_max))`, which
starts at `lr_max` when `t=0`, decreases slowly at first, accelerates
through the middle, then flattens out again as it approaches `lr_min` near
`t=T_max`. The flattening at both ends is the deliberate feature: near the
start, a fast initial decay would waste the large steps still useful for
covering distance toward a good region of parameter space; near the end, a
slowly-flattening tail spends many epochs taking very small, careful steps
that let the optimizer settle precisely into a minimum instead of
overshooting it on the last few epochs — the same "smaller steps generalize
better" argument from Module 02's gradient-boosting learning rate, applied
to a schedule instead of a fixed constant.

**Mixed precision saves memory and time by using fp16 for matrix
multiplies while `autocast` and `GradScaler` protect against fp16's narrow
representable range.** fp16 uses half the bits of fp32, so tensors take half
the memory and matrix multiplications (the dominant cost in any linear or
attention layer) run faster on GPUs with dedicated fp16 hardware paths.
The catch: fp16 has a much smaller exponent range than fp32, so very small
gradient values can **underflow to exactly zero** during `backward()`.
`GradScaler` addresses this mechanically by multiplying the loss by a large
scale factor *before* `backward()` — this proportionally scales up every
computed gradient by the chain rule (multiplying a function's output by a
constant multiplies its gradient by the same constant), lifting small
gradients back into fp16's representable range — then `scaler.step()`
divides the gradients back down by that same factor before the actual
optimizer update, so the update itself is mathematically unaffected.
`autocast` separately keeps loss computation and other precision-sensitive
reductions in fp32 automatically, since summing many fp16 values can
introduce enough rounding error to destabilize training if left unmanaged.

**Gradient norm clipping is a hard cap applied uniformly across all
parameters, preserving gradient *direction* while limiting *magnitude*.**
`clip_grad_norm_` computes the combined L2 norm across every parameter's
gradient tensor (`total_norm = sqrt(Σ ‖grad_p‖²)` over all parameters `p`),
and if that exceeds `max_norm`, rescales *every* gradient tensor by the same
factor `max_norm / total_norm`. Because the same scalar multiplies every
parameter's gradient, the relative proportions between different
parameters' gradients — and therefore the overall descent *direction* —
are preserved exactly; only the step's total *size* shrinks. This is
mechanically why clipping fixes exploding-gradient instability (a single
batch producing enormous gradients, common with large learning rates or
deep recurrent architectures) without changing what direction training is
moving in, unlike simply lowering the learning rate globally, which would
shrink every step uniformly regardless of whether that particular batch
needed it.

## Exercise

Train `model2` from the worked example for 5 full epochs both with and
without `clip_grad_norm_` (same `lr=0.5`, same data, same seed). Track the
training loss after each epoch for both runs and report whether clipping
prevents the divergence that an unclipped, deliberately-too-large learning
rate would otherwise cause — connecting your observation to the
"preserves direction, limits magnitude" mechanism above.
