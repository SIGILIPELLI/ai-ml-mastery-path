---
description: "Distributed & Accelerated Training — Single-GPU training (or CPU, as in earlier modules) eventually hits a wall: the model or dataset is too large, or…"
---

# 07 · Distributed & Accelerated Training

Single-GPU training (or CPU, as in earlier modules) eventually hits a wall:
the model or dataset is too large, or training takes too long. This module
covers what a GPU actually accelerates, data parallelism across multiple
GPUs, and when scaling out is genuinely worth the added complexity.

## What a GPU actually speeds up

```python
import torch, time

size = 4096
a = torch.randn(size, size)
b = torch.randn(size, size)

start = time.time()
c_cpu = a @ b
cpu_time = time.time() - start
print(f"CPU matmul: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu, b_gpu = a.to("cuda"), b.to("cuda")
    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()      # GPU ops are async -- wait for completion before timing
    gpu_time = time.time() - start
    print(f"GPU matmul: {gpu_time:.3f}s  ({cpu_time / gpu_time:.0f}x faster)")
```

Neural network training is dominated by large matrix multiplications
(every `nn.Linear`, every attention `Q @ K.T`) — exactly the operation
GPUs parallelize across thousands of cores simultaneously.

## Data parallelism: replicate the model, split the batch

```python
import torch.nn as nn

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)      # simplest form: single-process, multi-GPU
model.to("cuda" if torch.cuda.is_available() else "cpu")
```

`DataParallel` is the simplest approach but has known bottlenecks
(gradient aggregation happens on one GPU). Production code uses
`DistributedDataParallel` (DDP) instead — one process per GPU, no single
bottleneck device.

## DistributedDataParallel: the production pattern

```python
# train_ddp.py -- launched with: torchrun --nproc_per_node=4 train_ddp.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

local_rank = setup()
model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)).to(local_rank)
ddp_model = DDP(model, device_ids=[local_rank])

optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# In the training loop, each of the 4 processes handles a different data shard:
# logits = ddp_model(xb)         -- forward pass, local to this GPU
# loss.backward()                -- DDP automatically all-reduces gradients across GPUs here
# optimizer.step()               -- each GPU applies the *same* averaged update
```

Each process trains on its own shard of the batch; `backward()` triggers an
automatic **all-reduce** that averages gradients across all processes
*before* `optimizer.step()`, so every replica ends the step with identical
weights despite having seen different data.

## Worked example: effective batch size and learning rate scaling

```python
per_gpu_batch = 32
n_gpus = 4
effective_batch = per_gpu_batch * n_gpus     # 128
base_lr = 1e-3
scaled_lr = base_lr * (effective_batch / 32)  # linear scaling rule
print(f"effective batch: {effective_batch}, scaled lr: {scaled_lr}")   # 128, 0.004
```

## When to scale out — and when not to

| Situation | Recommendation |
|---|---|
| Model + batch fit on one GPU, training finishes in hours | Don't bother — DDP adds real complexity |
| Model doesn't fit in one GPU's memory | Model/tensor parallelism (splits the *model*, not just data) |
| Training takes days, deadline is tighter | Data parallelism across GPUs is the standard fix |
| Dataset itself doesn't fit on one machine | Distributed data loading + sharding |

## Cheat sheet

| Concept | Tool |
|---|---|
| Quick multi-GPU, single process | `nn.DataParallel` (simple, has bottlenecks) |
| Production multi-GPU/multi-node | `DistributedDataParallel` + `torchrun` |
| Gradient sync across replicas | Automatic all-reduce inside `backward()` |
| Scaling learning rate with batch size | Linear scaling rule |

## How It Actually Works

**A GPU's speedup on matrix multiplication comes from massive parallelism
over independent scalar multiply-adds, not from any single operation being
"smarter."** Computing `C = A @ B` for `n×n` matrices requires `n³`
individual multiply-add operations, each independent of the others until
the final summation per output element. A CPU executes these largely
sequentially across a handful of cores (with some SIMD vectorization); a
GPU has thousands of simpler cores that can each compute a different
output element's partial products simultaneously, plus much higher memory
bandwidth to keep feeding them data. Neural network training reduces almost
entirely to sequences of exactly this operation (`nn.Linear`'s `Wx`,
attention's `QK^T` and `weights @ V`) — which is precisely why GPU
acceleration transfers so directly to deep learning: the hardware's
strength maps onto the software's dominant cost almost one-to-one, unlike,
say, decision tree building (Level 1 Module 05), whose sequential,
data-dependent branching parallelizes far less cleanly.

**DDP's all-reduce is what makes every replica converge to the *same*
model despite training on different data shards.** After each replica
computes `loss.backward()` on its own shard, it has a locally-computed
gradient for every parameter — one that reflects only that shard's data,
and would produce a different update than another replica's if applied
independently. All-reduce (typically a ring algorithm, communicating only
with neighbors in a virtual ring rather than all-to-all) sums each
parameter's gradient across all `N` processes and divides by `N`, so every
process ends up with the exact mathematical average of the `N`
shard-specific gradients — identical to what a single process would have
computed by processing all `N` shards' data in one giant batch and
averaging the loss. This equivalence (distributed average = single-machine
average over the combined batch) is the formal guarantee that DDP training
is mathematically equivalent to large-batch single-GPU training, not just
an approximation of it — which is also exactly why the effective batch
size is `per_gpu_batch × n_gpus`.

**The linear learning-rate scaling rule follows directly from how gradient
noise changes with batch size.** A gradient computed from a batch of size
`b` is a noisy estimate of the true full-dataset gradient, with variance
that shrinks roughly as `1/b` (averaging more samples reduces estimator
variance, the same statistical fact behind why larger training sets give
more reliable estimates generally). Quadrupling the batch size (32 → 128
across 4 GPUs) roughly quarters the gradient estimate's variance, which
means each step is a more reliable, lower-noise direction — justifying a
proportionally larger step size (`lr` scaled up by the same 4×) without the
increased step size introducing more instability than the original,
noisier, smaller-batch updates already had. This is a heuristic derived
from that variance argument, not an exact law — which is why large-scale
distributed training in practice usually pairs it with a brief LR
*warmup* period (small `lr` for the first few steps, ramping up) to avoid
instability before the model's weights have moved away from their initial,
more sensitive values.

## Exercise

Using the CPU-vs-GPU timing code above (or a CPU-only environment, timing
different matrix sizes instead), measure matmul time for sizes `[512,
1024, 2048, 4096]` and plot time against `size³` (the theoretical operation
count). Report whether the relationship looks linear, and if it deviates
at small sizes, explain — using the "parallelism only helps when there's
enough independent work to fill it" idea above — why a GPU's advantage
might shrink or vanish for very small matrices.
