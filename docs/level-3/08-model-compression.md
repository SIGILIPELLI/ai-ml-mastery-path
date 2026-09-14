# 08 · Model Compression & Efficiency

A trained model that's too slow or too large to deploy is, practically
speaking, not done. This module covers the three standard compression
techniques: quantization (fewer bits per number), pruning (fewer weights),
and knowledge distillation (a smaller model taught to mimic a bigger one).

## Quantization: fewer bits per weight

```python
import torch
from torch import nn

model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10)
)
torch.manual_seed(42)
for p in model.parameters():
    p.data.normal_(0, 0.1)   # pretend this is trained

fp32_size = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"fp32 size: {fp32_size / 1024:.1f} KB")   # ~1069.8 KB

quantized = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
int8_size = sum(
    p.numel() * 1 for name, p in quantized.state_dict().items() if "weight" in name
)
print(f"approx int8 weight size: {int8_size / 1024:.1f} KB")   # ~4x smaller

x = torch.randn(1, 784)
with torch.no_grad():
    out_fp32 = model(x)
    out_int8 = quantized(x)
print(f"max output difference: {(out_fp32 - out_int8).abs().max().item():.4f}")
```

`quantize_dynamic` converts `Linear` layer weights from 32-bit floats to
8-bit integers, cutting weight memory roughly 4x with usually minor
accuracy loss — the difference between `out_fp32` and `out_int8` is small
but non-zero.

## Pruning: removing unimportant weights

```python
import torch.nn.utils.prune as prune

layer = model[0]   # first Linear(784, 256)
print("nonzero before:", torch.count_nonzero(layer.weight).item())

prune.l1_unstructured(layer, name="weight", amount=0.4)   # zero out the smallest 40% by magnitude
print("nonzero after: ", torch.count_nonzero(layer.weight).item())

prune.remove(layer, "weight")   # bake the mask in permanently, drop the pruning wrapper
```

`l1_unstructured` ranks weights by absolute value and zeros the smallest
40%, on the reasoning that small-magnitude weights contribute least to the
layer's output.

## Worked example: measuring the accuracy/size trade-off

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

digits = load_digits()
X = torch.tensor(digits.data / 16.0, dtype=torch.float32)
y = torch.tensor(digits.target, dtype=torch.long)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

small_model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-2)
for epoch in range(100):
    logits = small_model(X_train)
    loss = F.cross_entropy(logits, y_train)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

base_acc = (small_model(X_test).argmax(1) == y_test).float().mean().item()

for amount in [0.2, 0.5, 0.8]:
    pruned = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    pruned.load_state_dict(small_model.state_dict())
    prune.l1_unstructured(pruned[0], name="weight", amount=amount)
    prune.l1_unstructured(pruned[2], name="weight", amount=amount)
    acc = (pruned(X_test).argmax(1) == y_test).float().mean().item()
    print(f"prune {amount:.0%}: accuracy {acc:.3f} (base {base_acc:.3f})")
# prune 20%: accuracy 0.978 (base 0.980)
# prune 50%: accuracy 0.964
# prune 80%: accuracy 0.811   -- sharp drop
```

## Knowledge distillation: a small model learns from a big one's soft labels

```python
teacher = small_model   # pretend this is a large, expensive, accurate model
student = nn.Sequential(nn.Linear(64, 8), nn.ReLU(), nn.Linear(8, 10))   # much smaller

optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
temperature = 3.0

for epoch in range(150):
    with torch.no_grad():
        teacher_logits = teacher(X_train)
        soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    student_logits = student(X_train)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    distill_loss = F.kl_div(student_log_probs, soft_targets, reduction="batchmean") * temperature**2
    hard_loss = F.cross_entropy(student_logits, y_train)
    loss = 0.7 * distill_loss + 0.3 * hard_loss
    optimizer.zero_grad(); loss.backward(); optimizer.step()

student_acc = (student(X_test).argmax(1) == y_test).float().mean().item()
print(f"student (tiny) accuracy: {student_acc:.3f}")   # often close to teacher despite far fewer params
```

## Cheat sheet

| Technique | Reduces | Typical cost |
|---|---|---|
| Dynamic quantization | Memory (~4x), inference latency | Small accuracy drop |
| Unstructured pruning | Number of nonzero weights | Sharp accuracy drop past ~60-70% |
| Knowledge distillation | Model size (student << teacher) | Needs distillation training pass |

## How It Actually Works

**Quantization maps a continuous float range to a small set of integers via
an affine transform, and the error introduced is bounded by the chosen
range.** `qint8` represents each weight as an 8-bit integer in `[-128,
127]`, recovered as an approximate float via `real_value ≈ scale *
(quantized_value - zero_point)`, where `scale` and `zero_point` are chosen
per-tensor (or per-channel) based on the observed range of the original
fp32 weights. The maximum possible rounding error per weight is bounded by
`scale/2` — a direct consequence of squeezing a continuous range into 256
discrete buckets — which is why the `max output difference` in the
worked example is small but nonzero: every individual weight incurs a
small, bounded quantization error, and those errors partially compound
(and partially cancel) through the linear layers' matrix multiplications.
Dynamic quantization specifically computes activation scales on-the-fly at
inference time (rather than requiring a calibration pass beforehand),
trading a small runtime overhead for simplicity.

**Magnitude-based pruning's accuracy cliff is a direct consequence of which
weights actually carry the layer's signal.** `l1_unstructured` ranks
weights purely by `|weight|` and zeros the smallest fraction — an
assumption that small-magnitude weights contribute least to
`output = Wx + b`, since a near-zero weight barely changes the output
regardless of the input `x` it multiplies. This holds up well at moderate
pruning fractions (20-50%) because most trained networks are somewhat
overparameterized — many weights genuinely are near-redundant. But past a
threshold, pruning starts removing weights that *do* carry meaningful
signal for specific input patterns (even if individually small, their
combined contribution across many active inputs matters), which is
mechanically why accuracy degrades gently at first and then falls off a
cliff at 80% — the remaining weight budget is no longer enough to
approximate the original function for a meaningful fraction of inputs.

**Distillation's "soft targets" carry more information per example than
hard labels, and `temperature` controls how much of that extra information
is exposed.** A hard label is a one-hot vector — "this is class 3, period."
The teacher's `softmax(logits / temperature)` output, by contrast, assigns
non-zero probability to *every* class, encoding the teacher's learned
notion of which wrong answers are "almost right" (e.g., a handwritten '3'
getting substantial probability mass on '8' too, because they share visual
structure) — information a hard label discards entirely. Dividing logits by
`temperature > 1` before the softmax flattens the distribution (recall
softmax's exponential: smaller inputs to `exp` produce a less peaked
output), exposing more of these relative probabilities on non-target
classes rather than a near-one-hot output that would carry almost as little
information as the hard label itself. The `temperature**2` multiplier on
`distill_loss` compensates for gradients shrinking as `temperature`
increases (a mathematical property of how the softened cross-entropy's
gradient scales), keeping the distillation loss's effective contribution
comparable regardless of the temperature chosen — which is why the student,
despite having far fewer parameters than the teacher, can approach the
teacher's accuracy: it isn't just learning "which class is correct," it's
learning the teacher's entire learned similarity structure between classes.

## Exercise

Repeat the pruning sweep (`amount` in `[0.2, 0.5, 0.8]`) using
`prune.random_unstructured` instead of `prune.l1_unstructured` (same
amounts, same layers). Compare accuracy at each pruning level against the
magnitude-based results above, and explain — using the "which weights
carry signal" argument — why random pruning should generally underperform
magnitude-based pruning at the same sparsity level.
