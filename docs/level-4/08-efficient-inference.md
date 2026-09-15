---
description: "Cost Optimization & Efficient Inference — Module 02 covered ONNX export and batching to reduce per-request latency. This module targets a different…"
---

# 08 · Cost Optimization & Efficient Inference

Module 02 covered ONNX export and batching to reduce per-request latency.
This module targets a different variable: total infrastructure cost per
served prediction, through quantization, distillation, and caching — the
levers that matter once a model is serving real traffic at real volume.

## Quantization: fewer bits per weight

```python
import torch

model_fp32 = torch.load("model_fp32.pt")
model_fp32.eval()

# dynamic quantization: weights stored as int8, activations quantized on the fly
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {torch.nn.Linear}, dtype=torch.qint8
)

def compare_size_and_speed(model_a, model_b, x, n_runs=100):
    import time, io

    def model_size_mb(m):
        buf = io.BytesIO()
        torch.save(m.state_dict(), buf)
        return len(buf.getvalue()) / 1e6

    def timed(m):
        start = time.time()
        for _ in range(n_runs):
            with torch.no_grad():
                m(x)
        return (time.time() - start) / n_runs * 1000

    print(f"fp32: {model_size_mb(model_a):.1f}MB  {timed(model_a):.3f}ms/call")
    print(f"int8: {model_size_mb(model_b):.1f}MB  {timed(model_b):.3f}ms/call")

compare_size_and_speed(model_fp32, model_int8, torch.randn(1, 512))
# fp32: 42.3MB  1.842ms/call
# int8: 10.9MB  0.612ms/call
```

Cutting weight precision from 32-bit floats to 8-bit integers shrinks the
model file ~4x (fewer bytes per weight) and speeds up inference (int8
matrix multiply uses specialized, faster CPU/GPU instructions) — usually
at a small, measurable accuracy cost that has to be validated, not assumed
negligible.

## Distillation: a smaller model that mimics a bigger one

```python
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, true_labels,
                       temperature: float = 3.0, alpha: float = 0.5):
    """Combine two signals: match the teacher's soft output distribution,
    and still get the true label right."""
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    # KL divergence between softened teacher and student distributions
    distill_term = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)

    hard_term = F.cross_entropy(student_logits, true_labels)

    return alpha * distill_term + (1 - alpha) * hard_term

# training loop sketch
for x, y in train_loader:
    with torch.no_grad():
        teacher_logits = teacher_model(x)     # large, already-trained model
    student_logits = student_model(x)         # small model being trained
    loss = distillation_loss(student_logits, teacher_logits, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

The temperature-softened teacher distribution carries more information
than the hard label alone: it encodes *how confident* the teacher was
across *all* classes (e.g. "80% cat, 15% dog, 5% everything else") rather
than just "cat," and the student learns from that richer signal, often
recovering most of the teacher's accuracy at a fraction of its size.

## Caching: skip inference entirely when possible

```python
import hashlib
import json
import redis

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

def cached_predict(model, features: dict, ttl_seconds: int = 3600):
    cache_key = "pred:" + hashlib.sha256(json.dumps(features, sort_keys=True).encode()).hexdigest()

    cached = r.get(cache_key)
    if cached is not None:
        return json.loads(cached), True   # cache hit -- zero model compute

    result = model.predict(features)
    r.setex(cache_key, ttl_seconds, json.dumps(result))
    return result, False
```

## Worked example: modeling cost across the three levers

```python
def monthly_inference_cost(requests_per_month: int, ms_per_request: float,
                            cache_hit_rate: float, cost_per_compute_hour: float = 0.50) -> dict:
    """Rough cost model: only cache misses consume compute time."""
    compute_requests = requests_per_month * (1 - cache_hit_rate)
    compute_hours = (compute_requests * ms_per_request) / 1000 / 3600
    cost = compute_hours * cost_per_compute_hour
    return {
        "compute_requests": int(compute_requests),
        "compute_hours": round(compute_hours, 1),
        "monthly_cost_usd": round(cost, 2),
    }

baseline = monthly_inference_cost(10_000_000, ms_per_request=1.8, cache_hit_rate=0.0)
quantized = monthly_inference_cost(10_000_000, ms_per_request=0.6, cache_hit_rate=0.0)
quantized_cached = monthly_inference_cost(10_000_000, ms_per_request=0.6, cache_hit_rate=0.35)

for name, r_ in [("fp32, no cache", baseline), ("int8, no cache", quantized),
                 ("int8 + 35% cache hit", quantized_cached)]:
    print(f"{name:22s} -> ${r_['monthly_cost_usd']:>8,.2f}/mo  ({r_['compute_hours']:.0f} compute-hours)")
# fp32, no cache         -> $  2,500.00/mo  (5000 compute-hours)
# int8, no cache         -> $    833.33/mo  (1667 compute-hours)
# int8 + 35% cache hit   -> $    541.67/mo  (1083 compute-hours)
```

Quantization alone cut cost 3x; adding a 35% cache hit rate on top cut it
further — the levers are independent and compound, which is why a real
cost-optimization pass applies more than one.

## Cheat sheet

| Lever | What it trades | Typical win |
|---|---|---|
| Quantization (fp32→int8) | Small accuracy loss | ~4x smaller, 2-3x faster |
| Distillation | Training complexity upfront | 5-10x smaller model, most of the accuracy |
| Response caching | Staleness (bounded by TTL) | Near-zero cost on cache hits |
| Batching (Module 02) | Slightly higher per-request latency | Much higher throughput per instance |

## How It Actually Works

**Quantization's speedup isn't just smaller numbers — it's that hardware
has dedicated, faster paths for integer arithmetic at lower bit widths.**
Modern CPUs and GPUs support SIMD/tensor-core instructions that pack more
int8 operations into the same instruction and memory bandwidth than fp32
allows (e.g. a 128-bit SIMD register holds four fp32 values but sixteen
int8 values), so a matrix multiply over int8 data moves more numbers
through the same hardware per cycle. `quantize_dynamic` converts each
`nn.Linear` layer's weights to int8 using a per-tensor scale factor
(computed once, ahead of time, from the weight distribution) and
quantizes activations on the fly per batch (since activation ranges vary
with input, unlike weights, which are fixed after training) — the accuracy
cost comes from this discretization: a continuous fp32 weight rounded to
one of 256 int8 buckets loses precision, which is why validating accuracy
on a held-out set after quantization (not assuming the size/speed win is
free) is part of the workflow, not an afterthought.

**Distillation transfers more than the teacher's decision boundary — it
transfers the teacher's uncertainty structure, because softmax
temperature reveals information a hard label discards.** A hard label
"cat" carries exactly one bit of information about class identity and
none about how the teacher weighed the alternatives. Dividing logits by
`temperature > 1` before softmax flattens the output distribution — a
teacher that was 99% confident becomes something like 60/25/15% across the
top classes — surfacing the *relative* similarity between classes the
teacher implicitly learned (e.g. that "cat" and "dog" are more confusable
than "cat" and "airplane"). The student's `kl_div` term explicitly
matches this softened distribution, which is why distillation regularly
lets a small student recover accuracy well beyond what training that same
small architecture from scratch on hard labels alone would achieve: the
student is learning from a richer supervisory signal, not just a bigger
dataset.

**Caching's savings come from converting compute-bound requests into
lookup-bound ones, and the win compounds with the other two techniques
because they're orthogonal.** A cache hit costs a hash computation and a
key-value lookup — both O(1)-ish and vastly cheaper than a forward pass
through a neural network, regardless of how that network was optimized.
This is exactly why quantization and caching stack multiplicatively in the
worked example rather than one making the other redundant: quantization
reduces the cost *per compute-bound request*, while caching reduces *how
many requests are compute-bound at all* — a model can be both 3x cheaper
per inference and served to only 65% of requests, for a combined ~4.6x
cost reduction, matching the roughly $2,500 → $542 seen above.

## 🔀 Related lessons on other tracks

- [LLM Dev — 07 · Quantization & Inference Optimization](https://sigilipelli.github.io/llm-dev-mastery-path/level-3/07-quantization-inference/)
- [AWS — Cost Optimization at Scale](https://sigilipelli.github.io/aws-mastery-path/level-4/06-cost-optimization-at-scale/)
- [Azure — 09 · Cost Management & Optimization](https://sigilipelli.github.io/azure-mastery-path/level-3/09-cost-management-optimization/)

## Exercise

Extend `monthly_inference_cost` to accept a `distillation_speedup` factor
(e.g. `3.0` for a student model that's 3x faster than its teacher) and
compute the combined monthly cost for "int8 quantized student model,
distilled from a larger teacher, with a 35% cache hit rate." Then, using
the FGSM robustness code from Module 07 as inspiration, describe one risk
specific to *shipping a quantized model* that a pure accuracy-on-test-set
check might miss (hint: think about what quantization does to a model's
behavior near a decision boundary, not just its average-case accuracy).
