---
description: "Model Serving at Scale — A trained model is useless until something can call it and get a prediction back, fast, under real load. This module covers…"
---

# 02 · Model Serving at Scale

A trained model is useless until something can call it and get a
prediction back, fast, under real load. This module covers wrapping a
model in a REST API with FastAPI, exporting to ONNX for portable/faster
inference, and batching — the core levers for serving at scale.

## A minimal serving API

```python
# serve.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("model.pkl")

class PredictRequest(BaseModel):
    features: list[float]

class PredictResponse(BaseModel):
    prediction: int
    probability: float

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    x = np.array(req.features).reshape(1, -1)
    proba = model.predict_proba(x)[0]
    pred = int(np.argmax(proba))
    return PredictResponse(prediction=pred, probability=float(proba[pred]))
```

```bash
uvicorn serve:app --host 0.0.0.0 --port 8000 --workers 4
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"features": [14.2, 20.1, 91.3, ...]}'
# {"prediction": 1, "probability": 0.983}
```

`pydantic` models (`PredictRequest`/`PredictResponse`) validate the request
shape and types automatically — a malformed request gets a clear 422 error
before it ever reaches the model.

## Exporting to ONNX for portable, optimized inference

```python
import torch
from torch import nn

model_pt = nn.Sequential(nn.Linear(30, 64), nn.ReLU(), nn.Linear(64, 2))
model_pt.eval()

dummy_input = torch.randn(1, 30)
torch.onnx.export(
    model_pt, dummy_input, "model.onnx",
    input_names=["features"], output_names=["logits"],
    dynamic_axes={"features": {0: "batch_size"}, "logits": {0: "batch_size"}},
)

import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
outputs = session.run(["logits"], {"features": dummy_input.numpy()})
print(outputs[0].shape)   # (1, 2)
```

ONNX Runtime applies graph-level optimizations (operator fusion, constant
folding) that are framework-agnostic — the exported model can be served
from Python, C++, or a mobile runtime without PyTorch installed at all.

## Batching requests for throughput

```python
import asyncio
import time

class BatchPredictor:
    def __init__(self, model, max_batch_size=32, max_wait_ms=10):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = []

    async def predict(self, features):
        future = asyncio.get_event_loop().create_future()
        self.queue.append((features, future))
        if len(self.queue) >= self.max_batch_size:
            await self._flush()
        else:
            asyncio.create_task(self._flush_after_timeout())
        return await future

    async def _flush_after_timeout(self):
        await asyncio.sleep(self.max_wait_ms / 1000)
        if self.queue:
            await self._flush()

    async def _flush(self):
        batch, self.queue = self.queue, []
        if not batch:
            return
        X = [f for f, _ in batch]
        preds = self.model.predict(X)         # one call, many rows -- the whole point
        for (_, future), pred in zip(batch, preds):
            if not future.done():
                future.set_result(pred)
```

Instead of one model call per request, requests arriving within a
`max_wait_ms` window are grouped into a single batched call — trading a
small amount of added latency per request for a large increase in total
throughput.

## Worked example: measuring the batching trade-off

```python
import numpy as np

def time_predictions(model, n_requests, batch_size):
    X = np.random.randn(n_requests, 30)
    start = time.time()
    n_batches = (n_requests + batch_size - 1) // batch_size
    for i in range(n_batches):
        batch = X[i*batch_size:(i+1)*batch_size]
        model.predict(batch)
    elapsed = time.time() - start
    return elapsed, elapsed / n_requests * 1000

for bs in [1, 8, 32, 128]:
    elapsed, per_req_ms = time_predictions(model, 1000, bs)
    print(f"batch_size={bs:4d}  total={elapsed:.3f}s  per-request={per_req_ms:.3f}ms")
# batch_size=1     total=0.412s  per-request=0.412ms
# batch_size=8     total=0.089s  per-request=0.089ms
# batch_size=32    total=0.041s  per-request=0.041ms
# batch_size=128   total=0.023s  per-request=0.023ms
```

## Cheat sheet

| Concern | Tool |
|---|---|
| Request/response validation | FastAPI + `pydantic` models |
| Framework-agnostic, optimized inference | ONNX export + `onnxruntime` |
| High throughput under load | Request batching (with a max-wait bound) |
| Concurrent request handling | `uvicorn --workers N`, `async def` handlers |

## How It Actually Works

**Batching amortizes fixed per-call overhead across many requests, which is
why throughput improves faster than the raw compute would suggest.** Each
call into a model (`model.predict(X)`) carries fixed costs independent of
`X`'s size — Python function call overhead, memory allocation, and for
matrix-multiply-heavy models, underutilized parallel hardware when `X` has
too few rows to fill all available compute lanes (the same GPU-parallelism
argument from Level 3 Module 07: a single-row matrix multiply leaves most
of a GPU's — or even a CPU's SIMD units' — capacity idle). Grouping 32
requests into one `model.predict(batch_of_32)` call pays that fixed
overhead once instead of 32 times, and lets the underlying linear-algebra
library parallelize across the full batch — which is mechanically why
per-request latency drops roughly 10-20x from `batch_size=1` to
`batch_size=32` in the worked example, far more than a linear "32 times
less overhead" would predict on its own, because the parallel hardware
utilization is also improving simultaneously.

**`max_wait_ms` exists because batching trades individual-request latency
for aggregate throughput, and that trade needs an explicit bound.** Without
a timeout, `BatchPredictor` would wait indefinitely for `max_batch_size`
requests to accumulate before running any of them — under low traffic, a
single request could wait forever. `_flush_after_timeout`'s
`asyncio.sleep(max_wait_ms / 1000)` guarantees that even a lone request
gets processed within a bounded delay, at the cost of not achieving the
full batching benefit when traffic is sparse. This is a genuine trade-off,
not a free win: setting `max_wait_ms` too high improves throughput under
load but adds real latency to every request during quiet periods, while
setting it too low (or to 0) approaches the `batch_size=1` case, forfeiting
most of the throughput gain — the right value depends on the application's
actual latency SLA.

**ONNX's speedup comes from graph-level optimizations that are only
possible once the model is represented as a static computation graph
rather than a live Python object.** Exporting to ONNX traces the model's
forward pass once (using `dummy_input`) and serializes the resulting
sequence of tensor operations into a static graph format, independent of
PyTorch's Python runtime. `onnxruntime` can then apply optimizations a
live PyTorch model can't easily benefit from at inference time: fusing
consecutive operations (e.g. a linear layer immediately followed by a ReLU
becomes one fused kernel instead of two separate memory-bound calls),
eliminating operations with statically-known outputs (constant folding),
and choosing hardware-specific optimized kernels for the target CPU or GPU
without any Python interpreter overhead per operation — which is why ONNX
inference is typically faster than the equivalent PyTorch `model(x)` call
even for the exact same mathematical function, purely from execution-plan
optimization rather than any change to what's being computed.

## Exercise

Add a `/health` endpoint to the FastAPI app that returns `{"status": "ok",
"model_version": ...}`, and a `/predict_batch` endpoint accepting a list of
feature vectors in one request (rather than the async queue-based batching
above — a simpler, explicit batch endpoint). Compare, using the timing
function from the worked example, the throughput of sending 1000 requests
one at a time to `/predict` versus sending them as 8 requests of 125 items
each to `/predict_batch`, and report the speedup.
