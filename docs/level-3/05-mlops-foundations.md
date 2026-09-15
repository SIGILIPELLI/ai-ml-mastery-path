---
description: "MLOps Foundations — Module 04 tracked experiments. This module widens the lens to the full ML lifecycle: versioning data and models (not just code)…"
---

# 05 · MLOps Foundations

Module 04 tracked experiments. This module widens the lens to the full ML
lifecycle: versioning data and models (not just code), reproducible
environments, and the pipeline structure that takes a notebook experiment
into something a team can maintain.

## Why `git` alone isn't enough

Code changes are cheap to diff; multi-gigabyte datasets and model
checkpoints are not. **DVC** (Data Version Control) tracks large files the
way git tracks code, storing only lightweight pointers in git itself.

```bash
pip install dvc
dvc init
dvc add data/train.csv          # creates data/train.csv.dvc (a small pointer file)
git add data/train.csv.dvc .gitignore
git commit -m "Track training data v1 with DVC"

dvc remote add -d storage s3://my-bucket/dvc-store
dvc push                        # uploads the actual data to remote storage
```

`data/train.csv.dvc` is a tiny text file (a content hash + size) that git
tracks directly; the actual multi-GB file lives in the DVC remote and is
fetched on demand with `dvc pull`.

## Reproducible environments

```python
# requirements.txt (pinned, not loose)
# scikit-learn==1.4.2
# pandas==2.2.1
# xgboost==2.0.3
```

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "train.py"]
```

Pinned versions plus a Dockerfile mean "works on my machine" becomes "works
in this exact, reproducible container" — the same image runs identically
on a teammate's laptop, CI, and production.

## Structuring a pipeline: stages with explicit dependencies

```yaml
# dvc.yaml
stages:
  prepare:
    cmd: python prepare.py
    deps: [data/raw.csv, prepare.py]
    outs: [data/processed.csv]
  train:
    cmd: python train.py
    deps: [data/processed.csv, train.py]
    params: [n_estimators, max_depth]
    outs: [models/model.pkl]
    metrics: [metrics.json]
```

```bash
dvc repro       # runs only the stages whose deps changed since last run
```

## Worked example: detecting a stale pipeline

```python
import hashlib, json, os

def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]

def pipeline_is_stale(lock_path="dvc.lock", stage_deps=None):
    if not os.path.exists(lock_path):
        return True
    with open(lock_path) as f:
        recorded = json.load(f)
    for dep_path in stage_deps:
        if file_hash(dep_path) != recorded.get(dep_path):
            print(f"stale: {dep_path} changed since last successful run")
            return True
    return False

# Conceptually what `dvc repro` checks before deciding to rerun a stage
```

This is the mechanism (simplified) behind `dvc repro` skipping unchanged
stages: a content hash of each declared dependency is stored after a
successful run, and compared before the next.

## Cheat sheet

| Concern | Tool / pattern |
|---|---|
| Version large data/model files | `dvc add`, `dvc push`/`dvc pull` |
| Reproducible dependencies | Pinned `requirements.txt` |
| Reproducible runtime | `Dockerfile` |
| Pipeline with explicit stage dependencies | `dvc.yaml` + `dvc repro` |
| Skip re-running unchanged stages | Content hashing of declared deps |

## How It Actually Works

**DVC's pointer-file trick works because git and content-addressable
storage solve different problems well.** `dvc add` computes a content hash
(MD5/SHA) of the large file, moves the actual bytes into DVC's local cache
(keyed by that hash) and a configured remote, and writes a small `.dvc`
file containing just the hash, size, and path. Git — which is genuinely bad
at diffing and storing large binary blobs efficiently — now only ever sees
the tiny pointer file change, while the *actual* data lives in
storage designed for large objects (S3, GCS, or a shared filesystem).
`dvc checkout`/`dvc pull` reads the hash from the pointer file and fetches
the matching content from the cache/remote — meaning two different git
commits can reference two different multi-GB dataset versions while git's
own repository size stays tiny, because git never stores the data itself,
only a reference to it.

**`dvc repro`'s staleness check is literally the hashing pattern in the
worked example, generalized across a DAG of stages.** Before running a
stage, DVC computes a content hash of every file listed under `deps:` and
compares it against the hash recorded in `dvc.lock` from the last
successful run of that stage. If every dependency's hash matches, the
stage's output is guaranteed to be identical to last time (same inputs,
same code, same command → same output, assuming determinism), so DVC skips
re-running it and moves to checking the next stage — exactly mirroring
`pipeline_is_stale`'s logic. This is why changing `train.py` triggers a
rerun of the `train` stage but not `prepare` (whose only declared
dependencies, `data/raw.csv` and `prepare.py`, are untouched) — the
dependency graph, not a blanket "rerun everything," determines exactly
which stages actually need recomputation.

**A pinned `requirements.txt` and a Dockerfile solve two different layers
of the same non-determinism problem.** Loose version constraints
(`scikit-learn>=1.0`) let `pip install` silently resolve to whatever the
latest compatible release is *at install time* — different on different
days, since scikit-learn 1.5 might change a default hyperparameter or
numerical algorithm relative to 1.4.2, silently altering results without
any code change. Pinning exact versions removes that ambiguity for Python
packages specifically, but the *operating system*, system libraries, and
Python interpreter version can still differ across machines. A Docker
image freezes all of that: the base OS layer, system packages, Python
version, and — via the pinned `requirements.txt` — every Python
dependency, into one immutable, shareable artifact identified by an image
hash, which is the actual mechanism behind "runs identically everywhere":
not a promise, but the elimination of every layer of environment variation
that could otherwise cause it not to.

## Exercise

Set up a two-stage `dvc.yaml` (`prepare` producing a processed CSV,
`train` consuming it) for a small script of your own, run `dvc repro`
once, then touch only `prepare.py` (no functional change, just a comment)
and run `dvc repro` again. Observe which stage(s) actually re-execute, and
explain — using the dependency-hash mechanism above — why touching a file
that's declared as a dependency triggers a rerun even when its *content
hash* would be identical after a whitespace-only edit is reverted.
