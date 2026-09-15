---
description: "Experiment Tracking (MLflow & W&B) — By now you've trained dozens of model variants across print statements and notebooks. That doesn't scale — you can't…"
---

# 04 · Experiment Tracking (MLflow & W&B)

By now you've trained dozens of model variants across print statements and
notebooks. That doesn't scale — you can't compare 40 runs by memory. This
module covers logging parameters, metrics, and artifacts systematically
with MLflow, so every experiment is reproducible and comparable later.

## Logging a single run

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

data = load_breast_cancer(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.25, random_state=42, stratify=data.target
)

mlflow.set_experiment("breast-cancer-rf")

with mlflow.start_run(run_name="rf-baseline"):
    params = {"n_estimators": 200, "max_depth": 5, "random_state": 42}
    mlflow.log_params(params)

    model = RandomForestClassifier(**params).fit(X_train, y_train)
    pred = model.predict(X_test)

    mlflow.log_metric("accuracy", accuracy_score(y_test, pred))
    mlflow.log_metric("f1", f1_score(y_test, pred))
    mlflow.sklearn.log_model(model, "model")

print("run recorded — view with `mlflow ui`")
```

Every call to `start_run()` creates a new, independently addressable
record: parameters, metrics, and the serialized model itself, all tied
together and timestamped.

## Sweeping and comparing runs programmatically

```python
results = []
for n_est in [50, 200, 500]:
    for max_depth in [3, 5, None]:
        with mlflow.start_run(run_name=f"rf-{n_est}-{max_depth}"):
            params = {"n_estimators": n_est, "max_depth": max_depth, "random_state": 42}
            mlflow.log_params(params)
            model = RandomForestClassifier(**params).fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            mlflow.log_metric("accuracy", acc)
            results.append({**params, "accuracy": acc})

import pandas as pd
print(pd.DataFrame(results).sort_values("accuracy", ascending=False).head(3))
```

## Querying past runs programmatically

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name("breast-cancer-rf")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=3,
)
for run in runs:
    print(run.data.params, "->", run.data.metrics.get("accuracy"))
```

This is the payoff: three weeks from now, "which config got the best
recall?" is a query, not an archaeology project through old notebooks.

## Worked example: reproducing a logged run exactly

```python
best_run = runs[0]
logged_params = {k: (int(v) if v.isdigit() else (None if v == "None" else v))
                 for k, v in best_run.data.params.items()}
reproduced = RandomForestClassifier(**logged_params).fit(X_train, y_train)
print("reproduced accuracy:", accuracy_score(y_test, reproduced.predict(X_test)))
print("originally logged:  ", best_run.data.metrics["accuracy"])
```

Because every hyperparameter that affects the model was logged (including
`random_state`), retraining from the logged params reproduces the exact
same accuracy — the entire point of tracking.

## Cheat sheet

| Task | Code |
|---|---|
| Group runs | `mlflow.set_experiment(name)` |
| Start a tracked run | `with mlflow.start_run():` |
| Log hyperparameters | `mlflow.log_params({...})` |
| Log a metric | `mlflow.log_metric(name, value)` |
| Save the model artifact | `mlflow.sklearn.log_model(model, "model")` |
| Browse runs | `mlflow ui` (local web dashboard) |
| Query programmatically | `MlflowClient().search_runs(...)` |

## How It Actually Works

**`mlflow.start_run()` creates a directory/database record before any
logging call, which is why later calls can attach to it.** Entering the
`with` block generates a unique run ID and writes an initial metadata
record (start time, experiment ID, status) to MLflow's backing store — a
local `mlruns/` directory by default, or a database/server in production.
Every subsequent `log_params`/`log_metric`/`log_model` call inside the
block is a separate write tagged with that same run ID, and exiting the
`with` block (even via an exception) marks the run's end time and final
status. This is why a run's parameters, metrics, and model artifact stay
linked as one coherent record even though they're logged via separate
function calls at different points in the script — the run ID, held
implicitly by the active context manager, is the join key.

**The sweep's comparability depends entirely on which parameters are
actually captured — an unlogged hyperparameter is invisible to every later
query.** `search_runs(order_by=["metrics.accuracy DESC"])` sorts purely on
what was written to the metrics store; if a run's code silently used a
different `random_state`, feature set, or preprocessing step that was never
passed to `log_params`, two runs with identical logged parameters could
still produce different results with no recorded explanation. The
worked example's reproduction only works because every parameter that
affects `RandomForestClassifier`'s behavior (`n_estimators`, `max_depth`,
`random_state`) was captured in `log_params` before training — proving,
mechanically, that MLflow's "reproducibility" guarantee is only as strong
as the completeness of what a script chooses to log, not something MLflow
enforces automatically.

**`log_model` serializes the fitted estimator's actual state, not just a
description of it.** `mlflow.sklearn.log_model` pickles the trained
`RandomForestClassifier` object — including the specific 200 (or 500) fitted
decision trees, their learned split thresholds, and leaf values, exactly as
they exist in memory after `.fit()` — and stores that alongside a small
metadata file recording the library version and a standardized "flavor"
interface. This is why a logged model can be loaded and used for inference
later (`mlflow.sklearn.load_model(uri)`) without rerunning `.fit()` at
all: the artifact *is* the trained parameters, not a recipe for retraining
them, which is a categorically different (and much faster) form of
reproducibility than re-executing the training script from logged
hyperparameters.

## Exercise

Extend the sweep loop to also log, for each run, the top-5 permutation
importances (Level 2 Module 07) as a metric per feature (e.g.
`mlflow.log_metric(f"importance_{feature_name}", value)`). Then use
`search_runs` to find whether the best-accuracy configuration also has the
most stable top feature across runs, or whether different hyperparameter
settings lead the model to rely on different features — a question that
would be nearly impossible to answer without systematic logging.
