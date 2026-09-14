# 01 · Production ML System Design

Everything so far has ended at a trained model in a notebook. A production
ML system is a set of connected pipelines: data flows in, features get
computed consistently, models get trained and served, and predictions feed
back into monitoring. This module maps that architecture and its most
common failure mode — training/serving skew.

## The anatomy of a production ML system

```
   [Raw data sources]
          |
   [Feature pipeline] ---> [Feature store] ---> [Online feature lookup]
          |                                             |
   [Training pipeline]                          [Serving/inference API]
          |                                             |
   [Model registry]  --------(deploy)-----------> [Production model]
          |                                             |
   [Offline evaluation]                         [Online monitoring/logging]
                                                          |
                                              [Feedback -> new training data]
```

Every arrow is a place where a bug or a mismatch can quietly enter the
system — this module focuses on the training/serving boundary, the single
most common source of production ML failures.

## Training/serving skew: the same bug, two codebases

```python
import pandas as pd

def compute_features_training(df: pd.DataFrame) -> pd.DataFrame:
    """Used once, in batch, over the full historical dataset."""
    out = df.copy()
    out["avg_purchase_30d"] = out.groupby("user_id")["amount"].transform(
        lambda s: s.rolling(30, min_periods=1).mean()
    )
    out["days_since_signup"] = (out["event_time"] - out["signup_time"]).dt.days
    return out

def compute_features_serving(user_row: dict) -> dict:
    """Used per-request, at inference time -- often reimplemented separately."""
    features = dict(user_row)
    # BUG: this recomputation used a *90-day* window, not 30 -- easy to miss in review
    features["avg_purchase_30d"] = user_row.get("avg_purchase_90d_cache", 0.0)
    features["days_since_signup"] = (pd.Timestamp.now() - user_row["signup_time"]).days
    return features
```

The training pipeline computes `avg_purchase_30d` correctly over a 30-day
window; the serving path, written separately for latency reasons, silently
substitutes a 90-day cached value. The model was trained on one
distribution of that feature and served a different one — with no error,
no crash, and no signal beyond degraded live accuracy that's hard to
localize.

## Worked example: catching skew before it ships

```python
def check_feature_parity(training_fn, serving_fn, sample_rows, tolerance=1e-6):
    mismatches = []
    for row in sample_rows:
        train_val = training_fn(pd.DataFrame([row])).iloc[0].to_dict()
        serve_val = serving_fn(row)
        for key in train_val:
            if key in serve_val:
                diff = abs(train_val[key] - serve_val[key]) if isinstance(train_val[key], (int, float)) else None
                if diff is not None and diff > tolerance:
                    mismatches.append((key, train_val[key], serve_val[key]))
    return mismatches

# A parity test run in CI against a sample of real rows, comparing both code paths
# on identical inputs -- would have caught the 30d vs 90d bug above before deploy.
```

## The single-pipeline alternative

The most reliable fix is architectural, not procedural: compute features
with **one** shared function/library, called both by the offline training
pipeline and the online serving path (often via a feature store, covered
next module).

```python
def compute_avg_purchase(purchases: pd.DataFrame, window_days: int = 30) -> float:
    """The ONE implementation, imported by both training and serving code."""
    cutoff = purchases["event_time"].max() - pd.Timedelta(days=window_days)
    return purchases[purchases["event_time"] >= cutoff]["amount"].mean()
```

## Cheat sheet

| Failure mode | Cause | Fix |
|---|---|---|
| Training/serving skew | Feature logic duplicated in two codebases | Shared feature functions / feature store |
| Silent data drift | Input distribution shifts after deploy | Monitoring (Module 05) |
| Stale features at inference | Online store lags the source data | Freshness SLAs, staleness checks |
| "Works in the notebook" | No parity test between offline/online paths | Parity tests in CI |

## How It Actually Works

**Training/serving skew degrades a model without changing a single weight,
because it changes the input distribution the fixed weights are applied
to.** A trained model is a fixed function `f(x)` learned to be accurate for
inputs `x` drawn from the *training* feature distribution. If the serving
path computes a feature differently (30-day window vs. 90-day), the
resulting `x_serving` is drawn from a *different* distribution than
`x_training` — the model's learned weights, tuned to the statistical
relationships in the 30-day version of that feature, are now being applied
to values that follow different statistics. No code crashes and no
metric alarms fire at the feature-computation layer, because both
implementations produce syntactically valid numbers — the degradation only
shows up several layers downstream, in prediction quality, which is
exactly why this failure mode is notoriously hard to trace back to its
root cause.

**A parity test mechanically converts an architectural risk into a testable
assertion by running both code paths on identical inputs and diffing the
outputs.** `check_feature_parity` takes the *same* raw row and feeds it
through both the training feature function and the serving feature
function, then compares each shared feature key numerically. Because both
functions receive identical inputs, any discrepancy in output *must* come
from a difference in the transformation logic itself (different window,
different rounding, different null-handling) rather than from different
input data — isolating the exact class of bug that would otherwise
manifest only as unexplained accuracy degradation weeks after deployment.
Running this as an automated CI check on a sample of real rows means the
30-day-vs-90-day bug above would fail a test *before* merge, rather than
being discovered via a slow, expensive investigation into "why did
accuracy quietly drop."

**A shared feature function structurally eliminates skew by making
divergence impossible, not merely detected.** `compute_avg_purchase`
called from both the training pipeline (in batch, over historical data)
and the serving path (per-request, over recent data) uses the literal same
code — there is no second implementation that could drift out of sync with
the first, because there is only one implementation. This is the
architectural principle a feature store (Module 03) generalizes: rather
than relying on discipline (remembering to update both copies whenever
logic changes) or detection (parity tests catching drift after the fact),
it removes the duplication that makes skew possible in the first place —
the online and offline paths both read from, or are both computed by, the
same underlying feature definition.

## Exercise

Take the buggy `compute_features_serving` function above and rewrite it to
call the shared `compute_avg_purchase` function instead of the cached
90-day value. Then write a small `check_feature_parity`-style test with 3
sample rows (varying purchase history) that would have failed against the
*original* buggy version and passes against your fix — demonstrating the
parity-test workflow end to end.
