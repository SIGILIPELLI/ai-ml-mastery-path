# 05 · Monitoring, Drift Detection & Retraining

A model that passed every CI gate can still degrade in production, because
the world keeps changing after deployment while the model stays frozen.
This module covers what to monitor, how to detect drift statistically
rather than by gut feel, and how to close the loop with automated
retraining triggers.

## What to log at prediction time

```python
# serve.py — log everything needed to detect drift later
import json
import time
from datetime import datetime, timezone

def log_prediction(features: dict, prediction, probability: float, model_version: str):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "features": features,
        "prediction": prediction,
        "probability": probability,
    }
    # append-only log; a real system ships this to Kafka/S3/a feature store,
    # not a local file, but the shape is the same
    with open("prediction_log.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
```

Without this log, drift detection is impossible after the fact — you need
the *actual inputs the model saw in production*, not just the training
distribution, to compare against.

## Detecting feature drift: population stability index

```python
import numpy as np

def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index between a training distribution
    (expected) and a production distribution (actual) for one feature."""
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf

    expected_pct = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, breakpoints)[0] / len(actual)

    # avoid log(0) / div-by-zero for empty buckets
    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-6, actual_pct)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))

# training-time "income" distribution vs. last week's production "income"
train_income = np.random.lognormal(mean=10.5, sigma=0.4, size=10_000)
prod_income = np.random.lognormal(mean=10.9, sigma=0.4, size=2_000)  # shifted up

score = psi(train_income, prod_income)
print(f"PSI = {score:.3f}")
# PSI < 0.1  -> no significant drift
# 0.1 - 0.25 -> moderate drift, investigate
# > 0.25     -> significant drift, retrain likely needed
```

## Detecting prediction/label drift with KS test

```python
from scipy import stats

def prediction_drift(train_scores: np.ndarray, prod_scores: np.ndarray, alpha: float = 0.01):
    """Kolmogorov-Smirnov test: are two samples drawn from the same
    distribution? Used here on model output probabilities."""
    statistic, p_value = stats.ks_2samp(train_scores, prod_scores)
    drifted = p_value < alpha
    return drifted, statistic, p_value

drifted, stat, p = prediction_drift(
    train_scores=np.random.beta(2, 5, 5000),
    prod_scores=np.random.beta(3, 4, 1000),   # shifted distribution
)
print(f"drifted={drifted}  KS statistic={stat:.3f}  p={p:.4f}")
```

## Worked example: a monitoring job that decides whether to retrain

```python
def check_drift_and_decide(feature_logs: dict[str, np.ndarray],
                            training_reference: dict[str, np.ndarray],
                            psi_threshold: float = 0.25) -> dict:
    """Run per-feature PSI checks and produce a retraining recommendation."""
    results = {}
    for feature_name, prod_values in feature_logs.items():
        ref_values = training_reference[feature_name]
        score = psi(ref_values, prod_values)
        results[feature_name] = round(score, 4)

    drifted_features = [f for f, s in results.items() if s > psi_threshold]

    return {
        "psi_by_feature": results,
        "drifted_features": drifted_features,
        "recommend_retrain": len(drifted_features) > 0,
        "reason": (
            f"{len(drifted_features)} feature(s) exceeded PSI {psi_threshold}: {drifted_features}"
            if drifted_features else "no feature exceeded drift threshold"
        ),
    }

report = check_drift_and_decide(
    feature_logs={"income": prod_income, "age": np.random.normal(38, 10, 2000)},
    training_reference={"income": train_income, "age": np.random.normal(35, 9, 10_000)},
)
print(report)
# {'psi_by_feature': {'income': 0.412, 'age': 0.031},
#  'drifted_features': ['income'],
#  'recommend_retrain': True,
#  'reason': "1 feature(s) exceeded PSI 0.25: ['income']"}
```

Wired into the nightly job from Module 04, `recommend_retrain: True` is
what actually triggers the retraining pipeline — not a fixed calendar
schedule, but a statistical signal that the input distribution has moved
far enough to matter.

## Cheat sheet

| Signal | Metric | What it catches |
|---|---|---|
| Feature distribution shift | PSI per feature | Input data changed shape |
| Prediction distribution shift | KS test on output scores | Model behaving differently on new inputs |
| Label drift (when labels arrive late) | Accuracy/AUC on delayed ground truth | Model actually getting worse, not just inputs shifting |
| Operational health | Latency, error rate, request volume | Infra problems, not model problems |

## How It Actually Works

**PSI is a symmetrized measure of how much probability mass moved between
two histograms, weighted so that moves in low-density regions still
register.** For each bucket, `(actual_pct - expected_pct) * log(actual_pct
/ expected_pct)` is zero only when the two proportions are identical in
that bucket, and grows both when a bucket gains mass it didn't have before
*and* when a bucket loses mass it used to have — the log term specifically
penalizes buckets where the *ratio* changed a lot even if the absolute
percentage difference is small (e.g. a bucket going from 0.5% to 2% of the
population is a 4x relative change that a raw percentage-point difference
would understate). Summing across all buckets gives one number that is
zero for identical distributions and grows without bound as they diverge —
the 0.1/0.25 thresholds are empirical conventions from credit-risk
modeling, not derived constants, which is why the worked example treats
them as a starting point to tune per feature, not a law of nature.

**The KS test on prediction scores catches a different failure mode than
PSI on features.** Feature-level PSI can look calm even while the model's
outputs shift, if the drift is in a feature's *relationship* to the label
rather than its marginal distribution — e.g. `income` might be distributed
identically in training and production, but if the economy changed such
that high income no longer predicts the same outcome it used to, PSI on
`income` alone won't show it. Running KS directly on the model's output
probabilities catches this because it measures whether the *function*
output changed distribution, which reflects both input drift and this kind
of relationship drift (concept drift) that feature-only monitoring misses.
This is why production monitoring stacks run both: feature PSI localizes
*which input* changed (useful for debugging), while prediction-score KS
catches drift regardless of *why* it happened.

**Automated retraining triggered by drift, not a calendar, closes the loop
correctly because the two failure modes have different timescales.**
A weekly cron retrain works fine when drift is slow and predictable (e.g.
gradual seasonal change) but wastes compute when nothing has changed, and
reacts too slowly when drift is sudden (e.g. a marketing campaign
shifts customer income distribution overnight). Wiring `recommend_retrain`
from a nightly drift check into the CI/CD pipeline's schedule-trigger path
(Module 04) means retraining happens exactly when the statistical evidence
justifies it — fast reaction to sudden drift, no wasted retraining during
stable periods — while the CI/CD promotion gate still guards against a
freshly retrained model actually being worse, so drift detection triggers
retraining but never bypasses quality control.

## Exercise

Simulate a scenario where `age` drifts gradually over 12 "weeks" (shift the
mean by a small amount each week) while `income` stays fixed. Run
`check_drift_and_decide` once per simulated week and plot PSI for `age`
over time. At which week does PSI cross 0.1, and at which week does it
cross 0.25? Then argue, from the shape of that curve, why a drift-triggered
retrain would fire earlier than a monitoring setup that only alerted on
raw accuracy drop (which requires waiting for delayed ground-truth labels
to arrive).
