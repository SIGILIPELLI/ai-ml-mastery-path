# 04 · CI/CD for Machine Learning

Regular software CI/CD checks code. ML systems fail in ways code-only
pipelines never catch: a schema change silently drops a feature, a
retraining job produces a model that's technically valid but worse than
production, or a dependency bump changes a metric's numeric behavior. This
module covers testing data and models (not just code), and the automated
retraining + safe-deployment pipeline that ties it together.

## Three kinds of tests an ML pipeline needs

```python
# tests/test_data.py — data validation
import pandas as pd
import pytest

def test_no_nulls_in_required_columns(training_df):
    required = ["age", "income", "label"]
    nulls = training_df[required].isnull().sum()
    assert nulls.sum() == 0, f"Unexpected nulls:\n{nulls[nulls > 0]}"

def test_label_distribution_within_range(training_df):
    positive_rate = training_df["label"].mean()
    # a sudden shift in class balance usually means an upstream data bug,
    # not a real change in the world
    assert 0.05 < positive_rate < 0.40, f"positive_rate={positive_rate:.3f} out of expected range"

def test_feature_ranges(training_df):
    assert training_df["age"].between(0, 120).all()
    assert training_df["income"].ge(0).all()
```

```python
# tests/test_model.py — model behavior, not just code correctness
import numpy as np

def test_model_beats_naive_baseline(trained_model, X_test, y_test):
    baseline_acc = max(y_test.mean(), 1 - y_test.mean())  # predict majority class
    model_acc = trained_model.score(X_test, y_test)
    assert model_acc > baseline_acc + 0.05, (
        f"model_acc={model_acc:.3f} barely beats baseline={baseline_acc:.3f}"
    )

def test_invariance_to_irrelevant_feature(trained_model, X_test):
    # changing a customer ID shouldn't change the prediction
    X_perturbed = X_test.copy()
    X_perturbed["customer_id"] = X_perturbed["customer_id"] + 1_000_000
    preds_before = trained_model.predict(X_test)
    preds_after = trained_model.predict(X_perturbed)
    assert np.array_equal(preds_before, preds_after)

def test_no_regression_vs_production_model(trained_model, prod_model, X_test, y_test):
    new_acc = trained_model.score(X_test, y_test)
    prod_acc = prod_model.score(X_test, y_test)
    assert new_acc >= prod_acc - 0.01, (
        f"new model ({new_acc:.3f}) regresses vs production ({prod_acc:.3f})"
    )
```

Unit tests check that code runs; these tests check that the *data* looks
like data the model was designed for, and that the *model* behaves
sensibly — beats a trivial baseline, ignores features it shouldn't use,
and doesn't quietly regress against what's already serving traffic.

## A GitHub Actions pipeline: test, train, gate, deploy

```yaml
# .github/workflows/ml-pipeline.yml
name: ML CI/CD
on:
  push:
    branches: [main]
  schedule:
    - cron: "0 3 * * *"   # nightly retrain on fresh data

jobs:
  validate-and-train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt

      - name: Data validation tests
        run: pytest tests/test_data.py -v

      - name: Train candidate model
        run: python train.py --output models/candidate.pkl

      - name: Model quality gate
        run: pytest tests/test_model.py -v
        env:
          CANDIDATE_MODEL_PATH: models/candidate.pkl
          PROD_MODEL_PATH: models/production.pkl

      - name: Upload candidate artifact
        if: success()
        uses: actions/upload-artifact@v4
        with:
          name: candidate-model
          path: models/candidate.pkl

  deploy-canary:
    needs: validate-and-train
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
        with: { name: candidate-model, path: models/ }
      - name: Push to canary (5% traffic)
        run: |
          ./deploy.sh --model models/candidate.pkl --target canary --traffic-pct 5
```

The pipeline never deploys straight to 100% of traffic. It gates on data
quality first (garbage in, garbage out — no point training on bad data),
then gates on model quality against the *current production model*, not
just an absolute threshold, and only then rolls out gradually.

## Worked example: a promotion gate function

```python
def should_promote(candidate_metrics: dict, prod_metrics: dict,
                    min_improvement: float = -0.005) -> tuple[bool, str]:
    """Decide whether a candidate model should replace production.

    min_improvement is allowed to be slightly negative: a model that is
    marginally worse on one held-out metric but simpler/cheaper/fairer
    might still be a reasonable ship, but a real regression should block.
    """
    checks = []
    delta_acc = candidate_metrics["accuracy"] - prod_metrics["accuracy"]
    checks.append(("accuracy", delta_acc >= min_improvement, delta_acc))

    # a latency regression is a hard fail regardless of accuracy gains
    delta_latency = candidate_metrics["p99_latency_ms"] - prod_metrics["p99_latency_ms"]
    checks.append(("p99_latency_ms", delta_latency <= 10, delta_latency))

    # never regress fairness metrics even slightly
    delta_fairness = candidate_metrics["demographic_parity_diff"] - prod_metrics["demographic_parity_diff"]
    checks.append(("fairness_gap", delta_fairness <= 0.0, delta_fairness))

    failed = [(name, delta) for name, ok, delta in checks if not ok]
    if failed:
        reasons = ", ".join(f"{name} delta={delta:.4f}" for name, delta in failed)
        return False, f"blocked: {reasons}"
    return True, "all gates passed"

# should_promote(
#     {"accuracy": 0.842, "p99_latency_ms": 45, "demographic_parity_diff": 0.03},
#     {"accuracy": 0.839, "p99_latency_ms": 42, "demographic_parity_diff": 0.03},
# )
# -> (True, "all gates passed")
```

## Cheat sheet

| Concern | Test type | Where it runs |
|---|---|---|
| Schema/null/range violations | Data validation tests | Before training |
| Model beats trivial baseline | Model behavior tests | After training |
| Model doesn't regress vs prod | Comparison gate | Before deploy |
| Gradual rollout | Canary deploy (5% → 25% → 100%) | After gate passes |
| Fresh data drift | Scheduled nightly retrain | Cron trigger |

## How It Actually Works

**The comparison gate exists because absolute thresholds silently drift out
of relevance.** A rule like "accuracy must exceed 0.80" made sense the day
it was written, but if the underlying data distribution shifts favorably
over six months, 0.80 might become trivially easy to clear with a
degraded model, or if it shifts unfavorably, an genuinely improved model
might never clear a threshold set for easier times. Comparing the
candidate directly against whatever is *currently serving production
traffic* (`prod_metrics`, computed on the *same* held-out test set as the
candidate) makes the gate self-relative: it always asks "is this strictly
better than what users are getting right now," which remains meaningful
regardless of how the absolute difficulty of the task has drifted. This is
exactly why `test_no_regression_vs_production_model` loads `prod_model` as
a fixture rather than hardcoding a number.

**Canary deployment limits the blast radius of a promotion gate that
passed but was still wrong.** Model quality tests run on a held-out test
set that is, by construction, a finite, historical sample — it cannot
capture every real-world input distribution, especially adversarial or
rare inputs. Routing only 5% of live traffic to the candidate first means
that if the offline tests missed a real regression (say, a rare category
combination that wasn't represented in the test set), the damage is
contained to a twentieth of users while online metrics (error rate,
business KPIs, latency) are monitored, and rollback is a traffic-routing
change, not a data-loss incident. The percentage ramps up (5% → 25% → 100%)
only as the canary's *live* metrics keep looking healthy — this is a
second, independent gate on top of the offline one, and it's the reason
production ML pipelines almost never do an instant full cutover even after
passing every offline test.

**The nightly retrain cron and the push-triggered pipeline are the same
job for two different reasons.** A code change (a push) needs immediate
validation because a training script bug should be caught before merge.
A schedule-triggered run with no code change re-runs the *same* pipeline
against whatever data now exists in the training set — which is how drift
gets caught automatically: if yesterday's data pushed the candidate's
`positive_rate` test near its boundary, or the accuracy gate started
failing against production despite no code change, that's a real signal
about the world changing under the model, not a code regression, and it
surfaces on the same dashboard/gate a code push would have used.

## Exercise

Extend `should_promote` with a fourth gate: reject the candidate if its
model file size exceeds 1.5x the production model's size (a proxy for
"got much more expensive to serve without a compensating accuracy gain").
Then write a `test_data.py` case that would have caught a real incident:
a training set where a categorical column's cardinality jumped from 12 to
50,000 distinct values overnight (e.g. because an upstream ID column was
accidentally joined in as a "category"). What specific assertion catches
this before it reaches `train.py`?
