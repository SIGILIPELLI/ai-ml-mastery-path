# 10 · Capstone — Operate a Production ML Service

This capstone ties Modules 01-09 together into one small but complete
service: a fraud-scoring API that is trained, tested, served, monitored
for drift, gated by CI/CD, and evaluated with an A/B test before a full
rollout — the same shape as a real production ML system, just small enough
to build end-to-end yourself.

## The service: a fraud-scoring API

```python
# train.py — reproducible training with pinned deps (Module 03/05 pattern)
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd

def train(data_path: str, output_path: str) -> dict:
    df = pd.read_parquet(data_path)
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    joblib.dump(model, output_path)
    return {"auc": auc, "n_train": len(X_train), "n_test": len(X_test)}
```

```python
# serve.py — FastAPI serving with logging for drift detection (Module 02/05)
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from datetime import datetime, timezone
import json

app = FastAPI()
model = joblib.load("model.pkl")
MODEL_VERSION = "v7"

class ScoreRequest(BaseModel):
    amount: float
    merchant_category_risk: float
    hours_since_last_transaction: float
    is_new_device: bool

class ScoreResponse(BaseModel):
    fraud_probability: float
    flagged: bool
    model_version: str

@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    x = np.array([[req.amount, req.merchant_category_risk,
                   req.hours_since_last_transaction, float(req.is_new_device)]])
    proba = float(model.predict_proba(x)[0][1])
    response = ScoreResponse(fraud_probability=proba, flagged=proba > 0.5, model_version=MODEL_VERSION)

    with open("prediction_log.jsonl", "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_version": MODEL_VERSION,
            "features": req.model_dump(),
            "fraud_probability": proba,
        }) + "\n")

    return response
```

## Testing before it ships (Module 04)

```python
# tests/test_service.py
def test_no_regression_vs_current_production(candidate_model, prod_model, X_test, y_test):
    from sklearn.metrics import roc_auc_score
    candidate_auc = roc_auc_score(y_test, candidate_model.predict_proba(X_test)[:, 1])
    prod_auc = roc_auc_score(y_test, prod_model.predict_proba(X_test)[:, 1])
    assert candidate_auc >= prod_auc - 0.005, f"candidate {candidate_auc:.4f} regresses vs prod {prod_auc:.4f}"

def test_amount_monotonicity(candidate_model):
    # a sanity/invariance check: all else equal, a much larger transaction
    # amount should not produce a LOWER fraud score
    base = np.array([[50.0, 0.3, 12.0, 0.0]])
    large = np.array([[5000.0, 0.3, 12.0, 0.0]])
    assert candidate_model.predict_proba(large)[0][1] >= candidate_model.predict_proba(base)[0][1]

def test_fairness_gap_within_bound(candidate_model, X_test, y_test, customer_region):
    report = fairness_report(y_test.values, candidate_model.predict(X_test), customer_region)
    assert demographic_parity_difference(report) <= 0.10
```

## CI/CD pipeline wiring it together (Module 04)

```yaml
# .github/workflows/fraud-service.yml
name: Fraud Service CI/CD
on:
  push: { branches: [main] }
  schedule: [{ cron: "0 4 * * *" }]

jobs:
  train-test-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - name: Data validation
        run: pytest tests/test_data.py -v
      - name: Train candidate
        run: python train.py --output models/candidate.pkl
      - name: Quality + fairness gate
        run: pytest tests/test_service.py -v
      - name: Register candidate
        run: python platform/register_model.py --name fraud-detector --path models/candidate.pkl

  canary-rollout:
    needs: train-test-gate
    runs-on: ubuntu-latest
    steps:
      - run: ./deploy.sh --model fraud-detector --stage staging --traffic-pct 5
```

## Monitoring drift once it's live (Module 05)

```python
# monitor_job.py — scheduled nightly (matches the CI/CD cron)
def nightly_drift_check():
    prod_logs = load_recent_predictions("prediction_log.jsonl", days=1)
    training_reference = load_training_reference("data/train_reference.parquet")

    report = check_drift_and_decide(
        feature_logs={col: prod_logs[col].values for col in ["amount", "merchant_category_risk"]},
        training_reference={col: training_reference[col].values for col in ["amount", "merchant_category_risk"]},
    )
    if report["recommend_retrain"]:
        trigger_retraining_pipeline(reason=report["reason"])
    return report
```

## A/B testing the new model before full cutover (Module 06)

```python
def run_fraud_model_experiment(candidate_version: str, prod_version: str,
                                daily_transactions: int = 8000):
    n_needed = required_sample_size(baseline_rate=0.012, min_detectable_effect=0.002)
    days_needed = math.ceil(n_needed / (daily_transactions * 0.5))
    print(f"Need {n_needed} transactions/variant -> ~{days_needed} days at current traffic")

    # after the experiment runs for days_needed days:
    result = analyze_ab_test(
        control_conversions=142, control_total=32_000,        # "conversion" = correctly caught fraud
        treatment_conversions=168, treatment_total=32_010,
    )
    return result
```

## Worked example: the full promotion decision, end to end

```python
def full_promotion_decision(candidate_metrics, prod_metrics, fairness_reports,
                             ab_test_result, drift_report) -> tuple[bool, list[str]]:
    """Every gate from every module, combined into one ship/no-ship call."""
    blockers = []

    promote_ok, reason = should_promote(candidate_metrics, prod_metrics)  # Module 04
    if not promote_ok:
        blockers.append(f"CI/CD gate: {reason}")

    dp_gap = demographic_parity_difference(fairness_reports["candidate"])  # Module 07
    if dp_gap > 0.10:
        blockers.append(f"fairness gate: demographic parity gap {dp_gap:.3f} > 0.10")

    if not ab_test_result["significant"] or ab_test_result["absolute_lift"] <= 0:  # Module 06
        blockers.append(f"A/B test: no significant positive lift (p={ab_test_result['p_value']})")

    if drift_report["recommend_retrain"]:  # Module 05 -- stale reference data invalidates the comparison
        blockers.append(f"drift gate: {drift_report['reason']} -- re-run comparison on fresh reference data")

    return len(blockers) == 0, blockers

ship, blockers = full_promotion_decision(
    candidate_metrics={"accuracy": 0.958, "p99_latency_ms": 38, "demographic_parity_diff": 0.04},
    prod_metrics={"accuracy": 0.951, "p99_latency_ms": 40, "demographic_parity_diff": 0.04},
    fairness_reports={"candidate": fairness_report(y_true, y_pred_candidate, region)},
    ab_test_result={"significant": True, "absolute_lift": 0.008, "p_value": 0.021},
    drift_report={"recommend_retrain": False, "reason": "no feature exceeded drift threshold"},
)
print("SHIP" if ship else f"BLOCKED: {blockers}")
```

## Cheat sheet: the full lifecycle

| Stage | Module | Artifact in this capstone |
|---|---|---|
| Design the system, avoid train/serve skew | 01 | Shared feature computation in `train.py`/`serve.py` |
| Serve with a validated API | 02 | `serve.py` FastAPI endpoint |
| Feature consistency at scale | 03 | (would move to a feature store beyond this size) |
| Test data + model, gate promotion | 04 | `tests/test_service.py`, `fraud-service.yml` |
| Detect drift, trigger retraining | 05 | `nightly_drift_check` |
| Validate the change with real traffic | 06 | `run_fraud_model_experiment` |
| Check fairness before shipping | 07 | `test_fairness_gap_within_bound` |
| Control serving cost | 08 | (quantize `model.pkl` if latency/cost demands it) |
| Multi-team infrastructure | 09 | `register_model.py` against a shared registry |
| **All of the above, one decision** | — | `full_promotion_decision` |

## How It Actually Works

**Every module in this level turned out to be one gate in a single
pipeline, and the capstone's real lesson is that none of those gates is
sufficient alone.** A model can pass the CI/CD accuracy gate (Module 04)
while failing fairness (Module 07); it can pass both while the A/B test
shows no real user-facing improvement (Module 06); it can pass all three
while drift monitoring (Module 05) reveals the comparison itself was run
against stale reference data and needs to be redone. `full_promotion_decision`
makes this explicit by checking all four independently and refusing to
ship if *any* one fails — this mirrors why real ML platforms accumulate
gates over time rather than picking one "good enough" metric: each new
production incident (a fairness complaint, a drift-driven silent
regression, a canary that looked fine offline but hurt a real experiment)
tends to be exactly the failure mode the *next* gate was added to catch.

**The order of the pipeline matters as much as its contents, and it
follows the same "cheap checks first, expensive commitments last"
principle from Module 09.** Data validation runs before training (no point
training on bad data); the CI/CD quality+fairness gate runs before any
deployment (no point serving a model that will get rolled back); the
canary rollout happens before a full A/B test at scale (limit exposure
before investing in a multi-week experiment); and the A/B test's own
result only counts if drift monitoring confirms the comparison data is
still valid (no point trusting an experiment run against a reference
distribution that's since shifted). Reordering any of these — say, running
the expensive A/B test before the cheap fairness check — would waste weeks
of traffic on a candidate that a five-line assertion could have rejected
in seconds.

**The prediction log is the single artifact that makes every later stage
possible, which is why it's written at serve time rather than reconstructed
after the fact.** `serve.py` logs every request's features and prediction
the moment it's made — this is the only way `nightly_drift_check` can later
compare *actual production inputs* against the training reference (Module
05 needs real logged data, not a synthetic proxy), and it's the same log
format an A/B test's variant-level outcome tracking (Module 06) and a
platform-wide audit trail (Module 09) would both build on. A service that
skips prediction logging "to keep serve.py simple" quietly forecloses drift
detection, experiment analysis, and incident debugging months later, once
the gap in historical data can no longer be filled in retroactively.

## Exercise

Wire `full_promotion_decision` into the `canary-rollout` GitHub Actions job
as an actual gate (pseudocode is fine): the job should call it with real
metrics pulled from the training run, the fairness report, the most recent
completed A/B test, and the latest drift check, and fail the workflow step
if `ship` is `False`, printing every blocker. Then describe, in a few
sentences, which of the four gates you would keep as a hard blocker versus
which you'd make a warning that still allows a human to override and
ship anyway — and justify the difference using the trade-offs discussed
across Modules 04-07.
