---
description: "Responsible AI — Fairness, Privacy, Security — A model that's accurate on average can still systematically fail one subgroup, leak information about…"
---

# 07 · Responsible AI — Fairness, Privacy, Security

A model that's accurate on average can still systematically fail one
subgroup, leak information about individuals in its training data, or be
tricked by a crafted input. This module covers measuring fairness with
concrete metrics, the mechanics of differential privacy, and adversarial
robustness — not as ethics theory, but as things you compute and test.

## Measuring fairness across groups

```python
import pandas as pd
import numpy as np

def fairness_report(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": group})
    rows = []
    for g, sub in df.groupby("group"):
        tp = ((sub.y_true == 1) & (sub.y_pred == 1)).sum()
        fp = ((sub.y_true == 0) & (sub.y_pred == 1)).sum()
        fn = ((sub.y_true == 1) & (sub.y_pred == 0)).sum()
        tn = ((sub.y_true == 0) & (sub.y_pred == 0)).sum()
        rows.append({
            "group": g,
            "n": len(sub),
            "selection_rate": sub.y_pred.mean(),          # for demographic parity
            "true_positive_rate": tp / (tp + fn) if (tp + fn) else np.nan,   # for equal opportunity
            "false_positive_rate": fp / (fp + tn) if (fp + tn) else np.nan,
        })
    return pd.DataFrame(rows)

def demographic_parity_difference(report: pd.DataFrame) -> float:
    return report["selection_rate"].max() - report["selection_rate"].min()

def equal_opportunity_difference(report: pd.DataFrame) -> float:
    return report["true_positive_rate"].max() - report["true_positive_rate"].min()

# a loan-approval model's outputs, split by an applicant attribute
report = fairness_report(y_true, y_pred, group=applicant_group)
print(report)
print(f"Demographic parity gap: {demographic_parity_difference(report):.3f}")
print(f"Equal opportunity gap:  {equal_opportunity_difference(report):.3f}")
```

Demographic parity (equal *selection rates*) and equal opportunity (equal
*true positive rates* among those who should be approved) are different,
sometimes mutually exclusive, definitions of fairness — a model can
satisfy one and violate the other, which is why a fairness report always
computes both rather than picking one number to optimize.

## Differential privacy: adding calibrated noise

```python
import numpy as np

def laplace_mechanism(true_value: float, sensitivity: float, epsilon: float) -> float:
    """Add Laplace noise scaled to sensitivity/epsilon to make a query
    differentially private. Smaller epsilon = more privacy, more noise."""
    scale = sensitivity / epsilon
    return true_value + np.random.laplace(loc=0.0, scale=scale)

def private_mean_income(incomes: np.ndarray, epsilon: float, income_cap: float = 500_000) -> float:
    # clip first: an unbounded value has unbounded sensitivity, breaking the
    # privacy guarantee entirely -- DP requires a known worst-case sensitivity
    clipped = np.clip(incomes, 0, income_cap)
    true_mean = clipped.mean()
    sensitivity = income_cap / len(incomes)   # one person's max possible influence on the mean
    return laplace_mechanism(true_mean, sensitivity, epsilon)

incomes = np.random.lognormal(mean=10.8, sigma=0.5, size=5000)
for eps in [0.1, 1.0, 10.0]:
    noisy = private_mean_income(incomes, epsilon=eps)
    print(f"epsilon={eps:5.1f}  true_mean={incomes.mean():.0f}  private_mean={noisy:.0f}")
# epsilon=  0.1  true_mean=53819  private_mean=54912   (loose privacy budget -> lots of noise)
# epsilon=  1.0  true_mean=53819  private_mean=53780
# epsilon= 10.0  true_mean=53819  private_mean=53822   (tight budget -> barely any noise)
```

## Adversarial robustness: a simple evasion attack

```python
import torch
import torch.nn.functional as F

def fgsm_attack(model, x: torch.Tensor, y_true: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Fast Gradient Sign Method: perturb input in the direction that
    increases the loss the fastest, bounded by epsilon."""
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    loss = F.cross_entropy(logits, y_true)
    loss.backward()

    perturbation = epsilon * x.grad.sign()
    x_adversarial = (x + perturbation).clamp(0, 1)   # keep valid pixel/feature range
    return x_adversarial.detach()

# measuring how much accuracy drops under attack
def robustness_eval(model, X_test, y_test, epsilons=(0.0, 0.01, 0.05, 0.1)):
    for eps in epsilons:
        if eps == 0.0:
            X_eval = X_test
        else:
            X_eval = fgsm_attack(model, X_test, y_test, eps)
        preds = model(X_eval).argmax(dim=1)
        acc = (preds == y_test).float().mean().item()
        print(f"epsilon={eps:.3f}  accuracy={acc:.3f}")
# epsilon=0.000  accuracy=0.968
# epsilon=0.010  accuracy=0.891
# epsilon=0.050  accuracy=0.412
# epsilon=0.100  accuracy=0.083
```

## Worked example: comparing two models on fairness before deployment

```python
def compare_models_fairness(reports: dict[str, pd.DataFrame], max_gap: float = 0.10) -> str:
    """Given fairness reports for candidate vs. production model, decide
    whether the candidate is deployable on fairness grounds alone."""
    for name, report in reports.items():
        dp_gap = demographic_parity_difference(report)
        eo_gap = equal_opportunity_difference(report)
        status = "PASS" if max(dp_gap, eo_gap) <= max_gap else "FAIL"
        print(f"{name}: demographic_parity_gap={dp_gap:.3f} "
              f"equal_opportunity_gap={eo_gap:.3f} -> {status}")

compare_models_fairness({
    "candidate": fairness_report(y_true, y_pred_candidate, applicant_group),
    "production": fairness_report(y_true, y_pred_prod, applicant_group),
})
```

This slots directly into the CI/CD promotion gate from Module 04: a
fairness regression (`compare_models_fairness` returning `FAIL` for the
candidate where production passed) should block promotion exactly like an
accuracy regression does.

## Cheat sheet

| Concern | Metric / technique |
|---|---|
| Equal approval rates across groups | Demographic parity difference |
| Equal accuracy for those who deserve approval | Equal opportunity difference |
| Bound what a query reveals about one individual | Differential privacy (Laplace mechanism) |
| Robustness to crafted malicious inputs | Adversarial evaluation (FGSM, PGD) |
| Gate deployment on fairness, not just accuracy | Fairness check in the CI/CD pipeline |

## How It Actually Works

**Demographic parity and equal opportunity can conflict because they
constrain different conditional distributions, and no model generally
satisfies both unless the base rates are already equal across groups.**
Demographic parity requires `P(prediction=1 | group=A) = P(prediction=1 |
group=B)` — the *same fraction* of each group gets approved, regardless of
each group's true qualification rate. Equal opportunity requires `P(pred=1
| group=A, label=1) = P(pred=1 | group=B, label=1)` — the same fraction of
*truly qualified* applicants get approved. If group A and group B have
different true positive rates in the underlying population (a real base
rate difference, not a model artifact), a perfectly calibrated model that
satisfies equal opportunity will *necessarily* have different overall
selection rates and thus violate demographic parity, and vice versa — this
is a mathematical impossibility result (proven formally for calibration +
equalized odds together), not an engineering failure, which is exactly why
the report computes both gaps and a human has to decide which trade-off
the deployment context calls for rather than treating either metric as
the single ground truth for "fair."

**The Laplace mechanism's privacy guarantee comes from making the output
distribution nearly identical whether or not any single individual's data
is included.** Differential privacy formally requires that for any two
datasets differing in exactly one record, the probability of any given
output changes by at most a factor of `e^epsilon`. Adding Laplace noise
scaled to `sensitivity/epsilon` — where sensitivity is the maximum amount
one record could possibly change the true answer — achieves exactly this:
because one person's income is capped at `income_cap`, their maximum
influence on the mean is bounded (`sensitivity = income_cap / n`), and the
Laplace distribution's specific shape (its density ratio at two points `x`
and `x+d` is exactly `e^(-|d|/scale)`) makes the *ratio* of output
probabilities with vs. without that person's true contribution bounded by
exactly `e^epsilon` — not approximately, but by the mechanism's
mathematical construction. This is why clipping to `income_cap` has to
happen *before* computing sensitivity: an unclipped value has unbounded
sensitivity (one outlier income of $50M could shift the mean arbitrarily),
which would require infinite noise to guarantee the same privacy bound.

**FGSM's attack works because it directly uses the same gradient that
training uses, just pointed the opposite direction and applied to the
input instead of the weights.** Training moves *weights* in the direction
that decreases loss (gradient descent). FGSM computes the gradient of the
loss with respect to the *input pixels* and moves the input in the
direction that *increases* loss, by a tiny bounded amount (`epsilon`) per
pixel — `x.grad.sign()` takes just the direction, not magnitude, of each
pixel's gradient, so every pixel moves by exactly `epsilon` in whichever
direction most increases the model's error, and clamped to a valid input
range so the result still looks like a legitimate input. Because neural
networks are locally near-linear in high-dimensional input space, a large
number of small, coordinated per-pixel perturbations (in gradient
direction) compounds into a large change in the loss even though no
individual pixel changed by more than `epsilon` — which is why accuracy in
the worked example collapses from 96.8% to 8.3% at `epsilon=0.1`, a
perturbation imperceptible to a human but catastrophic to the model,
because the attack is exploiting the exact same gradient signal the model
itself was trained with.

## 🔀 Related lessons on other tracks

- [AI Tools — 06 · Data Privacy & Security When Using AI Tools](https://sigilipelli.github.io/ai-tools-mastery-path/level-2/06-data-privacy-security/)

## Exercise

Using `fairness_report`, compute demographic parity and equal opportunity
gaps for a synthetic dataset where group A has a true positive rate of 40%
and group B has a true positive rate of 15% (simulate labels accordingly),
using a single shared classification threshold for both groups. Then try
two different thresholds per group, tuned so that both groups' true
positive rates match. Report what happens to demographic parity when you
do this, and explain — using the impossibility argument above — why you
could not have made *both* metrics converge to zero simultaneously.
