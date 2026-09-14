# 07 · Model Interpretability

A model that predicts well but can't explain *why* is a liability in
regulated domains (credit, healthcare, hiring) and a debugging nightmare
everywhere else. This module covers three complementary tools: permutation
importance, partial dependence, and SHAP values.

## Setup

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print(f"R^2: {rf.score(X_test, y_test):.3f}")   # ~0.805
```

## Permutation importance

`feature_importances_` (Module 01/02) measures how often a feature was used
for splitting — but that can be biased toward high-cardinality features.
**Permutation importance** instead measures the actual performance drop
when a feature's values are shuffled, breaking its relationship to the
target.

```python
from sklearn.inspection import permutation_importance
import numpy as np

result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
order = np.argsort(result.importances_mean)[::-1]
for i in order[:5]:
    print(f"{X.columns[i]:12s} {result.importances_mean[i]:.3f} +/- {result.importances_std[i]:.3f}")
# MedInc       0.891 +/- 0.021
# AveOccup     0.079 +/- 0.006
# Latitude     0.075 +/- 0.005
# Longitude    0.068 +/- 0.006
# HouseAge     0.021 +/- 0.002
```

The number reported is the drop in R² caused by shuffling that one column —
directly interpretable, model-agnostic (works for any fitted estimator), and
computed on held-out `X_test`, unlike `feature_importances_`.

## Partial dependence: what happens as one feature changes

Importance says *how much* a feature matters; **partial dependence** shows
*how* — the shape of the relationship.

```python
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

PartialDependenceDisplay.from_estimator(rf, X_train, features=["MedInc", "HouseAge"])
plt.savefig("pdp.png")
```

For `MedInc` (median income), the curve rises steeply then flattens —
predicted house value increases with income, but with diminishing returns
past a point, a pattern invisible from the importance number alone.

## SHAP values: per-prediction attribution

Importance and PDP describe the model *globally*. **SHAP** (SHapley
Additive exPlanations) explains one specific prediction: how much each
feature pushed *this* prediction above or below the average.

```python
import shap

explainer = shap.TreeExplainer(rf)
sample = X_test.iloc[[0]]
shap_values = explainer(sample)

print("base value (avg prediction):", shap_values.base_values[0])
print("this prediction:", rf.predict(sample)[0])
for name, val in zip(X.columns, shap_values.values[0]):
    print(f"{name:12s} {val:+.3f}")
# base value (avg prediction): 2.07
# this prediction: 3.41
# MedInc       +1.02
# AveOccup     +0.18
# Latitude     +0.09
# ...
```

The SHAP values for one row sum (plus the base value) to exactly that row's
prediction: `base_value + Σ shap_values = prediction` — every unit of
"why is this prediction what it is" is accounted for.

## Cheat sheet

| Question | Tool |
|---|---|
| Which features matter overall? | `permutation_importance` |
| What shape is the relationship? | `PartialDependenceDisplay` |
| Why did the model predict *this* for *this row*? | `shap.TreeExplainer` |
| Fast but potentially biased importance | `model.feature_importances_` |

## How It Actually Works

**Permutation importance measures a causal-ish counterfactual, not a
correlation.** For feature `j`, the procedure takes the test set, randomly
shuffles only column `j` across rows (so each row now has a *wrong*, random
value for feature `j` but correct values for everything else), computes the
model's score on this corrupted data, and reports the drop from the
original score, averaged over `n_repeats=10` independent shuffles (the
`+/- std` reflects shuffle-to-shuffle variance). Because shuffling
specifically destroys the statistical relationship between column `j` and
the target while leaving every other column's relationship intact, the
resulting score drop isolates the *marginal, model-relied-upon* value of
that one feature — mechanically different from `feature_importances_`,
which just counts how often and how effectively a feature reduced impurity
during tree-building (Module 05's Gini mechanism) and can overweight
high-cardinality features simply because they offer more possible split
thresholds to try.

**Partial dependence averages out every other feature to isolate one.** For
a target feature (say `MedInc`) and a grid of values `v_1, ..., v_k`
spanning its observed range, the partial dependence at `v_i` is computed by
taking *every* row in the dataset, overwriting its `MedInc` value with
`v_i` (keeping all its other features as they actually are), running the
model on this modified dataset, and averaging the predictions. Repeating
this for each grid value and plotting the resulting curve shows how the
model's average prediction moves as `MedInc` moves, with the influence of
every other feature averaged away by construction — which is precisely why
it can reveal a shape (steep-then-flat) that a single importance number
cannot.

**SHAP values are Shapley values from cooperative game theory, applied to
features as "players" and the prediction as the "payout."** Conceptually,
for a given row, SHAP considers every possible subset (coalition) of
features, measures how much adding feature `j` to a subset changes the
model's output (averaged over subsets and orderings, weighted so that all
orderings of adding features are treated fairly), and assigns that averaged
marginal contribution to feature `j` as its SHAP value. `TreeExplainer`
computes this exactly and efficiently for tree ensembles by walking each
tree's structure rather than literally enumerating all `2^30` feature
subsets. The resulting values have one mathematically guaranteed property
that makes them trustworthy for individual explanations: **efficiency** —
they sum exactly to `prediction - base_value`, so there's no leftover
"unexplained" contribution, which is what makes `base_value + Σ shap_values
= prediction` an equality rather than an approximation.

## Exercise

Compute permutation importance for `rf` on the *training* set instead of
the test set. Compare the ranking and magnitudes against the test-set
version above, and explain — using the "shuffling destroys the true
relationship" mechanism — why training-set permutation importance tends to
overstate the importance of features the model has overfit to.
