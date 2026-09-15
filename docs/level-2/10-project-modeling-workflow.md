---
description: "Project — End-to-End Modeling Workflow — This capstone ties Level 2 together into one realistic workflow: engineer features, train and tune a boosted-tree…"
---

# 10 · Project — End-to-End Modeling Workflow

This capstone ties Level 2 together into one realistic workflow: engineer
features, train and tune a boosted-tree model, interpret it, and evaluate
it honestly on an imbalanced classification problem — predicting customer
churn.

## The dataset and problem framing

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
n = 4000
df = pd.DataFrame({
    "tenure_months": rng.integers(1, 72, n),
    "monthly_charge": rng.normal(65, 20, n).clip(10, 150),
    "num_support_tickets": rng.poisson(1.2, n),
    "contract_type": rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20]),
    "has_addon": rng.choice([0, 1], n, p=[0.6, 0.4]),
})
# Churn probability driven by tenure, contract type, and support tickets
logit = (-2.0 - 0.04 * df.tenure_months + 0.35 * df.num_support_tickets
         + (df.contract_type == "month-to-month") * 1.1 - 0.3 * df.has_addon)
prob_churn = 1 / (1 + np.exp(-logit))
df["churned"] = (rng.random(n) < prob_churn).astype(int)
print(df["churned"].value_counts(normalize=True))   # ~0.80 / 0.20 -- imbalanced, realistic
```

## Feature engineering pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

X = df.drop(columns="churned")
y = df["churned"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

numeric_features = ["tenure_months", "monthly_charge", "num_support_tickets"]
categorical_features = ["contract_type"]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(drop="first"), categorical_features),
], remainder="passthrough")   # passes has_addon through unchanged
```

## Model + hyperparameter search

```python
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

pipe = Pipeline([
    ("prep", preprocess),
    ("clf", XGBClassifier(eval_metric="logloss", random_state=42)),
])

param_dist = {
    "clf__n_estimators": [100, 200, 400],
    "clf__max_depth": [3, 4, 6],
    "clf__learning_rate": [0.01, 0.05, 0.1],
    "clf__scale_pos_weight": [1, 3, 4],   # counteracts the 80/20 imbalance
}

search = RandomizedSearchCV(
    pipe, param_dist, n_iter=15, scoring="average_precision",
    cv=5, random_state=42, n_jobs=-1,
)
search.fit(X_train, y_train)
print("best params:", search.best_params_)
print(f"best CV average precision: {search.best_score_:.3f}")
```

## Evaluation

```python
from sklearn.metrics import classification_report, average_precision_score

best_model = search.best_estimator_
proba = best_model.predict_proba(X_test)[:, 1]
pred = best_model.predict(X_test)

print(classification_report(y_test, pred, target_names=["stayed", "churned"]))
print(f"test average precision: {average_precision_score(y_test, proba):.3f}")
```

## Interpretation

```python
import shap

xgb_model = best_model.named_steps["clf"]
feature_names = (numeric_features
                  + list(best_model.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(categorical_features))
                  + ["has_addon"])
X_test_transformed = best_model.named_steps["prep"].transform(X_test)

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(X_test_transformed)
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
for name, val in sorted(zip(feature_names, mean_abs_shap), key=lambda p: -p[1])[:5]:
    print(f"{name:30s} {val:.3f}")
# tenure_months                  0.612
# contract_type_month-to-month   0.398
# num_support_tickets            0.221
# ...
```

## Cheat sheet: the full workflow

| Step | Module this reuses |
|---|---|
| Realistic imbalanced target | 08 · Imbalanced Data |
| `ColumnTransformer` for mixed feature types | Level 1 · Feature Engineering |
| Boosted trees | 02 · Gradient Boosting & XGBoost |
| `RandomizedSearchCV` scored on `average_precision` | 08 · Imbalanced Data |
| SHAP explanation of the final model | 07 · Model Interpretability |

## How It Actually Works

**`scale_pos_weight` is XGBoost's version of `class_weight="balanced"`,
applied to gradient boosting's residual mechanism.** Recall from Module 02
that gradient boosting fits each tree to the negative gradient of the loss
(roughly, `y - p` for log loss). `scale_pos_weight` multiplies the gradient
and Hessian contributions of positive-class (churned) examples by the given
factor before each tree is fit, which is mechanically equivalent to telling
the boosting procedure "a mistake on a churned customer costs `w` times as
much as a mistake on a retained one." With a true class ratio near 4:1, a
`scale_pos_weight` around 3–4 roughly rebalances the effective gradient
contribution of each class, pushing the sequence of fitted trees to pay
proportionally more attention to correctly separating the minority class
rather than optimizing overall accuracy, which the 80% majority class would
otherwise dominate.

**Scoring the hyperparameter search on `average_precision` rather than
accuracy changes which configuration wins, not just how it's reported.**
`RandomizedSearchCV`'s `scoring=` parameter determines the number that
`cv=5`'s cross-validation folds actually optimize for when ranking the 15
sampled parameter combinations — it isn't a cosmetic label applied after
the fact. A configuration with high accuracy but poor recall on the
minority class (predicting "stayed" for almost everyone, as Module 05's
dummy classifier did on imbalanced data) would score well on accuracy but
poorly on average precision, because average precision (Module 08's
PR-curve area) is computed entirely from how well predicted probabilities
rank actual positives above negatives — a metric structurally insensitive
to the majority class's sheer size. This is why `scoring="average_
precision"` is not an evaluation-time footnote here: it is the literal
objective the search selects `best_params_` against.

**Pipeline transformers are refit inside every cross-validation fold,
preventing leakage the same way `StandardScaler` did in Level 1.**
`ColumnTransformer`'s `OneHotEncoder` and `StandardScaler` are wrapped
inside the `Pipeline`, which `RandomizedSearchCV` clones and refits from
scratch on each of the 5 training folds during search, and once more on the
full training set for the final `best_estimator_`. This guarantees the
one-hot categories and the scaler's mean/std are always learned only from
that fold's training rows — mirroring the "fit on train, transform on test"
discipline from Modules 04 and elsewhere, just automated across every fold
of the search instead of manually enforced once.

## Exercise

Refit `best_estimator_`'s underlying `XGBClassifier` with `scale_pos_weight`
fixed at `1` (no imbalance correction) while keeping every other
hyperparameter from `search.best_params_`. Compare the resulting
`classification_report` and average precision against the tuned version.
Identify specifically which metric (precision or recall on the "churned"
class) degrades most, and explain why using the gradient-reweighting
mechanism above.
