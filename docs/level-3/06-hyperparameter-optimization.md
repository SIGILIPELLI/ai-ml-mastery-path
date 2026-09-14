# 06 · Hyperparameter Optimization at Scale

`GridSearchCV` and `RandomizedSearchCV` (Level 1-2) work for a handful of
hyperparameters. Modern models can have dozens, and each trial can be
expensive. This module covers **Bayesian optimization** with Optuna,
which searches more intelligently than random or grid search, and
**pruning**, which kills bad trials early.

## Random search's blind spot

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier

data = load_breast_cancer(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.25, random_state=42, stratify=data.target
)
```

Random search samples hyperparameters independently of past results —
trial 50 learns nothing from trials 1-49, no matter how informative they
were.

## Bayesian optimization with Optuna

```python
import optuna

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    }
    model = XGBClassifier(**params, eval_metric="logloss", random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc").mean()
    return score

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50)

print("best params:", study.best_params)
print(f"best CV ROC AUC: {study.best_value:.4f}")
```

`suggest_float(..., log=True)` for `learning_rate` samples on a log scale —
appropriate because the meaningful difference between `0.01` and `0.02` is
much larger (in effect) than between `0.20` and `0.21`.

## Pruning: killing bad trials early

For models trained iteratively (like XGBoost's boosting rounds), Optuna can
monitor intermediate performance and abandon a trial the moment it's
clearly worse than the best trial so far — without finishing training.

```python
def objective_with_pruning(trial):
    params = {
        "n_estimators": 500,
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
    }
    model = XGBClassifier(
        **params, eval_metric="logloss", random_state=42,
        early_stopping_rounds=20,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return model.best_score

study2 = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
)
study2.optimize(objective_with_pruning, n_trials=30)
print("trials pruned:", sum(1 for t in study2.trials if t.state == optuna.trial.TrialState.PRUNED))
```

## Worked example: visualizing the search

```python
importances = optuna.importance.get_param_importances(study)
for name, imp in importances.items():
    print(f"{name:16s} {imp:.3f}")
# learning_rate    0.412
# max_depth        0.318
# n_estimators     0.201
# subsample        0.069
```

`get_param_importances` fits a lightweight model (a random forest, per
Level 2 Module 07's importance mechanism) predicting the objective from
`(trial's params -> trial's score)` pairs across all 50 trials — telling you
which hyperparameter actually drove the score differences, distinct from
which one you happened to search over the widest range.

## Cheat sheet

| Concept | Tool |
|---|---|
| Search space definition | `trial.suggest_int/float/categorical` |
| Search strategy | `TPESampler` (Bayesian) vs. `RandomSampler` |
| Early-kill bad trials | `optuna.pruners.MedianPruner` |
| Which params mattered | `optuna.importance.get_param_importances` |
| Log-scale search | `suggest_float(..., log=True)` for rate-like params |

## How It Actually Works

**TPE (Tree-structured Parzen Estimator) models "good" and "bad"
hyperparameter regions as two separate probability distributions, and picks
the next trial to maximize their ratio.** After some initial random trials,
TPE splits observed trials into the top fraction (say, best 20% of scores
so far) and the rest, then fits two probability density estimates over the
hyperparameter space — `l(x)` for the good group, `g(x)` for the rest.
It then proposes the next trial's hyperparameters by sampling candidates
and picking the one maximizing `l(x)/g(x)` — a point that the "good" model
considers likely but the "bad" model considers unlikely. This is
mechanically why TPE outperforms random search over many trials: each new
trial is chosen using the accumulated evidence of which regions have
historically scored well, concentrating future search there, rather than
sampling blind every time.

**`MedianPruner` compares a trial's intermediate score against the *median*
of other trials at the same step, and stops the trial the moment it falls
behind.** As `model.fit(..., eval_set=...)` trains, Optuna's pruning
integration reports the validation metric after each boosting round (a
`trial.report(value, step)` call under the hood). `MedianPruner` maintains,
for each step number, the median of all *completed* (non-pruned) trials'
reported values at that same step; if the current trial's value at step
`k` is worse than that median, the trial is pruned — its `fit()` call is
aborted — on the reasoning that a trial already behind the pack halfway
through training rarely catches up to become the best trial overall.
`n_startup_trials=5` disables pruning for the first 5 trials specifically
because there aren't yet enough completed trials to compute a meaningful
median to compare against.

**Parameter importance from `get_param_importances` is permutation
importance (Level 2 Module 07) applied to a surrogate model of the search
itself.** Optuna fits a random forest regressor where each row is one
completed trial, the input features are that trial's sampled
hyperparameter values, and the target is the resulting objective score.
Permutation importance is then computed on *that* surrogate model exactly
as in Module 07 — shuffle one hyperparameter's values across trials, measure
the surrogate's drop in predictive accuracy for the objective. A
hyperparameter with high importance is one whose value, across the 50
actual trials run, was strongly and consistently predictive of the
resulting score — which can differ sharply from "the hyperparameter with
the widest search range," since a wide but irrelevant range would show
essentially no importance despite being heavily sampled.

## Exercise

Rerun `study.optimize` with `optuna.samplers.RandomSampler(seed=42)`
instead of `TPESampler`, same `n_trials=50` budget. Plot (or print) the
running best score after each trial for both samplers on the same axes.
Report at what trial count, if any, TPE's running-best pulls ahead of
random search's, and connect the gap (or lack of one) to the "search
concentrates on good regions over time" mechanism above.
