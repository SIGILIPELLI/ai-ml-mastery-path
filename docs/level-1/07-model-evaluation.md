# 07 · Model Evaluation & Cross-Validation

You've been scoring models on a single held-out test set. That works, but it
has two weaknesses: the score depends on which rows happened to land in the
test split, and if you *tune* against the test set you slowly overfit to it.
This module builds the honest evaluation habit used everywhere in practice —
k-fold cross-validation, proper hyperparameter search with `GridSearchCV`,
and the bias-variance vocabulary for diagnosing what's wrong with a model.

## The problem with one split

Train the same model on five different random splits:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)

for seed in range(5):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_tr, y_tr)
    print(f"seed {seed}: {model.score(X_te, y_te):.3f}")
# seed 0: 0.972
# seed 1: 0.986
# seed 2: 0.958
# seed 3: 0.993
# seed 4: 0.972
```

Same model, same data — accuracy wobbles by 3+ points depending on the
split. Any single number in that range is partly luck.

## k-fold cross-validation

Cross-validation (CV) removes the luck: split the data into k equal "folds",
then train k times, each time holding out a different fold as the test set.
Every row gets used for testing exactly once, and you report the mean ± std
of the k scores:

```python
from sklearn.model_selection import cross_val_score

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
scores = cross_val_score(model, X, y, cv=5)   # 5-fold CV
print(scores.round(3))        # [0.982 0.982 0.973 0.982 0.973]
print(f"{scores.mean():.3f} +/- {scores.std():.3f}")
# 0.979 +/- 0.004
```

Now you have both an estimate (97.9%) *and* its stability (±0.4%). Notes:

- `cv=5` or `cv=10` are the standard choices; for classifiers scikit-learn
  automatically stratifies the folds.
- Cost: k model fits instead of 1.
- Because the model is a `Pipeline`, the scaler is re-fit *inside each fold*
  on that fold's training portion — CV with preprocessing done leakage-free.
  If you scaled `X` once up front and then cross-validated, every fold's
  test data would have leaked into scaling.
- `cross_validate` (plural) returns multiple metrics and fit times if you
  need more than one score.

## Hyperparameter tuning done right: GridSearchCV

Parameters like `k` in k-NN, `max_depth` in trees, or `C` in logistic
regression are **hyperparameters** — you choose them, the model doesn't learn
them. Choosing them by repeatedly checking the *test set* burns its honesty.
`GridSearchCV` does it properly: for every combination in a grid, run CV *on
the training data only*, pick the best, and only then touch the test set
once.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("knn", KNeighborsClassifier()),
])

param_grid = {
    "knn__n_neighbors": [1, 3, 5, 7, 11, 15, 21],
    "knn__weights": ["uniform", "distance"],
}

search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
search.fit(X_train, y_train)

print(search.best_params_)
# {'knn__n_neighbors': 11, 'knn__weights': 'uniform'}
print(f"best CV score:  {search.best_score_:.3f}")        # 0.972
print(f"test score:     {search.score(X_test, y_test):.3f}")  # 0.965
```

Details worth internalizing:

- `knn__n_neighbors` — the `step-name__param` syntax reaches inside a
  pipeline.
- 7 × 2 combinations × 5 folds = 70 fits; `n_jobs=-1` parallelizes across
  CPU cores.
- `search` is itself a fitted model: after the search it refits the best
  combination on *all* training data automatically.
- The final test score (96.5%) is slightly below the best CV score (97.2%) —
  typical and honest: the CV score guided a choice, so it's mildly
  optimistic.
- For big grids, `RandomizedSearchCV` samples random combinations instead of
  trying all — usually finds a near-best answer at a fraction of the cost.

## The bias-variance tradeoff

The vocabulary that ties all of this together:

- **Bias** — error from a model too simple to capture the pattern. High-bias
  models *underfit*: train and CV scores are both poor, and close together.
- **Variance** — error from a model so flexible it fits the noise of its
  particular training sample. High-variance models *overfit*: train score
  excellent, CV score notably worse.

You can watch a model slide from high bias to high variance by turning one
knob:

```python
from sklearn.tree import DecisionTreeClassifier
import numpy as np

for depth in [1, 2, 4, 8, 16, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    cv = cross_val_score(tree, X_train, y_train, cv=5)
    tree.fit(X_train, y_train)
    train = tree.score(X_train, y_train)
    print(f"depth {str(depth):>4}: train {train:.3f}  cv {cv.mean():.3f}  gap {train - cv.mean():+.3f}")
# depth    1: train 0.923  cv 0.911  gap +0.012   <- high bias (underfit)
# depth    2: train 0.951  cv 0.932  gap +0.019
# depth    4: train 0.988  cv 0.937  gap +0.051   <- sweet spot region
# depth    8: train 1.000  cv 0.925  gap +0.075
# depth   16: train 1.000  cv 0.923  gap +0.077   <- high variance (overfit)
# depth None: train 1.000  cv 0.923  gap +0.077
```

(Your exact numbers may differ slightly; read the *pattern*.) The diagnosis
guide:

| Symptom | Diagnosis | Remedies |
|---------|-----------|----------|
| Train score low, CV score low & close | High bias (underfitting) | More features, more flexible model, less regularization |
| Train score ≈ perfect, CV score clearly lower | High variance (overfitting) | Simpler model, more data, regularization, fewer features |
| Both high, small gap | Healthy | Ship it (after one final test-set check) |

## The honest evaluation recipe

1. Split off a test set **once**. Lock it away.
2. Explore, preprocess (in pipelines), and tune with **CV on the training
   set only** — e.g. `GridSearchCV`.
3. When all decisions are final, evaluate on the test set **once**. That
   number is your claim about the real world.
4. Never go back and re-tune because the test number disappointed you — if
   you must iterate, you need a fresh test set (or nested CV).

## Cheat sheet

| Task | Code |
|------|------|
| 5-fold CV score | `cross_val_score(model, X_train, y_train, cv=5)` |
| Report | `f"{scores.mean():.3f} +/- {scores.std():.3f}"` |
| Multiple metrics | `cross_validate(model, X, y, scoring=["accuracy","f1"])` |
| Grid search | `GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)` |
| Pipeline params in grids | `"stepname__param": [values]` |
| Best result | `search.best_params_`, `search.best_score_` |
| Cheap big-grid search | `RandomizedSearchCV(..., n_iter=50)` |
| Non-default metric | `scoring="f1"` / `"r2"` / `"neg_mean_absolute_error"` |
| Diagnose fit | Compare train score vs. CV score (the gap) |

## Exercise

Using the breast cancer training split: build a
`Pipeline(StandardScaler, LogisticRegression(max_iter=5000))` and grid-search
`C` over `[0.001, 0.01, 0.1, 1, 10, 100]` (`C` is inverse regularization —
small C = strong regularization = simpler model). Print the CV mean for each
C (`search.cv_results_["mean_test_score"]`), identify where the model is
underfitting vs. overfitting along the C axis, then report the final
test-set score of the best model — once.
