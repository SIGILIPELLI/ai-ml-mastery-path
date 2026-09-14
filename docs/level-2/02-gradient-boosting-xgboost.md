# 02 · Gradient Boosting & XGBoost

Where random forests build many independent trees and average them,
**boosting** builds trees *sequentially*, each one specifically trained to
correct the errors of the ensemble so far. It's usually the single strongest
technique for tabular data, and XGBoost/LightGBM are the production-standard
implementations.

## The core idea: fit the residuals

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

gbm = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
)
gbm.fit(X_train, y_train)
print(f"gradient boosting: {gbm.score(X_test, y_test):.3f}")   # ~0.972
```

Each of the 200 trees is shallow (`max_depth=3` — a "weak learner") and adds
only a small, `learning_rate`-scaled correction. No single tree tries to
solve the whole problem; the ensemble solves it collectively over many
small steps.

## XGBoost in practice

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=42,
)
xgb.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)], verbose=False,
)
print(f"xgboost: {xgb.score(X_test, y_test):.3f}")   # ~0.979

# Early stopping: stop adding trees once validation loss stalls
xgb_es = XGBClassifier(
    n_estimators=2000, learning_rate=0.05, max_depth=4,
    early_stopping_rounds=20, eval_metric="logloss", random_state=42,
)
xgb_es.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print("trees actually used:", xgb_es.best_iteration)   # e.g. 143 of 2000
```

`subsample`/`colsample_bytree` (row and column subsampling per tree) inject
bagging-style randomness into boosting to fight overfitting — XGBoost is,
under the hood, "boosting + a bit of bagging."

## Worked example: tuning the learning rate / n_estimators trade-off

```python
import numpy as np
from sklearn.metrics import accuracy_score

for lr, n in [(0.3, 50), (0.1, 150), (0.02, 800)]:
    m = XGBClassifier(n_estimators=n, learning_rate=lr, max_depth=3, random_state=42)
    m.fit(X_train, y_train)
    print(f"lr={lr:<5} n={n:<4} acc={accuracy_score(y_test, m.predict(X_test)):.3f}")
# lr=0.3   n=50   acc=0.965
# lr=0.1   n=150  acc=0.972
# lr=0.02  n=800  acc=0.979
```

Smaller `learning_rate` + more trees generally generalizes better (finer,
more cautious corrections) at the cost of training time — the classic
boosting trade-off.

## Cheat sheet

| Knob | Effect |
|---|---|
| `n_estimators` | More trees = more correction rounds; too many overfits without early stopping |
| `learning_rate` | Shrinks each tree's contribution; smaller = needs more trees, generalizes better |
| `max_depth` | Boosted trees are shallow (3-6) by design — depth here isn't "how complex a single learner is" |
| `subsample` / `colsample_bytree` | Row/column bagging per tree, fights overfitting |
| `early_stopping_rounds` | Stop once validation metric stops improving |

## How It Actually Works

**Gradient boosting literally performs gradient descent, but in "function
space" instead of weight space.** At each round `m`, the ensemble's current
prediction is `F_m(x)`. Gradient boosting computes the **negative gradient**
of the loss with respect to that prediction for every training point — for
log loss this works out to `residual_i = y_i - p_i` (true label minus
predicted probability, the same intuition as "how wrong am I, and in which
direction") — and fits the *next* tree `h_m(x)` to predict those residuals,
then updates `F_{m+1}(x) = F_m(x) + learning_rate · h_m(x)`. This is exactly
the `w ← w - lr·grad` update from Module 09's neural network training, except
the "parameter" being updated is the entire ensemble function `F`, and each
"gradient step" is itself a whole decision tree rather than a number. That's
why weak, shallow trees are used as the per-round learner: each one only
needs to nudge the ensemble a little in the right direction, not solve the
problem outright.

**Why shrinking the learning rate and adding more trees usually generalizes
better.** A single large step (`lr=0.3`, few trees) commits hard to whichever
residual pattern the early trees found — including noise in that pattern.
Many small steps (`lr=0.02`, hundreds of trees) let the ensemble revise its
trajectory gradually, averaging out noisy residual estimates across rounds
much the way bagging averages out noisy trees — this is why the
`lr=0.02, n=800` configuration in the worked example outperforms `lr=0.3,
n=50` despite ending up as a "bigger" model.

**Early stopping is a direct read of the bias-variance trade-off along the
boosting trajectory.** Training loss decreases monotonically with more
rounds (each new tree is fit specifically to reduce the current residual),
but validation loss follows a U-shape: it falls while new trees still
capture real signal in the residuals, then rises once trees start fitting
noise in residuals that no longer contains real signal. `early_stopping_
rounds=20` mechanically watches the validation metric after each round and
halts once it hasn't improved for 20 consecutive rounds, keeping the model
at (approximately) the bottom of that U rather than at the training-loss
minimum — which is why `best_iteration` (143) is far short of the 2000-tree
budget: that's the empirically located bottom of the U for this dataset.

## Exercise

Reproduce the "worked example" sweep, but add a fourth setting `lr=0.3,
n=800` (large learning rate, many trees). Explain, from the negative-gradient
mechanism above, why this configuration is likely to *overfit* rather than
just take longer to converge — then confirm by comparing its train accuracy
(should be ~1.0) against its test accuracy.
