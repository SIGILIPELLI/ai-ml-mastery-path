---
description: "Ensemble Methods Deep Dive — A single decision tree overfits easily — it memorizes noise along with signal. Ensembles fix this by combining many models so…"
---

# 01 · Ensemble Methods Deep Dive

A single decision tree overfits easily — it memorizes noise along with signal.
Ensembles fix this by combining many models so their individual errors
average out. This module covers the three core ensemble families: bagging,
random forests (bagging's most successful special case), and voting/stacking
across different model types.

## Bagging: bootstrap aggregating

Bagging trains many copies of the same model on different **bootstrap
samples** (random samples drawn *with replacement*, same size as the
original) and averages their predictions.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)
print(f"single tree:  {single_tree.score(X_test, y_test):.3f}")   # ~0.930

bagged = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=200,
    max_samples=1.0, bootstrap=True, random_state=42, n_jobs=-1,
)
bagged.fit(X_train, y_train)
print(f"bagged trees: {bagged.score(X_test, y_test):.3f}")        # ~0.958
```

Bagging alone typically buys a few points of accuracy — mostly by taming
variance, not bias.

## Random forests: bagging + feature randomness

A random forest adds one more source of randomness: at each split, only a
random subset of features is considered (not all of them). This decorrelates
the trees further, since strong trees no longer all split on the same
dominant feature first.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt", random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
print(f"random forest: {rf.score(X_test, y_test):.3f}")   # ~0.972

importances = rf.feature_importances_
top5 = sorted(zip(X.columns, importances), key=lambda p: -p[1])[:5]
for name, imp in top5:
    print(f"{name:28s} {imp:.3f}")
# worst perimeter              0.140
# worst concave points         0.132
# worst radius                 0.109
# mean concave points          0.086
# worst area                   0.078
```

`max_features="sqrt"` (the default for classification) means each split
picks from `sqrt(30) ≈ 5` candidate features rather than all 30 — the single
biggest lever distinguishing a random forest from plain bagged trees.

## Voting and stacking: combining different model *types*

Bagging combines many copies of one model type. Voting/stacking combine
genuinely different models, betting that their mistakes won't overlap.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

lr  = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(5))
tree = DecisionTreeClassifier(max_depth=5, random_state=42)

voter = VotingClassifier(
    estimators=[("lr", lr), ("knn", knn), ("tree", tree)], voting="soft"
)
voter.fit(X_train, y_train)
print(f"soft voting: {voter.score(X_test, y_test):.3f}")   # ~0.965

stack = StackingClassifier(
    estimators=[("lr", lr), ("knn", knn), ("tree", tree)],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
)
stack.fit(X_train, y_train)
print(f"stacking:    {stack.score(X_test, y_test):.3f}")   # ~0.972
```

`voting="soft"` averages `predict_proba` outputs (needs every member to
support probabilities); `"hard"` just majority-votes the class labels.
Stacking goes further: a **meta-model** (here, logistic regression) learns
*how* to combine the base models' predictions, rather than averaging them
with fixed, equal weight.

## Cheat sheet

| Technique | Reduces | Key knob |
|---|---|---|
| `BaggingClassifier` | Variance (overfitting) | `n_estimators`, `max_samples` |
| `RandomForestClassifier` | Variance, correlation between trees | `max_features`, `n_estimators` |
| `VotingClassifier(voting="soft")` | Variance across model *types* | which base models you pick |
| `StackingClassifier` | Both, adaptively | `final_estimator`, `cv` |

## How It Actually Works

**Bagging's variance reduction is a direct consequence of the statistics of
averaging correlated estimators.** If each of `n` trees has prediction
variance `σ²` and pairwise correlation `ρ`, the variance of their *average*
is `ρσ² + (1-ρ)σ²/n` — not `σ²/n` as it would be for independent estimators,
because bootstrap samples overlap heavily (each bootstrap sample contains
about 63.2% of the unique original rows, since the probability a given row
is *never* drawn in `n` draws-with-replacement is `(1 - 1/n)^n → 1/e ≈
0.368`). As `n_estimators` grows, the first term `ρσ²` stops shrinking — it's
a floor set by how correlated the trees are — which is exactly why random
forests add a second randomization (feature subsampling) on top of bagging:
lowering `ρ` lowers that floor directly, which plain bagging cannot do no
matter how many trees you add.

**Feature subsampling is what actually lowers `ρ`.** With all 30 features
available, a "strong" feature like `worst perimeter` wins the split-quality
contest (lowest Gini impurity, from Module 05) at the root of almost every
tree, making the resulting trees highly similar and their errors highly
correlated. Restricting each split to a random `sqrt(30) ≈ 5`-feature subset
means that in many trees the strong feature simply isn't in the candidate
set at that node, forcing the tree to split on a different, weaker-but-still
-useful feature instead. Averaged over hundreds of trees, this produces a
genuinely diverse ensemble whose individual errors are less likely to point
the same direction — which is the literal mechanism, not a heuristic, behind
random forests consistently beating bagged trees at the same `n_estimators`.

**Stacking's meta-model formalizes what soft voting does by hand.** Soft
voting averages probabilities with implicit equal weight
(`(p_lr + p_knn + p_tree)/3`). `StackingClassifier` instead builds a new
training set where each row's features are the base models' out-of-fold
predictions (produced via the `cv=5` internal cross-validation, so the
meta-model never sees a base model's prediction on data that model was
trained on — avoiding leakage) and the target is still the true label; it
then fits `final_estimator` (logistic regression) on *that* dataset. The
meta-model's learned coefficients are literally the optimal weights for
combining the three base predictions on held-out data, which is why
stacking can match or beat voting even when one base model (say, k-NN) is
noticeably weaker — the meta-model learns to down-weight it rather than
average it in at full strength.

## Exercise

Using the breast cancer split above, sweep `RandomForestClassifier`'s
`max_features` over `["sqrt", "log2", None]` (`None` = all 30 features,
i.e. plain bagging) with `n_estimators=300` fixed. Print test accuracy for
each and explain, using the correlation argument above, why `None` should
be the worst of the three — then confirm whether your run actually shows
that.
