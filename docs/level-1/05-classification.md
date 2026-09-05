# 05 · Classification

Classification predicts a *category*: spam or not, which species, will this
customer churn. It's the workhorse of applied ML. This module trains three
classic classifiers — logistic regression, decision trees, and k-NN — on the
same dataset, then spends serious time on the part beginners skip: how to
*measure* a classifier, and why accuracy alone can lie to you.

## The dataset

The breast cancer dataset: 569 tumors, 30 numeric features, binary target
(0 = malignant, 1 = benign).

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
print(X.shape)                     # (569, 30)
print(y.value_counts().to_dict())  # {1: 357, 0: 212}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
```

`stratify=y` keeps the 63/37 class ratio identical in both splits.

## Logistic regression

Despite the name, logistic regression is a *classifier*. It computes a
weighted sum of features (like linear regression) and squashes it through a
sigmoid into a probability between 0 and 1; predictions are made by
thresholding at 0.5.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
logreg.fit(X_train, y_train)
print(f"accuracy: {logreg.score(X_test, y_test):.3f}")   # accuracy: 0.986

# Probabilities, not just labels:
proba = logreg.predict_proba(X_test[:3])
print(proba.round(3))
# [[0.    1.   ]
#  [1.    0.   ]
#  [0.002 0.998]]   # columns: P(class 0), P(class 1)
```

(We scale inside a small pipeline because logistic regression, like most
non-tree models, cares about feature scale.) `predict_proba` is a big deal in
practice: "97% likely benign" supports very different decisions than "51%
likely benign", even though `predict` returns 1 for both.

## Decision trees

A decision tree asks a sequence of threshold questions ("worst radius
≤ 16.8?") and routes each sample to a leaf. Trees need no scaling and are
directly inspectable:

```python
from sklearn.tree import DecisionTreeClassifier, export_text

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)
print(f"accuracy: {tree.score(X_test, y_test):.3f}")   # accuracy: 0.944

print(export_text(tree, feature_names=list(X.columns), max_depth=2))
# |--- worst radius <= 16.80
# |   |--- worst concave points <= 0.14
# |   |   |--- ...
# |--- worst radius >  16.80
# |   |--- mean texture <= 16.11
# ...
```

`max_depth` is the crucial knob: an unrestricted tree grows until it
memorizes the training set (100% train accuracy, worse test accuracy —
overfitting again). Shallow trees underfit; try `max_depth=None` vs. `3` and
compare train/test scores.

## k-nearest neighbors

k-NN has no training phase at all: to classify a point, find the `k` closest
training points and take a majority vote.

```python
from sklearn.neighbors import KNeighborsClassifier

knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
knn.fit(X_train, y_train)
print(f"accuracy: {knn.score(X_test, y_test):.3f}")   # accuracy: 0.965
```

Scaling is *essential* here — distances are meaningless when one feature
spans thousands and another spans decimals. Small `k` → flexible, noisy
(overfits); large `k` → smooth, blurry (underfits).

## Beyond accuracy: the confusion matrix

Accuracy answers "what fraction was right?" — but not *what kind* of wrong.
For tumors, calling a malignancy benign (false negative) is far worse than
the reverse. The confusion matrix shows all four outcomes:

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = logreg.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
# [[52  1]     rows = actual, cols = predicted
#  [ 1 89]]
```

|  | Predicted malignant (0) | Predicted benign (1) |
|--|--|--|
| **Actually malignant (0)** | 52 (true neg.) | 1 (**false pos.** — missed cancer!) |
| **Actually benign (1)** | 1 (false neg.) | 89 (true pos.) |

(Here class 1 = benign is the "positive" class by sklearn convention.)

## Precision, recall, F1

Three numbers summarize the matrix from the positive class's point of view:

- **Precision** = TP / (TP + FP) — *when the model says positive, how often
  is it right?*
- **Recall** = TP / (TP + FN) — *of all actual positives, how many did the
  model find?*
- **F1** = harmonic mean of the two — a single balanced score.

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=["malignant", "benign"]))
#               precision    recall  f1-score   support
#    malignant       0.98      0.98      0.98        53
#       benign       0.99      0.99      0.99        90
#     accuracy                           0.99       143
```

Precision and recall trade off against each other: lower the decision
threshold (call more things positive) and recall rises while precision
falls. Which matters more is a *product* decision, not a math one — spam
filters want high precision (don't eat real mail); cancer screening wants
high recall (don't miss cases).

## Why accuracy lies: class imbalance

Make a dataset where 95% of examples are class 0:

```python
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.metrics import recall_score

Xi, yi = make_classification(
    n_samples=2000, weights=[0.95, 0.05], random_state=42
)
Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(
    Xi, yi, random_state=42, stratify=yi
)

dummy = DummyClassifier(strategy="most_frequent").fit(Xi_tr, yi_tr)
print(f"dummy accuracy: {dummy.score(Xi_te, yi_te):.3f}")       # 0.950 (!)
print(f"dummy recall:   {recall_score(yi_te, dummy.predict(Xi_te)):.3f}")  # 0.000
```

A model that *never detects the rare class* scores 95% accuracy. On
imbalanced problems (fraud, disease, defects — i.e., most interesting
problems), report precision/recall/F1 for the rare class, and consider
`class_weight="balanced"`:

```python
lr = make_pipeline(StandardScaler(),
                   LogisticRegression(class_weight="balanced", max_iter=1000))
lr.fit(Xi_tr, yi_tr)
print(f"recall on rare class: {recall_score(yi_te, lr.predict(Xi_te)):.3f}")  # ~0.9
```

`class_weight="balanced"` makes mistakes on the rare class cost
proportionally more during training — usually trading a little precision for
a lot of recall.

## Cheat sheet

| Task | Code / rule |
|------|-------------|
| Probabilistic linear classifier | `LogisticRegression(max_iter=1000)` — scale features first |
| Rule-based, no scaling needed | `DecisionTreeClassifier(max_depth=3)` |
| Distance-based | `KNeighborsClassifier(n_neighbors=5)` — must scale |
| Get probabilities | `model.predict_proba(X)` |
| All-in-one report | `classification_report(y_test, y_pred)` |
| Error breakdown | `confusion_matrix(y_test, y_pred)` |
| "Model says yes — trust it?" | Precision |
| "Did we find them all?" | Recall |
| Imbalanced data | Never trust bare accuracy; use `stratify=`, `class_weight="balanced"` |
| Sanity baseline | `DummyClassifier(strategy="most_frequent")` |

## How It Actually Works

**The sigmoid is what turns a linear score into a probability.** Logistic
regression computes the same weighted sum `z = w1x1 + ... + w30x30 + b` as
linear regression, then passes `z` through the **sigmoid function**
`σ(z) = 1 / (1 + e^(-z))`, which maps any real number to `(0, 1)`: very
negative `z` → near 0, very positive `z` → near 1, `z = 0` → exactly 0.5.
That's the mechanical reason `predict_proba` returns numbers instead of just
labels — the model's actual output *is* a probability, and `predict()`
simply thresholds it at 0.5. Training doesn't use least squares here;
instead it maximizes the **log-likelihood** of the observed labels under
this probability model (equivalently, minimizes **log loss**
`-[y·log(p) + (1-y)·log(1-p)]` averaged over training rows), solved by an
iterative optimizer (scikit-learn's default is `lbfgs`, a quasi-Newton
method) rather than a closed-form formula — which is why `max_iter=1000`
exists: it caps how many optimization steps the solver takes before giving
up.

**A decision tree is built by exhaustively testing splits, not by
choosing an algorithm in advance.** At each node, `DecisionTreeClassifier`
considers every feature and, for each, every candidate threshold that lies
between two adjacent sorted values of that feature in the data present at
that node. For each candidate split it computes the **Gini impurity** of
the two resulting child groups — `Gini = 1 - Σ p_c²` where `p_c` is the
fraction of each class in that group (0 = perfectly pure, up to 0.5 for a
50/50 binary split) — and picks the feature/threshold pair that produces
the largest drop in weighted impurity from parent to children. This repeats
recursively on each child. `max_depth` simply caps how many times this
recursive splitting can happen along any path from root to leaf; an
unrestricted tree keeps splitting until every leaf is pure (or has one
sample), which is precisely how it comes to memorize the training set
100% — a training-set impurity of exactly 0 is not a coincidence, it's the
literal stopping condition.

**Why k-NN has "no training" and what predict actually computes.** Because
`fit()` just stores the data, all the computation lives in `predict()`:
for a query point, compute Euclidean distance to every (scaled) training
point, take the `k=5` smallest distances, and return the majority class
among those 5 labels — ties broken by scikit-learn's internal ordering.
"Scaling is essential" is a direct consequence of that distance formula:
`distance² = Σ(x_i - q_i)²` sums squared differences across *all* features
with equal weight, so a feature whose raw values span thousands (e.g. an
unscaled `area` in µm²) dominates the sum and makes every other feature's
contribution to the distance numerically irrelevant — not a design flaw of
the metric, but an arithmetic consequence of unequal units being added
together.

**The confusion matrix comes straight from counting label pairs.**
`confusion_matrix(y_test, y_pred)` does nothing more than tally, for every
one of the `(actual, predicted)` pairs across all 143 test points, how many
fall into each of the 4 (or `n_classes²`) cells — literally a 2-D histogram
of `(y_test[i], y_pred[i])`. Precision, recall, and F1 are then pure
arithmetic on those counts (`TP/(TP+FP)`, `TP/(TP+FN)`, their harmonic
mean) — no separate model or estimation step, just ratios of the tallied
cells. That's also why `class_weight="balanced"` works mechanically: it
multiplies each training example's contribution to the log-loss by a
per-class weight (inversely proportional to that class's frequency), so
misclassifying one of the rare 5% of examples raises the loss roughly as
much as misclassifying nineteen majority-class examples — which is what
pushes the optimizer to stop ignoring the minority class.

## Exercise

On the breast cancer split above, train `DecisionTreeClassifier` with
`max_depth` in `[1, 3, 5, None]`. For each, print train accuracy, test
accuracy, and the test *recall for the malignant class*
(`recall_score(y_test, y_pred, pos_label=0)`). Identify which depth overfits
and which depth you would ship if missing a malignancy is 10× worse than a
false alarm — and note whether those are the same answer.
