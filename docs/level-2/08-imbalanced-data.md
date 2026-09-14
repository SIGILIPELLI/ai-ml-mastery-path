# 08 · Imbalanced Data & Advanced Evaluation

Module 05 introduced `class_weight="balanced"` as one fix for imbalance.
This module goes deeper: resampling strategies, ROC and precision-recall
curves, and choosing a decision threshold deliberately instead of accepting
scikit-learn's default of 0.5.

## A realistically imbalanced dataset

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=5000, weights=[0.97, 0.03], flip_y=0.01, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print("train class counts:", dict(zip(*__import__("numpy").unique(y_train, return_counts=True))))
# train class counts: {0: 3638, 1: 112}
```

## Resampling: oversampling, undersampling, SMOTE

```python
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score

def fit_and_report(X_tr, y_tr, label):
    clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    pred = clf.predict(X_test)
    print(f"{label:16s} precision={precision_score(y_test, pred):.3f} "
          f"recall={recall_score(y_test, pred):.3f}")

fit_and_report(X_train, y_train, "baseline")

ros = RandomOverSampler(random_state=42)
X_ro, y_ro = ros.fit_resample(X_train, y_train)          # duplicates minority rows
fit_and_report(X_ro, y_ro, "oversampled")

sm = SMOTE(random_state=42)
X_sm, y_sm = sm.fit_resample(X_train, y_train)            # synthesizes new minority rows
fit_and_report(X_sm, y_sm, "SMOTE")

rus = RandomUnderSampler(random_state=42)
X_ru, y_ru = rus.fit_resample(X_train, y_train)           # drops majority rows
fit_and_report(X_ru, y_ru, "undersampled")
# baseline        precision=0.833  recall=0.179
# oversampled     precision=0.220  recall=0.786
# SMOTE           precision=0.241  recall=0.821
# undersampled    precision=0.198  recall=0.857
```

All three resampling techniques are applied **only to the training set** —
never to `X_test`, which must stay a faithful sample of real-world class
proportions to give an honest performance estimate.

## ROC and precision-recall curves: evaluating across all thresholds

```python
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:, 1]

fpr, tpr, roc_thresh = roc_curve(y_test, proba)
print(f"ROC AUC: {roc_auc_score(y_test, proba):.3f}")   # ~0.93

prec, rec, pr_thresh = precision_recall_curve(y_test, proba)
print(f"Average precision: {average_precision_score(y_test, proba):.3f}")   # ~0.55
```

On heavily imbalanced data, **precision-recall AUC (average precision) is
more informative than ROC AUC** — ROC's false-positive rate is computed
against the huge majority class, so it stays deceptively low even when
precision on the rare class is poor.

## Choosing a threshold deliberately

```python
import numpy as np

f1_scores = 2 * prec * rec / (prec + rec + 1e-12)
best_idx = np.argmax(f1_scores)
best_threshold = pr_thresh[best_idx]
print(f"best threshold: {best_threshold:.3f}  (F1={f1_scores[best_idx]:.3f})")

custom_pred = (proba >= best_threshold).astype(int)
print(f"precision={precision_score(y_test, custom_pred):.3f} recall={recall_score(y_test, custom_pred):.3f}")
```

## Cheat sheet

| Technique | Effect |
|---|---|
| `class_weight="balanced"` | Reweights loss, no data duplication |
| `RandomOverSampler` | Duplicates minority rows |
| `SMOTE` | Synthesizes new minority rows by interpolation |
| `RandomUnderSampler` | Drops majority rows (loses data) |
| ROC AUC | Threshold-independent, but optimistic on imbalance |
| Average precision (PR AUC) | Better summary for rare-class problems |
| Custom threshold | `(proba >= t)` instead of the default 0.5 |

## How It Actually Works

**SMOTE creates synthetic points by linear interpolation in feature space,
not by copying.** For each minority-class point `x`, SMOTE finds its `k`
nearest minority-class neighbors (Module 05's k-NN distance mechanism),
picks one neighbor `x_n` at random, and generates a new synthetic point
`x_new = x + λ(x_n - x)` for a random `λ ∈ (0, 1)` — a point on the line
segment between `x` and its neighbor. Doing this repeatedly densifies the
minority class's region of feature space with plausible-looking new points
rather than exact duplicates, which is mechanically why SMOTE tends to
generalize better than `RandomOverSampler` (which literally duplicates
existing rows, giving the model zero new information, just more weight on
the same points) — though it can misfire when minority points from
different sub-clusters get connected across an empty region that shouldn't
contain minority examples.

**Why ROC AUC looks better than it should feel on imbalanced data.** ROC
plots true positive rate (`TP/(TP+FN)`, recall) against false positive rate
(`FP/(FP+TN)`) as the threshold sweeps. On a 97:3 dataset, `TN` is huge
(thousands of majority-class rows), so even a substantial *number* of false
positives is a tiny *fraction* of all negatives — `FPR = FP/(FP+TN)` stays
near 0 for a wide range of thresholds regardless of how many false alarms
are actually generated relative to the (few) true positives. Precision
(`TP/(TP+FP)`), by contrast, divides by `TP+FP` — a quantity dominated by
how many false alarms occurred, not diluted by the majority class's size —
which is exactly why the precision-recall curve exposes problems (like the
`baseline` model's dismal recall=0.179 despite a respectable precision)
that ROC AUC's denominator structurally hides.

**The threshold is a free knob completely external to model training, and
moving it is pure arithmetic on `predict_proba`.** `predict()` internally
does nothing more than `(predict_proba(X)[:, 1] >= 0.5).astype(int)` — the
0.5 cutoff is a convention, not something learned during `fit()`. Sweeping
the threshold and recomputing precision/recall at each value (which is
literally what `precision_recall_curve` does — it tries every distinct
`proba` value as a candidate cutoff and recomputes the confusion-matrix
counts from Module 05 at each) is why threshold selection requires zero
retraining: the probabilities are already computed once, and every
downstream metric is just a different way of drawing the line between
"predict positive" and "predict negative" across those same fixed numbers.

## Exercise

Repeat the SMOTE + logistic regression pipeline, but compare `SMOTE`
against `class_weight="balanced"` (no resampling at all) using average
precision as the metric, on 5 different `random_state` splits. Report
whether one approach consistently wins, and connect any instability you see
to how few true minority-class examples (112) the training set actually
contains.
