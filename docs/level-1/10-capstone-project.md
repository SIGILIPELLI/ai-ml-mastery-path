# 10 · Capstone — End-to-End Tabular ML Project

Time to put every module together into one realistic project: predicting
California house values from census data. You'll structure it the way real
projects are structured — a small multi-file layout, not one giant notebook —
and walk the full arc: explore → prepare (leakage-free) → baseline → stronger
model → honest comparison → save the winner for reuse.

## Project layout

```text
housing-project/
├── explore.py       # EDA: look at the data before modeling
├── prepare.py       # shared: load data, engineer features, build the prep pipeline
├── train.py         # train, cross-validate, compare, evaluate on test
├── predict.py       # load the saved model and predict for new districts
└── model.joblib     # the trained pipeline (created by train.py)
```

The key design decision: *all* data logic lives in `prepare.py` and is
imported by both training and prediction — so the exact same code path that
prepared training data prepares future data. Divergence between those two
paths is where production ML bugs come from.

## Step 1 — Explore (`explore.py`)

Before modeling, look at the data. You're checking: what's the target's
shape? Any weird values? What correlates with price?

```python
# explore.py
import matplotlib
matplotlib.use("Agg")                      # save plots to files (no GUI needed)
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df = housing.frame                          # 20,640 rows, 8 features + target

print(df.shape)
print(df.describe().round(2).T[["mean", "min", "max"]])
#               mean     min       max
# MedInc        3.87    0.50     15.00
# HouseAge     28.64    1.00     52.00
# AveRooms      5.43    0.85    141.91   <- max 141 rooms/household?! outliers
# AveBedrms     1.10    0.33     34.07
# Population 1425.48    3.00  35682.00
# AveOccup      3.07    0.69   1243.33   <- 1243 people/household: data quirks
# Latitude     35.63   32.54     41.95
# Longitude  -119.57 -124.35   -114.31
# MedHouseVal   2.07    0.15      5.00   <- target is CAPPED at 5.0 ($500k)

# Correlation of each feature with the target:
print(df.corr(numeric_only=True)["MedHouseVal"].sort_values(ascending=False).round(3))
# MedHouseVal    1.000
# MedInc         0.688    <- income dominates
# AveRooms       0.152
# ...
# Latitude      -0.144

df["MedHouseVal"].hist(bins=50)
plt.savefig("target_hist.png")
```

Three findings that shape everything downstream: income is by far the
strongest single signal; `AveRooms`/`AveOccup` contain extreme outliers
(hence median-based thinking); and the target is capped at 5.0, so the model
can never be right about the most expensive districts — worth stating in any
report.

## Step 2 — Prepare (`prepare.py`)

```python
# prepare.py
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_data():
    """Return train/test splits with engineered features."""
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()

    # Feature engineering (Module 08): ratios expose per-household structure
    df["rooms_per_household"]  = df["AveRooms"] / df["AveOccup"]
    df["bedrooms_per_room"]    = df["AveBedrms"] / df["AveRooms"]
    df["people_per_room"]      = df["AveOccup"] / df["AveRooms"]

    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def make_preprocessor():
    """Numeric-only data here, so prep is impute + scale."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
```

## Step 3 — Train, compare, evaluate (`train.py`)

```python
# train.py
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

from prepare import load_data, make_preprocessor

X_train, X_test, y_train, y_test = load_data()

# --- Candidate models, all as full pipelines ---------------------------
candidates = {
    "linear": Pipeline([
        ("prep", make_preprocessor()),
        ("model", LinearRegression()),
    ]),
    "forest": Pipeline([
        ("prep", make_preprocessor()),
        ("model", RandomForestRegressor(n_estimators=100,
                                        random_state=42, n_jobs=-1)),
    ]),
}

# --- Compare with cross-validation (training data only) ----------------
for name, pipe in candidates.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=5,
                             scoring="r2", n_jobs=-1)
    print(f"{name:8s} CV R^2: {scores.mean():.3f} +/- {scores.std():.3f}")
# linear   CV R^2: 0.613 +/- 0.014
# forest   CV R^2: 0.806 +/- 0.006     <- the ensemble wins clearly

# --- Tune the winner ---------------------------------------------------
grid = GridSearchCV(candidates["forest"], {
    "model__n_estimators": [100, 300],
    "model__max_features": [0.5, 1.0],
}, cv=3, scoring="r2", n_jobs=-1)
grid.fit(X_train, y_train)
print("best params:", grid.best_params_)
# best params: {'model__max_features': 0.5, 'model__n_estimators': 300}
print(f"best CV R^2: {grid.best_score_:.3f}")     # ~0.81

# --- Final, one-time test evaluation (Module 07's recipe) --------------
best = grid.best_estimator_
y_pred = best.predict(X_test)
rmse = np.sqrt(((y_test - y_pred) ** 2).mean())
print(f"TEST  R^2:  {r2_score(y_test, y_pred):.3f}")            # ~0.82
print(f"TEST  MAE:  {mean_absolute_error(y_test, y_pred):.3f}") # ~0.32 ($32k)
print(f"TEST  RMSE: {rmse:.3f}")                                # ~0.50

# --- Persist the whole pipeline ---------------------------------------
joblib.dump(best, "model.joblib")
print("saved model.joblib")
```

Run it:

```bash
python train.py
```

Read the story in the numbers: the linear baseline explains ~61% of variance;
the random forest ~81% — worth the extra complexity. Typical predictions are
off by ~$32k (MAE) on a median-value scale of ~$207k. And because we saved
the *pipeline*, imputation and scaling are baked into the artifact.

## Step 4 — Use the saved model (`predict.py`)

```python
# predict.py
import joblib
import pandas as pd
from prepare import load_data

model = joblib.load("model.joblib")     # the entire prep+model pipeline

# Pretend these two districts just arrived from the real world:
_, X_test, _, y_test = load_data()
new_districts = X_test.iloc[:2]

preds = model.predict(new_districts)    # raw features in, prices out
for (idx, row), p, actual in zip(new_districts.iterrows(), preds, y_test.iloc[:2]):
    print(f"district {idx}: predicted ${p*100_000:,.0f}, actual ${actual*100_000:,.0f}")
# district 20046: predicted $51,000, actual $47,700
# district 3024:  predicted $75,000, actual $45,800
```

This file is deliberately tiny: load artifact, call `predict`. Anything more
elaborate at prediction time (manual scaling, ad-hoc feature math) would be a
red flag that training and serving have diverged.

## What "done" looks like

A checklist that generalizes to every tabular project you'll do after this
course:

- [x] EDA notes: target distribution, outliers, top correlations, data quirks (the cap!)
- [x] Split before everything; test set touched exactly once
- [x] All prep inside a `Pipeline` — leakage-proof by construction
- [x] Dumb baseline and a linear baseline before anything fancy
- [x] Model choice justified by cross-validated scores, not vibes
- [x] Final metrics reported in real units (MAE ≈ $32k)
- [x] One reusable artifact (`model.joblib`) + a minimal `predict.py`

## Cheat sheet

| Project stage | Tool / habit |
|---------------|--------------|
| EDA | `df.describe()`, `df.corr()["target"]`, histograms |
| Shared data logic | one `prepare.py` imported by train *and* predict |
| Baselines first | `DummyRegressor`, then `LinearRegression` |
| Compare fairly | `cross_val_score(pipe, X_train, y_train, cv=5)` |
| Tune the winner | `GridSearchCV` with `model__` params |
| Final claim | test set, once, in real units |
| Save / load | `joblib.dump(pipe, "model.joblib")` / `joblib.load` |

## Exercise

Extend the project three ways. (1) Add a `GradientBoostingRegressor` (or
`HistGradientBoostingRegressor` — much faster) as a third candidate and
report its CV score against the other two. (2) Drop the target-capped rows
(`y == 5.0`) from the *training* data only, retrain, and see whether test
MAE on uncapped districts improves. (3) Add a `region` categorical feature
(bin `Latitude` as in Module 08), route it through a `ColumnTransformer`
with `OneHotEncoder`, and confirm the whole thing still trains, saves, and
predicts through `predict.py` unchanged. Congratulations — you've completed
Level 1.
