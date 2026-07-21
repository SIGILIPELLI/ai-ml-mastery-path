# 08 · Feature Engineering & Pipelines

Two things separate tidy tutorial ML from real projects: the features rarely
arrive model-ready, and the preprocessing itself is where leakage bugs breed.
scikit-learn's answer to both is the **Pipeline** — a single object that
chains preprocessing and model, learns everything only from training data,
and travels as one unit through cross-validation, grid search, and
deployment. This module builds up to the standard real-world skeleton:
`ColumnTransformer` inside a `Pipeline`.

## Why pipelines exist

Recall the manual dance from Module 03 — `fit_transform` on train,
`transform` on test, in the right order, for every preprocessor. A `Pipeline`
does that dance for you and makes it impossible to get wrong:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000)),
])

pipe.fit(X_train, y_train)            # fits scaler on train, then model
print(f"test acc: {pipe.score(X_test, y_test):.3f}")   # test acc: 0.986
```

Calling `pipe.fit` runs `fit_transform` through each step in order, then fits
the final model; `pipe.predict`/`pipe.score` run only `transform` through the
steps. Crucially, inside cross-validation the whole pipeline is refit per
fold, so **preprocessing statistics never leak across folds**:

```python
scores = cross_val_score(pipe, X_train, y_train, cv=5)
print(f"CV: {scores.mean():.3f} +/- {scores.std():.3f}")   # CV: 0.977 +/- 0.011
```

This is the leakage-prevention property: you literally cannot forget to
re-fit the scaler per fold, because the pipeline owns it.

## Feature engineering: creating better columns

Models can only use the information you give them, in the form you give it.
Feature engineering means deriving columns that expose the signal more
directly. A small worked example:

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "length_m":  [2.0, 3.5, 1.2, 4.0, 2.8],
    "width_m":   [1.0, 2.0, 0.8, 2.5, 1.5],
    "built":     [1995, 2010, 1980, 2021, 2003],
    "city":      ["Austin", "Boston", "Austin", "Denver", "Boston"],
})

df["area_m2"]   = df["length_m"] * df["width_m"]     # interaction
df["age_years"] = 2026 - df["built"]                 # more meaningful unit
df["log_area"]  = np.log1p(df["area_m2"])            # tame skewed scales
```

Common recipes:

| Recipe | Example | Why |
|--------|---------|-----|
| Ratios / interactions | rooms ÷ households, length × width | Exposes relationships linear models can't invent. |
| Log transform | `np.log1p(price)` | Compresses long-tailed distributions. |
| Date decomposition | year, month, day-of-week from a timestamp | Calendars drive behavior. |
| Binning | age → age bracket | Lets linear models fit non-linear steps. |
| Aggregates | mean price per city (from *train* data only!) | Injects group-level context — leakage-prone, be careful. |
| Polynomial features | `PolynomialFeatures(degree=2)` | Automated interactions (from Module 04). |

Good features routinely beat fancier models. In the capstone you'll see
ratio features noticeably lift a plain linear model on housing data.

## ColumnTransformer: different prep for different columns

Real tables mix numeric and categorical columns, which need *different*
preprocessing. `ColumnTransformer` routes each group of columns through its
own steps and concatenates the results:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

numeric_features = ["area_m2", "age_years"]
categorical_features = ["city"]

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), numeric_features),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_features),
])

Xt = preprocess.fit_transform(df)
print(Xt.shape)                              # (5, 5): 2 numeric + 3 one-hot
print(preprocess.get_feature_names_out())
# ['num__area_m2' 'num__age_years' 'cat__city_Austin' 'cat__city_Boston' 'cat__city_Denver']
```

Each branch is itself a mini-pipeline (impute → scale, impute → encode).
This one object now encapsulates your entire data-prep policy.

## The full skeleton: prep + model, tuned together

Bolt a model onto the preprocessor and you have the standard shape of a
production scikit-learn workflow — one estimator, end to end. Let's do it on
a real mixed-type problem by binning California housing's latitude into a
categorical "region" column:

```python
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()
df["region"] = pd.cut(df["Latitude"], bins=[32, 34, 36, 38, 42],
                      labels=["south", "central", "bay", "north"])
X = df.drop(columns=["MedHouseVal", "Latitude"])
y = df["MedHouseVal"]

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = ["region"]

model = Pipeline([
    ("prep", ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])),
    ("rf", RandomForestRegressor(random_state=42, n_jobs=-1)),
])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

grid = GridSearchCV(model, {
    "rf__n_estimators": [100, 200],
    "rf__max_depth": [10, None],
}, cv=3, n_jobs=-1)
grid.fit(X_tr, y_tr)

print(grid.best_params_)
# {'rf__max_depth': None, 'rf__n_estimators': 200}
print(f"test R^2: {grid.score(X_te, y_te):.3f}")   # test R^2: ~0.80
```

Notice what just happened: imputation-free prep, encoding, scaling, model
fitting, and hyperparameter search all ran leakage-free, and the deliverable
is *one object* (`grid.best_estimator_`) that accepts a raw DataFrame and
returns predictions. That's exactly what you save to disk and ship —
which the capstone does with `joblib`.

## Cheat sheet

| Task | Code |
|------|------|
| Chain steps | `Pipeline([("scale", StandardScaler()), ("clf", model)])` |
| Quick anonymous version | `make_pipeline(StandardScaler(), model)` |
| Per-column-type prep | `ColumnTransformer([("num", ..., num_cols), ("cat", ..., cat_cols)])` |
| Untouched columns | `ColumnTransformer(..., remainder="passthrough")` |
| Select columns by dtype | `X.select_dtypes(include="number").columns` |
| Names after transform | `preprocess.get_feature_names_out()` |
| Tune through a pipeline | `"stepname__param"` (nested: `"prep__num__impute__strategy"`) |
| Inspect a fitted step | `pipe.named_steps["scale"].mean_` |
| Why pipelines | Leakage-proof CV + a single shippable object |

## Exercise

Build a `ColumnTransformer`-based pipeline for the region-augmented housing
data above, but add two engineered features first: `rooms_per_household`
(`AveRooms / AveOccup`) and `bedrooms_per_room` (`AveBedrms / AveRooms`).
Compare 5-fold CV R² of `LinearRegression` with and without the engineered
features, then swap in `RandomForestRegressor` and compare again. Which model
benefited more from feature engineering — and why would a tree need the
ratio spelled out less than a linear model does?
