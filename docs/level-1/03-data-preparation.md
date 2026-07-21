# 03 · Data Preparation

Models are only as good as the data you feed them, and real data arrives
messy: missing values, wildly different column scales, text categories that
models can't consume. This module covers the standard preparation steps —
splitting, imputing, scaling, encoding — and the single most important
concept in practical ML: **data leakage**, the silent bug that makes bad
models look good.

## Train/test split: the golden rule

The whole point of ML is predicting data you *haven't seen*. So before doing
anything else, set aside a test set and don't touch it until the very end:

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

housing = fetch_california_housing(as_frame=True)
X, y = housing.data, housing.target      # 20,640 districts, 8 features

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(X_train.shape, X_test.shape)
# (16512, 8) (4128, 8)
```

Evaluating on training data is meaningless — a model can simply memorize it.
A 1-nearest-neighbor model scores 100% on its own training set every time.

For classification, add `stratify=y` so each class appears in the same
proportion in both splits:

```python
from sklearn.datasets import load_iris
Xi, yi = load_iris(return_X_y=True)
Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(
    Xi, yi, test_size=0.25, random_state=42, stratify=yi
)
```

## Handling missing values

Real datasets have holes. The built-in datasets don't, so let's poke some
holes to practice:

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
df = X_train.copy()
mask = rng.random(df.shape) < 0.05        # knock out ~5% of values
df = df.mask(mask)

print(df.isna().sum().head(3))
# MedInc      826
# HouseAge    832
# AveRooms    827
```

Options, roughly in order of preference:

```python
# 1. Impute with a per-column statistic (median is robust to outliers):
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
df_imputed = pd.DataFrame(
    imputer.fit_transform(df), columns=df.columns, index=df.index
)
print(df_imputed.isna().sum().sum())   # 0

# 2. Drop rows -- only sensible when few rows are affected:
df_dropped = df.dropna()
print(len(df), "->", len(df_dropped))  # 16512 -> ~11000 (loses a third!)
```

Dropping columns is a third option when a column is mostly empty. The
imputer *learns* the medians from the data it's fit on — remember that fact;
it's about to matter.

## Feature scaling

Look at the raw feature ranges:

```python
print(X_train.agg(["min", "max"]).round(1))
#      MedInc  HouseAge  AveRooms  ...  Population
# min     0.5       1.0       0.8  ...         3.0
# max    15.0      52.0     141.9  ...     35682.0
```

`Population` spans tens of thousands while `MedInc` spans ~15. Distance-based
models (k-NN, k-means), regularized linear models, and neural networks all
implicitly treat "bigger numbers" as "more important", so unscaled features
distort them. Two standard fixes:

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization: (x - mean) / std  ->  mean 0, std 1 per column
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
print(X_train_s.mean(axis=0).round(2))   # [ 0.  0.  0. ...]
print(X_train_s.std(axis=0).round(2))    # [ 1.  1.  1. ...]

# Min-max: squeeze each column into [0, 1]
minmax = MinMaxScaler()
X_train_m = minmax.fit_transform(X_train)
```

Use `StandardScaler` as the default. Tree-based models (decision trees,
random forests, gradient boosting) are scale-invariant and don't need this.

## The fit/transform pattern — and doing it right

Every preprocessor has two verbs:

- `fit(X_train)` — learn parameters (medians, means, stds) *from training
  data only*;
- `transform(X)` — apply them to any data.

The correct sequence never lets the test set influence the learned
parameters:

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # learn on train, apply to train
X_test_scaled  = scaler.transform(X_test)       # apply SAME params to test
```

## Encoding categorical features

Models need numbers, not strings. Two main encodings:

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

cities = pd.DataFrame({"city": ["Austin", "Boston", "Denver", "Boston"]})

# One-hot: one 0/1 column per category (default choice for nominal data)
ohe = OneHotEncoder(sparse_output=False)
print(ohe.fit_transform(cities))
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [0. 1. 0.]]
print(ohe.get_feature_names_out())
# ['city_Austin' 'city_Boston' 'city_Denver']

# Ordinal: one integer per category (only when order is meaningful!)
sizes = pd.DataFrame({"size": ["small", "large", "medium"]})
oe = OrdinalEncoder(categories=[["small", "medium", "large"]])
print(oe.fit_transform(sizes).ravel())   # [0. 2. 1.]
```

Using `OrdinalEncoder` on unordered categories (like city names) invents a
fake ranking — `Denver > Boston > Austin` — that linear models will happily
exploit. When categories have no order, one-hot them.

Pass `handle_unknown="ignore"` to `OneHotEncoder` so a category that appears
only in the test set encodes as all zeros instead of crashing.

## Data leakage: the bug that flatters you

**Leakage** is any information from outside the training set — usually from
the test set or from the future — sneaking into training. The model then
scores brilliantly in your notebook and falls apart in the real world.

The classic beginner version:

```python
# WRONG: scaler sees ALL rows, including the test set,
# so test-set statistics leak into training.
X_all_scaled = StandardScaler().fit_transform(X)          # ✗
X_tr, X_te = train_test_split(X_all_scaled, random_state=42)

# RIGHT: split FIRST, fit preprocessing on the training part only.
X_tr, X_te = train_test_split(X, random_state=42)          # ✓
scaler = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)
```

With scaling the damage is small; with imputation, target encoding, or
feature selection it can be enormous. Other leakage flavors to watch for:

- **Target leakage** — a feature that is a proxy for the answer
  (e.g. predicting loan default using a "sent_to_collections" column).
- **Temporal leakage** — training on data from *after* the moment you're
  predicting (shuffle-splitting time series).
- **Duplicate leakage** — near-identical rows landing in both train and test.

The rule that prevents most of it: **split first; fit everything (models
*and* preprocessors) on training data only.** Module 08 introduces
`Pipeline`, which enforces this automatically.

## Cheat sheet

| Task | Tool | Key detail |
|------|------|-----------|
| Hold out a test set | `train_test_split(X, y, test_size=0.2, random_state=42)` | Do it *first*; `stratify=y` for classification. |
| Fill missing values | `SimpleImputer(strategy="median")` | Fit on train only. |
| Drop missing rows | `df.dropna()` | Only if few rows affected. |
| Standardize features | `StandardScaler()` | Needed for k-NN, linear models, neural nets; not trees. |
| Scale to [0, 1] | `MinMaxScaler()` | Alternative to standardization. |
| Encode unordered categories | `OneHotEncoder(handle_unknown="ignore")` | One 0/1 column per category. |
| Encode ordered categories | `OrdinalEncoder(categories=[...])` | Only when order is real. |
| Learn vs. apply | `.fit()` / `.transform()` | `fit_transform` on train, `transform` on test. |

## Exercise

Take the California housing training set, knock out 5% of values at random
(code above), then build the correct preparation flow: median-impute, then
standardize — fitting both on `X_train` only and applying them to `X_test`.
Print the test set's per-column means after scaling: they should be *near*
zero but not exactly zero. Write one sentence explaining why exactly-zero
test means would actually be evidence of leakage.
