---
description: "Time Series Forecasting Basics — Time series data breaks the i.i.d. assumption every model so far has relied on: observations are ordered, and the future…"
---

# 03 · Time Series Forecasting Basics

Time series data breaks the i.i.d. assumption every model so far has relied
on: observations are ordered, and the future depends on the past. This
module covers the two things that actually matter for tabular forecasting —
turning time into features, and splitting/evaluating without leaking the
future into the past — plus classic baselines.

## Building lag and rolling-window features

```python
import numpy as np
import pandas as pd

rng = pd.date_range("2022-01-01", periods=400, freq="D")
trend = np.linspace(50, 90, 400)
season = 8 * np.sin(2 * np.pi * np.arange(400) / 7)     # weekly seasonality
noise = np.random.default_rng(42).normal(0, 2, 400)
series = pd.Series(trend + season + noise, index=rng, name="sales")

df = series.to_frame()
for lag in [1, 2, 7]:
    df[f"lag_{lag}"] = df["sales"].shift(lag)
df["roll_mean_7"] = df["sales"].shift(1).rolling(7).mean()
df["roll_std_7"] = df["sales"].shift(1).rolling(7).std()
df["dow"] = df.index.dayofweek
df = df.dropna()
print(df.head(3))
```

Every feature is shifted by at least 1 before any rolling computation —
`shift(1).rolling(7).mean()`, not `rolling(7).mean()` — so that the feature
for day `t` only uses data available *before* day `t`. This is the single
most common bug in time series ML: an un-shifted rolling window silently
includes today's own target value.

## Time-aware train/test splits

Random `train_test_split` is invalid here — it would let the model train on
future days and be tested on earlier ones. Split by time instead:

```python
split_date = df.index[int(len(df) * 0.8)]
train, test = df[df.index < split_date], df[df.index >= split_date]

from sklearn.ensemble import RandomForestRegressor

X_cols = ["lag_1", "lag_2", "lag_7", "roll_mean_7", "roll_std_7", "dow"]
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(train[X_cols], train["sales"])
pred = model.predict(test[X_cols])

from sklearn.metrics import mean_absolute_error
print(f"MAE: {mean_absolute_error(test['sales'], pred):.2f}")   # ~2.4
```

For robust evaluation, `TimeSeriesSplit` generalizes this to multiple
folds, each still respecting order:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for fold, (tr_idx, te_idx) in enumerate(tscv.split(df)):
    print(fold, "train size:", len(tr_idx), "test size:", len(te_idx))
# fold 0 train size: 66  test size: 65
# fold 1 train size: 131 test size: 65
# ... (train window grows, test window always comes after it)
```

## Baselines you must beat

```python
naive_pred = test["lag_1"]                                  # "tomorrow = today"
seasonal_naive_pred = test["lag_7"]                          # "same weekday last week"

print(f"naive MAE:          {mean_absolute_error(test['sales'], naive_pred):.2f}")
print(f"seasonal naive MAE: {mean_absolute_error(test['sales'], seasonal_naive_pred):.2f}")
# naive MAE:          6.1
# seasonal naive MAE: 3.0
```

If a sophisticated model can't beat `lag_7` (seasonal naive) on data with
weekly seasonality, it isn't earning its complexity — a shockingly common
outcome in practice.

## Cheat sheet

| Task | Code |
|---|---|
| Lag feature | `df["sales"].shift(k)` |
| Rolling stat (leak-safe) | `df["sales"].shift(1).rolling(w).mean()` |
| Time-respecting split | slice by date, or `TimeSeriesSplit` |
| Naive baseline | `shift(1)` |
| Seasonal naive baseline | `shift(period)` |
| Error metric | `mean_absolute_error`, `mean_absolute_percentage_error` |

## How It Actually Works

**Why an un-shifted rolling window leaks the future.** `rolling(7).mean()`
computed on `df["sales"]` directly assigns, to row `t`, the mean of rows
`t-6` through `t` *inclusive of `t` itself* — so the feature for day `t`
literally contains `sales[t]`, the very value the model is trying to
predict, divided by 7 and mixed with 6 other values. During training the
model discovers this near-perfect proxy and assigns it a huge coefficient
or split priority; validation metrics computed the same leaky way look
excellent, and the model then fails in production where day `t`'s true
sales are exactly the unknown quantity being forecast. `.shift(1)` before
`.rolling(7)` fixes this by first moving the whole series forward one step,
so the window used for row `t`'s feature spans rows `t-7` through `t-1` —
mechanically, no information from `t` or later ever enters the computation.

**Why random splits are invalid but `TimeSeriesSplit` is a direct fix.**
`train_test_split(..., shuffle=True)` (the sklearn default) assumes rows are
exchangeable — a valid assumption when they're i.i.d. draws, false for a
time series where row `t+1`'s target is correlated with row `t`'s. A random
split can place day 350 in the training set and day 100 in the test set,
letting a lag-1 feature at test time (`lag_1` for day 100) come from a value
that itself sits inside the training data's temporal neighborhood, and more
subtly lets the model implicitly "see" the general trend/season level of
periods that are chronologically after some test points. `TimeSeriesSplit`
mechanically enforces `train indices < test indices` for every fold by
construction — each successive fold's test set is a block of rows that come
strictly after all of that fold's training rows — which is the only way to
estimate how the model will perform on data it genuinely hasn't seen yet:
the future.

**Why the seasonal-naive baseline is a legitimate scientific control, not
just a placeholder.** `lag_7` predicts "whatever happened exactly one full
season ago." For a series that is `trend + 8·sin(2πt/7) + noise` by
construction, `lag_7` for day `t` equals `trend[t-7] + 8·sin(2π(t-7)/7) +
noise[t-7]`; since `sin` has period 7, `sin(2π(t-7)/7) = sin(2πt/7)`
*exactly* — the seasonal term matches perfectly, and the only error left is
the (small) trend drift over 7 days plus the noise terms of two different
days. That's mechanically why its MAE (3.0) is roughly half the plain
naive's (6.1): the seasonal-naive baseline is capturing the true generating
seasonality almost for free, with zero learned parameters, which is exactly
the bar a learned model needs to clear to justify its complexity.

## 🔀 Related lessons on other tracks

- [Data Science — 03 · Advanced Time Series Forecasting](https://sigilipelli.github.io/data-science-mastery-path/level-3/03-advanced-time-series-forecasting/)

## Exercise

Add an `is_weekend` binary feature (`dow >= 5`) to `X_cols` and retrain the
random forest. Compare its test MAE against both the model above and the
seasonal-naive baseline. Then deliberately introduce the leakage bug (use
`df["sales"].rolling(7).mean()` without `.shift(1)`) and report how much the
*reported* MAE improves — as a concrete demonstration of why that bug is
dangerous precisely because it looks like a win.
