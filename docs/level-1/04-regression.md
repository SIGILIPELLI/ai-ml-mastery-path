# 04 · Regression

Regression predicts a *number*: a house price, tomorrow's temperature, a
patient's length of stay. It's the simplest place to build ML intuition
because the model — a weighted sum of features — is something you can read
and reason about. This module goes from the idea of fitting a line to
training `LinearRegression` on real data, measuring it properly, and meeting
the two failure modes (underfitting and overfitting) that haunt every model
you'll ever train.

## The idea: a line through data

Linear regression models the target as a weighted sum of the features plus an
intercept:

```
price = w1·income + w2·house_age + ... + w8·latitude + b
```

"Training" means finding the weights `w` and intercept `b` that minimize the
squared difference between predictions and actual values on the training set
(least squares). scikit-learn does that in one call:

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

housing = fetch_california_housing(as_frame=True)
X, y = housing.data, housing.target          # target: median house value, $100k units

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print(model.intercept_.round(3))             # -36.941
for name, w in zip(X.columns, model.coef_.round(3)):
    print(f"{name:12s} {w:8.3f}")
# MedInc          0.449
# HouseAge        0.010
# AveRooms       -0.123
# AveBedrms       0.783
# Population     -0.000
# AveOccup       -0.004
# Latitude       -0.420
# Longitude      -0.434
```

You can *read* this model: each extra unit of median income adds ~$44,900 to
the predicted value, holding everything else fixed. That interpretability is
why linear regression remains the baseline of choice.

Predicting is one call:

```python
y_pred = model.predict(X_test)
print(y_pred[:3].round(2))    # [0.72 1.76 2.71]
print(y_test[:3].round(2).to_numpy())  # [0.48 0.46 5.00]
```

## Measuring regression: MAE, MSE, RMSE, R²

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"MAE  {mae:.3f}")    # MAE  0.533
print(f"MSE  {mse:.3f}")    # MSE  0.556
print(f"RMSE {rmse:.3f}")   # RMSE 0.746
print(f"R^2  {r2:.3f}")     # R^2  0.576
```

| Metric | Meaning | Notes |
|--------|---------|-------|
| MAE | Average absolute error, in target units | Robust to outliers; easiest to explain ("off by ~$53k on average"). |
| MSE | Average *squared* error | Punishes large errors heavily; what least squares minimizes. |
| RMSE | √MSE, back in target units | The usual headline number. |
| R² | Fraction of target variance explained (1.0 = perfect, 0.0 = no better than predicting the mean) | Scale-free; can go *negative* for terrible models. |

R² = 0.576 means the linear model explains ~58% of the variation in house
values — decent for 8 features, far from perfect.

Always sanity-check against a **dumb baseline**:

```python
from sklearn.dummy import DummyRegressor
baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
print(r2_score(y_test, baseline.predict(X_test)))   # -0.000
```

Any model worth keeping must beat "always predict the average".

## Polynomial features: curves from a linear model

Straight lines can't fit curved relationships — but a linear model *on
transformed features* can. Generate synthetic curved data and watch:

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

rng = np.random.default_rng(42)
x = rng.uniform(-3, 3, size=200)
y_curve = 0.5 * x**2 - x + 2 + rng.normal(0, 1, size=200)   # a noisy parabola
X1 = x.reshape(-1, 1)                                        # (200, 1)

# Straight line: underfits
lin = LinearRegression().fit(X1, y_curve)
print(f"degree 1 R^2: {r2_score(y_curve, lin.predict(X1)):.3f}")   # ~0.27

# Add x^2 as a feature: now the "line" is a parabola
poly = PolynomialFeatures(degree=2, include_bias=False)
X2 = poly.fit_transform(X1)          # columns: [x, x^2]
quad = LinearRegression().fit(X2, y_curve)
print(f"degree 2 R^2: {r2_score(y_curve, quad.predict(X2)):.3f}")  # ~0.80
print(quad.coef_.round(2), quad.intercept_.round(2))
# [-1.    0.5 ]  2.0x  -- it recovered the true coefficients
```

The model is still linear *in its weights*; we just gave it richer features.
This trick — engineer features, keep the model simple — is a recurring ML
theme.

## Overfitting vs. underfitting

If degree 2 helped, why not degree 25? Because past some point the model
stops learning the *pattern* and starts memorizing the *noise*:

```python
from sklearn.model_selection import train_test_split

Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
    X1, y_curve, test_size=0.3, random_state=0
)

for degree in [1, 2, 5, 15, 25]:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    Xd_tr = poly.fit_transform(Xc_tr)
    Xd_te = poly.transform(Xc_te)
    m = LinearRegression().fit(Xd_tr, yc_tr)
    print(f"degree {degree:2d}: "
          f"train R^2 {r2_score(yc_tr, m.predict(Xd_tr)):.3f}  "
          f"test R^2 {r2_score(yc_te, m.predict(Xd_te)):.3f}")
# degree  1: train R^2 0.263  test R^2 0.297   <- underfit: bad everywhere
# degree  2: train R^2 0.799  test R^2 0.815   <- just right
# degree  5: train R^2 0.803  test R^2 0.810
# degree 15: train R^2 0.808  test R^2 0.774   <- test starts slipping
# degree 25: train R^2 0.816  test R^2 0.560   <- overfit: great on train, poor on test
```

(Exact numbers vary slightly by version; the *shape* of the pattern is the
point.)

- **Underfitting**: model too simple — poor on training *and* test data.
  Fix: richer features, more flexible model.
- **Overfitting**: model too flexible — excellent on training data, poor on
  test data. Fix: simpler model, more data, or regularization.

The gap between train and test scores is your overfitting alarm. This
tradeoff has a formal name — bias vs. variance — that Module 07 develops
further.

## Regularization in one paragraph

`Ridge` and `Lasso` are linear regression plus a penalty on large weights,
controlled by `alpha` — the standard first defense against overfitting in
linear models:

```python
from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha=1.0).fit(X_train, y_train)     # shrinks weights smoothly
lasso = Lasso(alpha=0.01).fit(X_train, y_train)    # can zero weights out entirely
print(f"ridge test R^2: {ridge.score(X_test, y_test):.3f}")   # ~0.576
print((lasso.coef_ == 0).sum(), "features eliminated by lasso")
```

`.score()` on a regressor returns R² directly — a handy shortcut.

## Cheat sheet

| Task | Code |
|------|------|
| Fit linear regression | `LinearRegression().fit(X_train, y_train)` |
| Inspect the model | `model.coef_`, `model.intercept_` |
| Predict | `model.predict(X_test)` |
| Metrics | `mean_absolute_error`, `mean_squared_error`, `r2_score` |
| R² shortcut | `model.score(X_test, y_test)` |
| Baseline | `DummyRegressor(strategy="mean")` |
| Curved fits | `PolynomialFeatures(degree=d)` before the model |
| Regularized linear | `Ridge(alpha=...)`, `Lasso(alpha=...)` |
| Diagnose overfitting | Compare train vs. test scores |

## Exercise

Using California housing: (1) train `LinearRegression` on *only* the
`MedInc` column (`X_train[["MedInc"]]`) and report test RMSE and R²; (2)
train on all 8 features and compare; (3) add degree-2 polynomial features on
all 8 features (`PolynomialFeatures(degree=2)`) and compare again, reporting
both train and test R². Does the polynomial model overfit, underfit, or
neither? Justify with the train/test gap, not intuition.
