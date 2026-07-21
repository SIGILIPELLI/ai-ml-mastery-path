# 02 · NumPy & Pandas Essentials

Every model you'll ever train eats numbers arranged in rectangles: rows of
examples, columns of features. NumPy is the library that stores those
rectangles and does math on them at C speed; pandas wraps NumPy in labeled,
spreadsheet-like tables for real-world data wrangling. This module covers the
20% of both libraries that does 95% of the work in ML.

## NumPy arrays

A NumPy `ndarray` is a fixed-type, N-dimensional grid of numbers:

```python
import numpy as np

a = np.array([1.0, 2.0, 3.0, 4.0])
M = np.array([[1, 2, 3],
              [4, 5, 6]])

print(a.shape, a.dtype)   # (4,) float64
print(M.shape)            # (2, 3)  -> 2 rows, 3 columns
```

`shape` is the single most-checked property in ML code. A dataset of 150
examples with 4 features is shape `(150, 4)`; a mismatch between shapes is
the most common bug you'll hit.

Useful constructors:

```python
np.zeros((2, 3))          # 2x3 of 0.0
np.ones(5)                # [1. 1. 1. 1. 1.]
np.arange(0, 10, 2)       # [0 2 4 6 8]
np.linspace(0, 1, 5)      # [0.   0.25 0.5  0.75 1.  ]

rng = np.random.default_rng(seed=42)
rng.normal(size=(2, 2))   # random draws from a normal distribution
```

## Vectorized operations — and why they matter for ML

Arithmetic on arrays applies element-wise, with no Python loop:

```python
x = np.array([1.0, 2.0, 3.0])
y = np.array([10.0, 20.0, 30.0])

print(x + y)        # [11. 22. 33.]
print(x * y)        # [10. 40. 90.]
print(x ** 2)       # [1. 4. 9.]
print(x > 1.5)      # [False  True  True]
```

This is **vectorization**: the loop happens in optimized C, not Python.
Measure the difference yourself:

```python
import numpy as np, time

big = np.arange(1_000_000, dtype=np.float64)

t0 = time.perf_counter()
total = sum(v * v for v in big)          # Python loop
t1 = time.perf_counter()
total_np = np.sum(big * big)             # vectorized
t2 = time.perf_counter()

print(f"python loop: {t1 - t0:.3f}s, numpy: {t2 - t1:.4f}s")
# python loop: ~0.15s, numpy: ~0.002s  (roughly 50-100x faster; exact times vary)
```

Training a model means doing millions of multiply-adds; without
vectorization, ML in Python would be unusably slow. The practical rule: *if
you're writing a `for` loop over data points, look for the array operation
instead.*

Aggregations and the `axis` argument:

```python
M = np.array([[1., 2., 3.],
              [4., 5., 6.]])

print(M.sum())          # 21.0        (everything)
print(M.mean(axis=0))   # [2.5 3.5 4.5]  (down the columns -> per-feature mean)
print(M.mean(axis=1))   # [2. 5.]        (across the rows  -> per-example mean)
```

`axis=0` ("per column / per feature") is everywhere in ML — feature means,
feature standard deviations, per-feature minimums for scaling.

## Indexing and boolean masks

```python
v = np.array([10, 20, 30, 40, 50])
print(v[0], v[-1])       # 10 50
print(v[1:4])            # [20 30 40]

mask = v > 25
print(mask)              # [False False  True  True  True]
print(v[mask])           # [30 40 50]  -- select rows where condition holds

M = np.arange(12).reshape(3, 4)
print(M[0, :])           # first row
print(M[:, 1])           # second column
print(M[M[:, 0] > 3])    # rows whose first column exceeds 3
```

Boolean masking is how you'll filter datasets: "all rows where price > 500k",
"all predictions the model got wrong".

## Pandas DataFrames

A `DataFrame` is a table: named columns (each effectively a NumPy array) plus
a row index.

```python
import pandas as pd

df = pd.DataFrame({
    "city":     ["Austin", "Boston", "Austin", "Denver", "Boston"],
    "bedrooms": [3, 2, 4, 3, 1],
    "sqft":     [1500, 900, 2200, 1600, 600],
    "price":    [450_000, 610_000, 720_000, 520_000, 400_000],
})
print(df.head())
#      city  bedrooms  sqft   price
# 0  Austin         3  1500  450000
# 1  Boston         2   900  610000
# 2  Austin         4  2200  720000
# 3  Denver         3  1600  520000
# 4  Boston         1   600  400000

print(df.shape)      # (5, 4)
print(df.dtypes)     # city: object, the rest: int64
df.describe()        # count/mean/std/min/quartiles/max per numeric column
```

Loading real data is one line: `pd.read_csv("file.csv")`. scikit-learn's
built-in datasets can hand you a DataFrame directly:

```python
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
print(iris.frame.head(3))
#    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
# 0                5.1               3.5                1.4               0.2       0
# 1                4.9               3.0                1.4               0.2       0
# 2                4.7               3.2                1.3               0.2       0
```

## Selection and filtering

```python
df["price"]                     # one column -> Series
df[["city", "price"]]           # multiple columns -> DataFrame

df.loc[2]                       # row by index label
df.iloc[0:2]                    # rows by position

# Filtering with boolean masks (same idea as NumPy):
df[df["price"] > 500_000]
df[(df["city"] == "Austin") & (df["bedrooms"] >= 3)]

# New columns from old — vectorized, no loop:
df["price_per_sqft"] = df["price"] / df["sqft"]
```

Note the operators: `&` (and), `|` (or), `~` (not), with parentheses around
each condition — Python's plain `and`/`or` don't work on arrays.

## Group-by: split, apply, combine

`groupby` answers "per-category" questions — the backbone of exploratory data
analysis:

```python
print(df.groupby("city")["price"].mean())
# city
# Austin    585000.0
# Boston    505000.0
# Denver    520000.0

print(df.groupby("city").agg(
    n=("price", "size"),
    avg_price=("price", "mean"),
    max_sqft=("sqft", "max"),
))
#         n  avg_price  max_sqft
# city
# Austin  2   585000.0      2200
# Boston  2   505000.0       900
# Denver  1   520000.0      1600
```

## From pandas to scikit-learn and back

scikit-learn accepts DataFrames directly; the convention is `X` for the
feature table and `y` for the target column:

```python
X = df[["bedrooms", "sqft"]]     # features: DataFrame
y = df["price"]                  # target:   Series
# model.fit(X, y) ...

# and .to_numpy() / .values when you need the raw array:
print(X.to_numpy().shape)        # (5, 2)
```

## Cheat sheet

| Operation | NumPy | pandas |
|-----------|-------|--------|
| Shape of the data | `a.shape` | `df.shape` |
| First rows | `a[:5]` | `df.head()` |
| Select column | `M[:, 2]` | `df["col"]` |
| Filter rows | `M[M[:, 0] > 3]` | `df[df["col"] > 3]` |
| Per-feature mean | `M.mean(axis=0)` | `df.mean(numeric_only=True)` |
| Elementwise math | `x * 2 + 1` | `df["col"] * 2 + 1` |
| Per-group summary | — | `df.groupby("k")["v"].mean()` |
| Random numbers | `np.random.default_rng(42)` | — |
| To the other type | — | `df.to_numpy()` |

## Exercise

Load the iris dataset as a DataFrame (`load_iris(as_frame=True)`, then take
`.frame`). Using only vectorized operations (no `for` loops): (1) add a
column `petal_area` = petal length × petal width; (2) filter to flowers with
`sepal length (cm)` above the dataset's mean sepal length; (3) group by
`target` and report the mean `petal_area` per species. You should find that
the three species separate almost perfectly on petal area alone — remember
this for the classification module.
