# 01 · Setup & the Scientific Python Stack

Machine learning in Python runs on a small, stable stack of libraries:
**NumPy** (fast arrays), **pandas** (tables), **scikit-learn** (classical ML),
**matplotlib** (plots), and — when you get to neural networks — **PyTorch**.
This module gets all of them installed in an isolated environment and ends
with a complete, working "hello ML" script, so you see the whole
train-and-predict loop on day one.

## Install Python and create a virtual environment

You need Python 3.9 or newer. Check what you have:

```bash
python3 --version
# Python 3.11.6
```

Always work inside a **virtual environment** — an isolated folder of packages
per project, so one project's library versions can't break another's:

```bash
mkdir ml-lessons && cd ml-lessons
python3 -m venv .venv

# Activate it (macOS/Linux):
source .venv/bin/activate

# Windows (PowerShell):
# .venv\Scripts\Activate.ps1
```

Your prompt gains a `(.venv)` prefix while the environment is active. `pip`
now installs into `.venv/` instead of your system Python. Deactivate anytime
with `deactivate`.

## Install the stack

```bash
pip install numpy pandas scikit-learn matplotlib torch
```

Verify everything imports and print the versions:

```python
# check_stack.py
import numpy, pandas, sklearn, matplotlib, torch

print("numpy       ", numpy.__version__)
print("pandas      ", pandas.__version__)
print("scikit-learn", sklearn.__version__)
print("matplotlib  ", matplotlib.__version__)
print("torch       ", torch.__version__)
```

```bash
python check_stack.py
# numpy        2.x.x
# pandas       2.x.x
# scikit-learn 1.x.x
# matplotlib   3.x.x
# torch        2.x.x
```

If all five lines print, your setup is done. Exact versions don't matter for
this course — anything from the last few years works.

## Jupyter notebooks vs. scripts

ML work happens in two modes:

- **Jupyter notebooks** (`pip install notebook`, then `jupyter notebook`) —
  interactive cells, inline plots, great for exploration and this course's
  experiments.
- **Plain `.py` scripts** — run top to bottom with `python file.py`,
  version-control friendly, what production code looks like.

A good habit from day one: *explore* in a notebook, then *consolidate* what
worked into a script. Everything in this course runs identically in both. If
you'd rather stay in one tool, VS Code runs `.py` files and notebooks side by
side.

## Your first ML program

Here is the entire supervised-learning workflow in ~20 lines: load a dataset,
split it, train a model, and measure how well it predicts data it has never
seen. Don't worry about the details yet — every step gets its own module
later.

```python
# hello_ml.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load a small built-in dataset: 150 iris flowers,
#    4 measurements each, 3 species to predict.
X, y = load_iris(return_X_y=True)
print(X.shape, y.shape)          # (150, 4) (150,)

# 2. Hold out 25% of the rows as a test set the model never sees.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 3. Train ("fit") a k-nearest-neighbors classifier.
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 4. Predict the held-out flowers and score the predictions.
y_pred = model.predict(X_test)
print("accuracy:", accuracy_score(y_test, y_pred))
```

```bash
python hello_ml.py
# (150, 4) (150,)
# accuracy: 0.9736842105263158
```

That ~97% means the model correctly identified the species of 37 of the 38
held-out flowers from measurements alone. The three-step rhythm you just saw —
`fit`, `predict`, score — is the same for nearly every model in scikit-learn,
from linear regression to gradient-boosted trees.

## The pieces of the stack, and what each is for

| Library | Role in ML work |
|---------|-----------------|
| NumPy | N-dimensional arrays and fast vectorized math — the data format everything else shares. |
| pandas | Labeled tables (`DataFrame`) — loading, cleaning, and reshaping real-world data. |
| scikit-learn | Classical ML: models, preprocessing, pipelines, metrics, cross-validation. |
| matplotlib | Plotting — histograms, scatter plots, learning curves. |
| PyTorch | Tensors + automatic differentiation — building and training neural networks. |
| Jupyter | Interactive notebook environment for exploration. |

## Reproducibility: `random_state`

ML is full of randomness — shuffling data, initializing models. Passing
`random_state=42` (any fixed integer) makes those random choices repeatable,
so you get the same split and the same accuracy every run. Every example in
this course pins `random_state` so your numbers match the expected output.

## Cheat sheet

| Command | Purpose |
|---------|---------|
| `python3 -m venv .venv` | Create a virtual environment in `.venv/`. |
| `source .venv/bin/activate` | Activate it (prompt shows `(.venv)`). |
| `pip install numpy pandas scikit-learn matplotlib torch` | Install the full course stack. |
| `pip freeze > requirements.txt` | Snapshot exact versions for reproducibility. |
| `jupyter notebook` | Launch the notebook interface. |
| `model.fit(X_train, y_train)` | Train a scikit-learn model. |
| `model.predict(X_test)` | Predict labels for new data. |

## Exercise

Set up a fresh virtual environment and install the stack. Then modify
`hello_ml.py` two ways: (1) change `n_neighbors` to 1 and then to 15 and note
how accuracy changes; (2) change `test_size` to 0.5 and rerun. Write down —
in a comment at the top of the file — one sentence on why the accuracy moves
when you hold out more data. Keep this file; you'll understand every line of
it by the end of Level 1.
