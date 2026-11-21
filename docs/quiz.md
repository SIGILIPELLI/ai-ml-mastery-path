---
title: "Final Quiz"
description: "Test what you have learned across all four levels of AI \u0026 Machine Learning Mastery Path with a 20-question final quiz."
---

# Final Quiz

20 questions drawn from all four levels of this course. You need **70% (14 out of 20)** to pass —
you can retake it as many times as you like. Your best result is saved in this browser.

<div id="mp-quiz-root"></div>

<script type="application/json" id="mp-quiz-data">
[
  {
    "question": "What shape does `np.array([[1, 2, 3], [4, 5, 6]])` have?",
    "options": [
      "(2,)",
      "(3, 2)",
      "(6,)",
      "(2, 3)"
    ],
    "correctIndex": 3,
    "explanation": "2 rows and 3 columns; shape is the most-checked property in ML code."
  },
  {
    "question": "For `M = np.array([[1., 2., 3.], [4., 5., 6.]])`, what does `M.mean(axis=0)` return?",
    "options": [
      "[2.5 3.5 4.5]",
      "[2. 5.]",
      "[1. 4.]",
      "3.5"
    ],
    "correctIndex": 0,
    "explanation": "axis=0 goes down the columns, giving one mean per column."
  },
  {
    "question": "Which line is leakage-free?",
    "options": [
      "`pipe = Pipeline([(\"scale\", StandardScaler()), (\"clf\", LogisticRegression())]); cross_val_score(pipe, X, y, cv=5)`",
      "`SimpleImputer().fit(X)` before splitting",
      "`X = StandardScaler().fit_transform(X); cross_val_score(model, X, y, cv=5)`",
      "`StandardScaler().fit(X_test)`"
    ],
    "correctIndex": 0,
    "explanation": "Inside a Pipeline the scaler is re-fit on each fold's training portion only."
  },
  {
    "question": "You evaluate a 1-nearest-neighbor model on its own training set and get 100%. What does it tell you?",
    "options": [
      "The model is underfitting",
      "Nothing: it memorized the data; evaluate on held-out data",
      "The data is clean",
      "The model is perfect"
    ],
    "correctIndex": 1,
    "explanation": "A model can simply memorize training data."
  },
  {
    "question": "Which call keeps the class ratio the same in train and test?",
    "options": [
      "`train_test_split(X, y, test_size=1)`",
      "`train_test_split(X, y, shuffle=False)`",
      "`train_test_split(X)`",
      "`train_test_split(X, y, stratify=y)`"
    ],
    "correctIndex": 3,
    "explanation": "stratify=y preserves class proportions."
  },
  {
    "question": "A fraud model has 3% positives. It scores 97% accuracy by predicting 'no fraud' always. What should you use instead?",
    "options": [
      "Precision/recall and PR curves, with a deliberately chosen threshold",
      "MAE",
      "R²",
      "Accuracy"
    ],
    "correctIndex": 0,
    "explanation": "Accuracy lies on imbalanced data."
  },
  {
    "question": "Which technique may you apply to the test set when handling imbalance?",
    "options": [
      "Oversampling",
      "Undersampling",
      "None: resampling is only for the training set",
      "SMOTE"
    ],
    "correctIndex": 2,
    "explanation": "The test set must keep real-world class proportions."
  },
  {
    "question": "k-means with k=4 gives inertia 942.3 and with k=5 gives 868.7, after 3673.8 at k=3. What does the elbow suggest?",
    "options": [
      "k=4, because improvements level off after it",
      "k=3",
      "k=1",
      "k=5"
    ],
    "correctIndex": 0,
    "explanation": "Big drops until k=4, then marginal gains."
  },
  {
    "question": "You fit a `StandardScaler` before `KMeans` because...",
    "options": [
      "KMeans can't handle floats",
      "Scaling adds clusters",
      "KMeans requires labels",
      "k-means is pure distance math, so feature scale would dominate"
    ],
    "correctIndex": 3,
    "explanation": "Unscaled large-range features dominate distances."
  },
  {
    "question": "Which metric tells you how much of the target's variance a regression explains?",
    "options": [
      "Accuracy",
      "F1",
      "R²",
      "AUC"
    ],
    "correctIndex": 2,
    "explanation": "R² compares your model with predicting the mean."
  },
  {
    "question": "A random forest differs from plain bagging because it...",
    "options": [
      "trains sequentially",
      "needs no data",
      "uses one tree",
      "considers only a random subset of features at each split"
    ],
    "correctIndex": 3,
    "explanation": "That extra randomness decorrelates the trees."
  },
  {
    "question": "Gradient boosting differs from a random forest because...",
    "options": [
      "it needs no learning rate",
      "trees are built sequentially, each correcting the errors so far",
      "trees are independent and averaged",
      "it uses no trees"
    ],
    "correctIndex": 1,
    "explanation": "Boosting fits residuals step by step."
  },
  {
    "question": "For time-series features you write `df['sales'].shift(1).rolling(7).mean()`. Why shift(1)?",
    "options": [
      "So the feature for day t only uses data before day t",
      "To normalize",
      "To drop NaNs",
      "To make it faster"
    ],
    "correctIndex": 0,
    "explanation": "An unshifted rolling window includes today's own target value, leaking the label."
  },
  {
    "question": "You split a daily-sales dataset for forecasting. Which split is valid?",
    "options": [
      "Random train_test_split",
      "Shuffle then split",
      "Test on the earliest dates",
      "Train on earlier dates, test on later dates (TimeSeriesSplit)"
    ],
    "correctIndex": 3,
    "explanation": "A random split lets the model train on the future."
  },
  {
    "question": "In PyTorch, `w = torch.tensor(3.0, requires_grad=True); loss = (w - 5) ** 2; loss.backward()` leaves `w.grad` equal to...",
    "options": [
      "-2",
      "-4",
      "4",
      "0"
    ],
    "correctIndex": 1,
    "explanation": "d/dw (w−5)² = 2(w−5) = −4 at w = 3."
  },
  {
    "question": "In a training loop, what is the correct order inside each step?",
    "options": [
      "optimizer.zero_grad(); loss.backward(); optimizer.step()",
      "loss.backward(); optimizer.zero_grad(); optimizer.step()",
      "optimizer.step(); loss.backward()",
      "loss.backward() only"
    ],
    "correctIndex": 0,
    "explanation": "Clear gradients, compute new ones, then update the weights."
  },
  {
    "question": "Why freeze the backbone with `requires_grad = False` in transfer learning?",
    "options": [
      "So only the new head trains and the pretrained features are kept",
      "To remove layers",
      "To free memory only",
      "To speed up the GPU"
    ],
    "correctIndex": 0,
    "explanation": "Only the new final layer's parameters are updated."
  },
  {
    "question": "A teammate's serving code computes `avg_purchase_30d` differently from training. What is this called?",
    "options": [
      "Overfitting",
      "Training/serving skew",
      "Data drift",
      "Underfitting"
    ],
    "correctIndex": 1,
    "explanation": "Duplicated feature logic in two code paths is a top production failure."
  },
  {
    "question": "Which tool compares the production distribution of a feature with its training distribution?",
    "options": [
      "Cross-validation",
      "MAE",
      "Population Stability Index (PSI)",
      "Accuracy"
    ],
    "correctIndex": 2,
    "explanation": "PSI above about 0.25 signals significant drift."
  },
  {
    "question": "You want to deploy a PyTorch model to run without PyTorch installed. What do you export to?",
    "options": [
      "ONNX",
      "pickle only",
      "JSON",
      "CSV"
    ],
    "correctIndex": 0,
    "explanation": "ONNX Runtime serves the model framework-independently."
  }
]
</script>

## Coding challenges

Try these on your own computer, then compare your output. Nothing you write is sent anywhere.

### Train/test split by hand

Without any libraries, split the list `data = list(range(10))` into a training list (first 80%) and a test list (last 20%) in order, as you would for time-ordered data, and print both lists.

**Starter code**

```python
data = list(range(10))
# split in order: first 80% train, last 20% test
# print(train) then print(test)
```

**Expected output**

```text
[0, 1, 2, 3, 4, 5, 6, 7]
[8, 9]
```

<details><summary>Show solution and explanation</summary>

<pre><code>data = list(range(10))
cut = int(len(data) * 0.8)
train, test = data[:cut], data[cut:]
print(train)
print(test)</code></pre>

<p>For time-ordered data you split by position, never randomly, so the model never trains on the future.</p>
</details>

### Mean absolute error

Write a function `mae(y_true, y_pred)` in plain Python that returns the mean absolute error, and print it for `y_true = [3, -0.5, 2, 7]` and `y_pred = [2.5, 0.0, 2, 8]`.

**Starter code**

```python
def mae(y_true, y_pred):
    # your code here
    pass

print(mae([3, -0.5, 2, 7], [2.5, 0.0, 2, 8]))
```

**Expected output**

```text
0.5
```

<details><summary>Show solution and explanation</summary>

<pre><code>def mae(y_true, y_pred):
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)

print(mae([3, -0.5, 2, 7], [2.5, 0.0, 2, 8]))</code></pre>

<p>The absolute errors are 0.5, 0.5, 0 and 1, which sum to 2 and average to 0.5.</p>
</details>
