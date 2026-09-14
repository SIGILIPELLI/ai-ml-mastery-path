# 04 · NLP Basics — Text Features & Classification

Text isn't numeric, but every model we've used so far requires numeric
input. This module covers the classic pipeline for turning text into
features — tokenization, bag-of-words, TF-IDF — and training a text
classifier on top, before Level 3 replaces this pipeline with transformers.

## From text to tokens to counts

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

categories = ["sci.space", "rec.sport.baseball"]
data = fetch_20newsgroups(subset="all", categories=categories,
                          remove=("headers", "footers", "quotes"))
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.25, random_state=42, stratify=data.target
)
print(len(X_train), "documents,", len(set(y_train)), "classes")

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words="english", max_features=5000)
X_train_counts = cv.fit_transform(X_train)
print(X_train_counts.shape)           # (~1180, 5000) -- sparse matrix
print(cv.get_feature_names_out()[:10])
```

`CountVectorizer` builds a vocabulary of the top 5000 words (after dropping
English stop words like "the", "is") and represents each document as a
vector of word counts — a **bag of words**: word order is thrown away
entirely.

## TF-IDF: down-weighting common words

Raw counts over-value words that are frequent everywhere ("game", "team" in
a sports+space corpus might both be common). **TF-IDF** rescales counts by
how *distinctive* a word is across the whole corpus.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)     # transform only -- reuse train vocabulary/IDF

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_tfidf, y_train)
print(f"accuracy: {clf.score(X_test_tfidf, y_test):.3f}")   # ~0.97
```

## Worked example: what the model actually learned

```python
import numpy as np

feature_names = np.array(tfidf.get_feature_names_out())
coefs = clf.coef_[0]
top_space = feature_names[np.argsort(coefs)[-10:]]
top_baseball = feature_names[np.argsort(coefs)[:10]]
print("most 'space' words:", top_space)
print("most 'baseball' words:", top_baseball)
# most 'space' words:    ['orbit' 'nasa' 'launch' 'shuttle' 'moon' ...]
# most 'baseball' words: ['pitching' 'braves' 'baseball' 'hitter' 'inning' ...]
```

A new document classified via the same `tfidf.transform([...])` pipeline:

```python
sample = ["The pitcher threw a curveball for strike three in the ninth inning."]
pred = clf.predict(tfidf.transform(sample))
print(data.target_names[pred[0]])   # rec.sport.baseball
```

## n-grams: recovering a little bit of order

```python
tfidf_bigram = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
X_train_bi = tfidf_bigram.fit_transform(X_train)
X_test_bi = tfidf_bigram.transform(X_test)
clf2 = LogisticRegression(max_iter=1000).fit(X_train_bi, y_train)
print(f"unigrams+bigrams accuracy: {clf2.score(X_test_bi, y_test):.3f}")
```

`ngram_range=(1, 2)` adds two-word phrases ("home run", "space shuttle") as
vocabulary entries alongside single words, partially recovering local word
order that bag-of-words otherwise discards.

## Cheat sheet

| Task | Code |
|---|---|
| Word counts | `CountVectorizer(stop_words="english")` |
| TF-IDF weighting | `TfidfVectorizer(stop_words="english")` |
| Fit vocabulary + transform | `.fit_transform(X_train)` (train only) |
| Reuse vocabulary | `.transform(X_test)` (never `fit_transform` on test) |
| Include phrases | `ngram_range=(1, 2)` |
| Inspect learned words | `feature_names_out()` + `clf.coef_` |

## How It Actually Works

**TF-IDF is a product of two independently interpretable numbers per
word-document pair.** For word `w` in document `d`: **TF** (term frequency)
is simply `count(w, d)` (or a normalized version), how often `w` appears in
*this* document. **IDF** (inverse document frequency) is
`log(N / df(w))` where `N` is the total number of documents and `df(w)` is
the number of documents containing `w` at least once — a word appearing in
every document gets `IDF ≈ log(1) = 0` (erased), while a word appearing in
only a handful of documents gets a large IDF. The TF-IDF weight is
`TF × IDF`: high for a word that appears often in this specific document
*and* rarely elsewhere. That's the literal mechanism behind "down-weighting
common words" — a word like "game" that's frequent in nearly every document
of a sports+space corpus gets driven toward zero by a small IDF regardless
of its raw count, while "orbit" (frequent only in space documents) keeps a
high weight in exactly the documents where it's distinctive.

**Why the vocabulary and IDF values must come only from training data.**
`fit_transform(X_train)` scans the training documents to build the
vocabulary (the top 5000 words by frequency) and compute each word's
document frequency `df(w)` for the IDF formula above — both are properties
*of the training set*. Calling `.transform(X_test)` (not `fit_transform`)
reuses that exact vocabulary and those exact IDF values, mapping test
documents into the same 5000-dimensional coordinate system without
recomputing anything from test data. If you instead called `fit_transform`
on the test set separately, its IDF values would differ (test-set word
frequencies aren't identical to train's) and, worse, produce a *different*
vocabulary with different feature indices — the resulting vectors wouldn't
even be comparable to what the classifier was trained on. This is the same
"fit on train, transform on test" discipline as `StandardScaler` from Level
1, applied to a much higher-dimensional transform.

**Why bag-of-words logistic regression still works despite discarding word
order.** Each document becomes a single point in 5000-dimensional space,
one axis per vocabulary word, and logistic regression (Module 05) finds a
hyperplane (`z = Σ w_i · tfidf_i + b`) separating the two classes. This
works well here because the *presence and weight of certain words* is
already highly predictive on its own — "orbit" appearing with high TF-IDF
weight is strong evidence for `sci.space` regardless of where in the
sentence it occurs — so a linear decision purely on word weights already
gets most of the signal. What's structurally lost is anything that depends
on word *order or context* (negation — "not a great game" vs. "a great
game" — or genuine multi-word idioms), which is precisely the gap n-grams
partially close (by adding "not_great" as its own vocabulary entry) and
which transformers in Level 3 close far more thoroughly by modeling
sequences directly instead of discarding order at the input stage.

## Exercise

Retrain the TF-IDF + logistic regression pipeline with `max_features` set
to `500`, `5000`, and `20000`. Plot or print test accuracy against
vocabulary size, and identify the point of diminishing (or negative)
returns. Explain, using the TF-IDF mechanism above, why accuracy might
*decrease* at very large vocabulary sizes on a training set of only ~1180
documents.
