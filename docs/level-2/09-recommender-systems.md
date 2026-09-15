---
description: "Recommender System Fundamentals — Recommenders predict which items a user will like, from a sparse matrix of past interactions. This module covers the two…"
---

# 09 · Recommender System Fundamentals

Recommenders predict which items a user will like, from a sparse
matrix of past interactions. This module covers the two foundational
approaches: neighborhood-based collaborative filtering, and matrix
factorization, plus how to evaluate ranked recommendations.

## The ratings matrix

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n_users, n_items = 200, 50
# Simulate a sparse ratings matrix: most user-item pairs are unobserved (NaN)
true_pref = np.random.default_rng(42).normal(size=(n_users, 20)) @ np.random.default_rng(1).normal(size=(20, n_items))
ratings = np.clip(true_pref + np.random.default_rng(2).normal(scale=0.5, size=true_pref.shape), 1, 5)
mask = np.random.default_rng(3).random(ratings.shape) < 0.15   # only 15% observed
observed = np.where(mask, ratings.round(), np.nan)

df = pd.DataFrame(observed, columns=[f"item_{i}" for i in range(n_items)])
print(f"observed ratings: {mask.sum()} of {mask.size} ({mask.mean():.1%})")
```

Real recommender datasets are typically >95% sparse — most users have
rated only a handful of the available items.

## User-based collaborative filtering

The idea: find users similar to you (by rating pattern), recommend what
they liked that you haven't seen yet.

```python
from sklearn.metrics.pairwise import cosine_similarity

filled = df.fillna(0).values                       # 0 = "no signal" for similarity purposes
user_sim = cosine_similarity(filled)
np.fill_diagonal(user_sim, 0)                       # exclude self-similarity

def predict_rating(user_idx, item_idx, k=10):
    sims = user_sim[user_idx]
    top_k = np.argsort(sims)[-k:]
    item_ratings = df.iloc[top_k, item_idx]
    weights = sims[top_k]
    mask_valid = item_ratings.notna()
    if mask_valid.sum() == 0:
        return df.iloc[:, item_idx].mean()          # fallback: item's global average
    return np.average(item_ratings[mask_valid], weights=weights[mask_valid])

print(f"predicted rating: {predict_rating(0, 5):.2f}")
```

## Matrix factorization: learning latent factors

Neighborhood methods scale poorly and struggle with sparsity. **Matrix
factorization** instead learns, for each user and item, a small vector of
latent factors such that their dot product approximates the observed
ratings.

```python
from sklearn.decomposition import NMF

R = df.fillna(0).values
model = NMF(n_components=10, init="random", random_state=42, max_iter=500)
W = model.fit_transform(R)      # (n_users, 10) -- user latent factors
H = model.components_           # (10, n_items) -- item latent factors

reconstructed = W @ H
print(reconstructed.shape)      # (200, 50) -- dense predicted rating for every user-item pair

observed_mask = ~df.isna().values
rmse = np.sqrt(np.mean((R[observed_mask] - reconstructed[observed_mask]) ** 2))
print(f"train RMSE on observed entries: {rmse:.3f}")
```

`W @ H` produces a *dense* rating prediction for every user-item pair,
including the ~85% that were never observed — this is the entire point of
factorization: filling in the sparse matrix.

## Worked example: top-N recommendations for one user

```python
user_id = 7
already_rated = df.iloc[user_id].notna()
scores = reconstructed[user_id].copy()
scores[already_rated] = -np.inf                     # never recommend what they've already rated
top5 = np.argsort(scores)[-5:][::-1]
print("recommend items:", [f"item_{i}" for i in top5])
```

## Evaluating rankings: precision@k

```python
def precision_at_k(true_liked_set, recommended_list, k=5):
    top_k = recommended_list[:k]
    hits = len(set(top_k) & true_liked_set)
    return hits / k

liked = set(np.where(ratings[user_id] >= 4)[0])       # "ground truth" from the full simulated matrix
print(f"precision@5: {precision_at_k(liked, list(top5), k=5):.2f}")
```

## Cheat sheet

| Approach | Idea | Scales to |
|---|---|---|
| User-based CF | Similar users → similar tastes | Small user bases |
| Item-based CF | Similar items → co-rated together | Slightly better scaling |
| Matrix factorization (`NMF`) | Learn compact latent factors | Large sparse matrices |
| `precision@k` / `recall@k` | Overlap between top-k and true likes | Any ranked recommender |

## How It Actually Works

**Cosine similarity mechanically compares rating *patterns*, ignoring
magnitude.** For two users represented as rating vectors `u` and `v` (with
unrated items as 0), `cosine_similarity = (u · v) / (‖u‖ ‖v‖)` — the dot
product measures how much the two vectors point the same direction, and
dividing by the norms removes the effect of vector *length* (a user who
rates everything 5 vs. one who rates everything 1 can still be judged
"similar" if their relative pattern across items agrees, unlike Euclidean
distance which would treat them as very far apart). This is why cosine
similarity, not raw dot product or Euclidean distance, is the standard
choice for rating data — it isolates agreement in taste from differences in
how generously each user rates overall.

**Non-negative matrix factorization solves an optimization problem, not a
lookup.** `NMF` seeks two non-negative matrices `W` (users × factors) and
`H` (factors × items) whose product `W @ H` minimizes the squared
reconstruction error `Σ (R_ij - (WH)_ij)²`, summed only over the *observed*
entries of `R` (unobserved entries, filled with 0 here, are effectively
told "not part of the loss" by only training against known masked
entries in a proper implementation — treating missing as literal 0 as
this simplified example does is a known caveat, addressed by the
exercise). This is solved iteratively (alternating least squares or
multiplicative updates, `max_iter=500` caps how many refinement rounds
run) — conceptually the same gradient-descent flavor of optimization as
Module 09's neural network training, except here the "parameters" are
literally the `W` and `H` matrices' entries rather than network weights.
Once trained, `W @ H` computes a rating estimate for *every* cell,
including cells that were never observed, purely because matrix
multiplication doesn't know or care which entries were used during
training — it just applies the learned factors uniformly.

**Latent factors are compressed, learned dimensions — not hand-labeled
"genre" scores.** Each of the 10 columns of `W` (and corresponding rows of
`H`) is a direction the optimization discovered that helps reconstruct the
observed ratings well; nothing forces factor 3 to mean "prefers comedies."
The dot product `W[user] · H[:, item]` predicts a high rating when a user's
latent vector and an item's latent vector point in similar directions in
this learned 10-dimensional space — mechanically identical to how cosine
similarity found "similar users" above, except now applied in a compressed
space learned specifically to explain the rating data, rather than the raw
sparse rating vectors themselves. This is exactly why factorization
degrades more gracefully with sparsity than neighborhood methods: two users
who never rated a single item in common can still end up with similar
latent vectors if their observed ratings are each explainable by a similar
combination of factors.

## Exercise

Modify the `NMF` training so the loss is computed *only* over observed
entries (mask `R` and `reconstructed` with `observed_mask` before computing
RMSE, as done in the worked example, but also verify by comparing train
RMSE on observed entries vs. RMSE on a held-out set of entries you
artificially mask out before fitting). Report the gap between the two RMSEs
and explain what it tells you about how well the factorization
generalizes to genuinely unseen user-item pairs.
