# 06 · Clustering & Unsupervised Learning

Everything so far was *supervised*: we had labels (`y`) to learn from.
Unsupervised learning finds structure with no labels at all — natural
groupings of customers, compressed views of high-dimensional data, unusual
points that fit no group. This module covers the two unsupervised tools
you'll actually reach for: **k-means clustering** (with honest ways to choose
k) and **PCA** for dimensionality reduction, plus a glimpse of DBSCAN.

## k-means: find k centers

k-means picks `k` center points ("centroids") and assigns every sample to its
nearest centroid, iterating until centers stop moving. Synthetic blobs make
the behavior visible:

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y_true = make_blobs(
    n_samples=500, centers=4, cluster_std=1.0, random_state=42
)
print(X.shape)          # (500, 2)

km = KMeans(n_clusters=4, n_init="auto", random_state=42)
labels = km.fit_predict(X)

print(labels[:10])              # [1 3 0 3 2 ...]  -- a cluster id per sample
print(km.cluster_centers_.round(2))
# [[ -6.83  -6.83]
#  [ -2.63   9.04]
#  [  4.74   2.01]
#  [ -8.9    7.12]]
```

Note there's no `y` anywhere — `fit_predict` consumed only `X`. Plot it to
see the result:

```python
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=labels, s=12, cmap="viridis")
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
            marker="x", s=200, c="red")
plt.title("k-means, k=4")
plt.savefig("kmeans.png")       # or plt.show() in a notebook
```

Two practical warnings:

- **Scale your features first** (`StandardScaler`) on real data — k-means is
  pure distance math.
- Cluster IDs are arbitrary: "cluster 2" this run may be "cluster 0" next
  run. Only the *grouping* is meaningful.

## Choosing k: elbow and silhouette

Real data doesn't announce its k. Two standard diagnostics:

**Elbow method** — plot *inertia* (sum of squared distances to the nearest
centroid) for each k; look for the bend where improvements level off:

```python
for k in range(1, 9):
    km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(X)
    print(f"k={k}  inertia={km.inertia_:10.1f}")
# k=1  inertia=   28063.9
# k=2  inertia=   10654.7
# k=3  inertia=    3673.8
# k=4  inertia=     942.3   <- big drops until here...
# k=5  inertia=     868.7   <- ...then marginal: the "elbow" is at 4
# k=6  inertia=     791.8
```

Inertia *always* decreases as k grows (more centers = shorter distances), so
you look for the elbow, not the minimum.

**Silhouette score** — for each sample, compares distance to its own cluster
vs. the nearest other cluster; ranges −1..1, higher is better, and unlike
inertia it *peaks* at good k:

```python
from sklearn.metrics import silhouette_score

for k in range(2, 8):
    labels = KMeans(n_clusters=k, n_init="auto", random_state=42).fit_predict(X)
    print(f"k={k}  silhouette={silhouette_score(X, labels):.3f}")
# k=2  silhouette=0.657
# k=3  silhouette=0.706
# k=4  silhouette=0.792   <- peak: 4 clusters
# k=5  silhouette=0.702
# k=6  silhouette=0.588
```

Use both, and remember they're heuristics — on real data the "right" k is
often a business question ("how many customer segments can marketing actually
act on?").

## DBSCAN: a glimpse of density-based clustering

k-means assumes round, similar-sized clusters and forces *every* point into
one. DBSCAN instead grows clusters from dense regions — it discovers the
number of clusters itself, handles weird shapes, and labels sparse points as
noise (`-1`):

```python
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

Xm, _ = make_moons(n_samples=400, noise=0.06, random_state=42)

km_labels = KMeans(n_clusters=2, n_init="auto", random_state=42).fit_predict(Xm)
db_labels = DBSCAN(eps=0.2, min_samples=5).fit_predict(Xm)

import numpy as np
print(np.unique(km_labels))   # [0 1]     -- but it cuts each moon in half!
print(np.unique(db_labels))   # [0 1]     -- follows each crescent correctly
```

Plot both and you'll see k-means slice the crescents with a straight line
while DBSCAN traces them. The cost: DBSCAN's `eps` (neighborhood radius) is
fiddly to tune. Rule of thumb — blob-like data: k-means; weird shapes or
noise/outlier detection: DBSCAN.

## PCA: dimensionality reduction

Principal Component Analysis rotates the feature space to find the directions
("components") along which the data varies most, letting you keep the top few
and drop the rest. Uses: visualizing high-dimensional data in 2D, compressing
correlated features, and speeding up downstream models.

The iris data has 4 dimensions — squash it to 2:

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()
X_scaled = StandardScaler().fit_transform(iris.data)   # scale before PCA!

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
print(X_2d.shape)                          # (150, 2)
print(pca.explained_variance_ratio_)       # [0.7296 0.2285]
print(pca.explained_variance_ratio_.sum()) # 0.958
```

Two components keep ~96% of the variance — the 4-D structure was mostly 2-D
all along. Plot `X_2d` colored by species and the three species form visibly
separate groups, *even though PCA never saw the labels*:

```python
plt.figure()
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=iris.target, s=15, cmap="viridis")
plt.xlabel("PC 1"); plt.ylabel("PC 2")
plt.savefig("iris_pca.png")
```

A common pattern chains the two tools: PCA down to a handful of components,
then k-means on the reduced data — faster, and often cleaner clusters.

To choose the number of components on real data, ask for a variance target
instead of a count: `PCA(n_components=0.95)` keeps however many components
are needed to explain 95% of variance.

## Cheat sheet

| Task | Code | Notes |
|------|------|-------|
| Cluster into k groups | `KMeans(n_clusters=k, n_init="auto").fit_predict(X)` | Scale features first; IDs are arbitrary. |
| Cluster quality (per k) | `km.inertia_` (elbow), `silhouette_score(X, labels)` (peak) | Heuristics, not oracles. |
| Shape-flexible clustering + noise | `DBSCAN(eps=..., min_samples=...)` | Label `-1` = noise; tune `eps`. |
| Reduce dimensions | `PCA(n_components=2).fit_transform(X_scaled)` | Always standardize first. |
| Variance kept | `pca.explained_variance_ratio_` | `n_components=0.95` targets 95%. |
| Synthetic data | `make_blobs`, `make_moons` | Perfect for building intuition. |

## How It Actually Works

**k-means is a two-step loop repeated to convergence — Lloyd's algorithm.**
After picking `k` initial centroids (scikit-learn actually runs
"k-means++" seeding by default, which spreads initial centroids apart
probabilistically rather than placing them uniformly at random, to avoid bad
starts), each iteration does exactly two things: (1) **assign** — for every
point, compute its Euclidean distance to all `k` centroids and label it with
the nearest one; (2) **update** — move each centroid to the arithmetic mean
of all points currently assigned to it (hence "k-*means*"). These two steps
repeat until assignments stop changing (or a max iteration count is hit).
Each iteration provably never increases inertia — the sum of squared
distances from every point to its assigned centroid — which is why it
converges, though only to a *local* minimum: different random initial
centroids can converge to different final clusterings, which is exactly why
`n_init` runs the whole loop several times from different starts and keeps
the lowest-inertia result.

**Why inertia always drops as k grows, mechanically.** Inertia is
`Σ ||x_i - centroid(x_i)||²`. Adding another centroid can only ever help or
be neutral, because the algorithm could always choose to leave a new
centroid unused (never assigned) and reproduce the exact previous
partition — so inertia is a non-increasing function of k by construction,
never a proper "best k" signal on its own. The elbow you look for is where
adding centroids stops buying much reduction because the data's true
cluster structure has already been captured; beyond that, more centroids
just start splitting genuine, single clusters into halves.

**Silhouette score is a per-point ratio of two distances, then averaged.**
For each point `i`, compute `a(i)` = mean distance to other points in `i`'s
own cluster, and `b(i)` = mean distance to points in the nearest *other*
cluster. The silhouette for that point is
`(b(i) - a(i)) / max(a(i), b(i))`. A point tightly packed with its own
cluster and far from every other cluster gets a value near +1; a point
about equally close to two clusters gets a value near 0; a clearly
misassigned point gets a negative value. Averaging across all points gives
the number printed above — which is why, unlike inertia, it can go down
again once k overshoots the true structure: points get pulled from a
genuine cluster into an artificial split, shrinking `b(i)-a(i)`.

**DBSCAN grows clusters by chaining dense neighborhoods, with no centroid
concept at all.** For each point, DBSCAN counts how many other points lie
within radius `eps`. A point with at least `min_samples` neighbors within
`eps` is a "core point"; DBSCAN then unions together every core point that
lies in another core point's `eps`-neighborhood, forming a single cluster
out of chains of overlapping dense regions — which is exactly how it
traces a curved crescent shape that k-means' "assign to nearest centroid"
rule cannot: k-means partitions space into flat-boundary (Voronoi) regions
around centroids, and a crescent moon shape straddles that boundary no
matter where the centroid sits. Points that aren't core points and aren't
within `eps` of one either get labeled `-1` (noise) — a mechanism k-means
has no equivalent for, since every point is forced into some cluster.

**PCA finds rotation axes via eigendecomposition of the covariance
matrix.** After standardizing, PCA computes the `n_features × n_features`
covariance matrix `C` (how much each pair of features varies together),
then finds its eigenvectors and eigenvalues (equivalently done via SVD on
the data matrix directly, which is what scikit-learn uses for numerical
stability). Each eigenvector is a "principal component" — a direction in
the original 4-D feature space — and its corresponding eigenvalue measures
how much variance the data has along that direction. Sorting components by
eigenvalue and keeping the top 2 is precisely what `n_components=2` does:
project every data point onto those 2 directions (a matrix multiply by the
top-2 eigenvectors), discarding the directions where the data barely
varies. `explained_variance_ratio_` is just each kept eigenvalue divided by
the sum of *all* eigenvalues — literally "what fraction of total spread
does this axis account for" — which is why the two ratios summing to 0.958
means almost no spread was thrown away by dropping to 2 dimensions.

## Exercise

Generate blobs with `make_blobs(n_samples=600, centers=5, cluster_std=1.5,
random_state=7)` — note the overlap at std 1.5. (1) Run the elbow and
silhouette loops for k = 2..9: do they still agree on 5? (2) Standardize the
data, PCA it (2 components — how much variance survives?), and re-run
k-means on the reduced data. (3) Compare the silhouette score before and
after PCA and write one sentence on whether reduction helped or hurt here,
and why it might.
