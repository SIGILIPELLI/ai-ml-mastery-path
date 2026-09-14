# 03 · Feature Stores & Data Engineering for ML

Module 01 identified training/serving skew as a top production failure
mode, caused by duplicated feature logic. A **feature store** is the
infrastructure pattern that eliminates the duplication directly: one
definition, computed once, served from two paths (batch for training,
low-latency for online inference).

## The offline/online split

```python
import pandas as pd

# Offline store: historical, point-in-time correct, used for training
offline_features = pd.DataFrame({
    "user_id": [1, 1, 2, 2],
    "event_time": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-15", "2024-02-15"]),
    "avg_purchase_30d": [42.1, 55.3, 12.0, 18.7],
})

# Online store: current snapshot only, used for low-latency serving
online_features = {
    1: {"avg_purchase_30d": 55.3, "last_updated": "2024-02-01"},
    2: {"avg_purchase_30d": 18.7, "last_updated": "2024-02-15"},
}
```

The offline store keeps full history (needed to reconstruct "what did this
feature look like at the time of each historical training example"); the
online store keeps only the current value per entity (all that's needed
for a live prediction request).

## Point-in-time correctness: the subtle bug feature stores fix

```python
def get_features_naive(user_id, offline_features):
    """WRONG for training: grabs the LATEST value, ignoring when the label happened."""
    return offline_features[offline_features.user_id == user_id].iloc[-1]

def get_features_point_in_time(user_id, as_of_time, offline_features):
    """CORRECT: only uses feature values that existed BEFORE the label's timestamp."""
    valid = offline_features[
        (offline_features.user_id == user_id) & (offline_features.event_time <= as_of_time)
    ]
    return valid.iloc[-1] if len(valid) else None

label_time = pd.Timestamp("2024-01-20")   # a churn label recorded on this date
wrong = get_features_naive(1, offline_features)
correct = get_features_point_in_time(1, label_time, offline_features)
print("naive (leaks the future):", wrong["avg_purchase_30d"])       # 55.3 -- from Feb, after the label!
print("point-in-time correct:  ", correct["avg_purchase_30d"])      # 42.1 -- what was actually known on Jan 20
```

Using `get_features_naive` for training data assembly is a form of label
leakage identical in spirit to Level 2 Module 03's rolling-window bug — the
model is trained on information that didn't exist yet at the moment the
label was generated.

## A minimal feature store interface

```python
class SimpleFeatureStore:
    def __init__(self):
        self.offline = pd.DataFrame(columns=["user_id", "event_time", "feature_name", "value"])
        self.online = {}

    def write(self, user_id, event_time, feature_name, value):
        row = {"user_id": user_id, "event_time": event_time, "feature_name": feature_name, "value": value}
        self.offline = pd.concat([self.offline, pd.DataFrame([row])], ignore_index=True)
        self.online.setdefault(user_id, {})[feature_name] = value   # online always holds the latest

    def get_training_features(self, user_id, feature_name, as_of_time):
        subset = self.offline[
            (self.offline.user_id == user_id)
            & (self.offline.feature_name == feature_name)
            & (self.offline.event_time <= as_of_time)
        ]
        return subset.iloc[-1]["value"] if len(subset) else None

    def get_online_features(self, user_id, feature_name):
        return self.online.get(user_id, {}).get(feature_name)

store = SimpleFeatureStore()
store.write(1, pd.Timestamp("2024-01-01"), "avg_purchase_30d", 42.1)
store.write(1, pd.Timestamp("2024-02-01"), "avg_purchase_30d", 55.3)

print(store.get_training_features(1, "avg_purchase_30d", pd.Timestamp("2024-01-20")))  # 42.1
print(store.get_online_features(1, "avg_purchase_30d"))                                 # 55.3 (latest)
```

The same `write()` call populates both stores; training assembly reads
`get_training_features` (point-in-time correct), serving reads
`get_online_features` (always latest) — one write path, two read paths
matched to two different correctness requirements.

## Cheat sheet

| Concept | Purpose |
|---|---|
| Offline store | Full history, point-in-time queries, training data assembly |
| Online store | Current snapshot only, low-latency lookup at inference |
| Point-in-time join | Only use feature values known *before* the label's timestamp |
| Single write path | Eliminates the duplicate-logic root cause of training/serving skew |

## How It Actually Works

**Point-in-time correctness is mechanically a filtered join on a
timestamp, and its absence is a specific, common form of label leakage.**
`get_features_point_in_time` filters `offline_features` to rows where
`event_time <= as_of_time` before taking the most recent one — restricting
the join to feature values that genuinely existed at (or before) the
moment the label was recorded. `get_features_naive`'s bug is subtler than
an off-by-one: it takes the *globally* latest row per user regardless of
when the label happened, meaning a feature value computed from February
purchase behavior can end up attached to a January churn label — feature
information about the user's behavior *after* the event being predicted.
This is mechanically identical to Level 2 Module 03's unshifted rolling
window (using `sales[t]` to predict `sales[t]`), just expressed as a join
condition instead of a `.shift()` call — the training pipeline discovers a
predictive relationship that cannot exist at serving time, since the
future user behavior a live request needs to predict-hasn't-happened-yet
by definition.

**A shared write path with two differently-filtered read paths is what
structurally prevents skew, as opposed to merely detecting it.**
`SimpleFeatureStore.write()` is the single place feature values enter the
system, appending to `self.offline` (preserving full history) and updating
`self.online` (overwriting with the latest value) in the same call. Both
`get_training_features` and `get_online_features` read from data produced
by that one write path — they differ only in *which* rows they select
(point-in-time filtered vs. latest-only), not in how the underlying feature
value was computed. This is the direct generalization of Level 4 Module
01's "one shared feature function" principle: instead of relying on two
independently-maintained feature computations staying in sync by
discipline, there's exactly one computation and write path, and the
offline/online distinction is purely a query-time filter applied
afterward.

**The online store's "latest value only" design is a deliberate trade
against storage and lookup cost, justified by what serving actually
needs.** A live prediction request needs the feature value *as of right
now* — there's no meaningful sense in which a production API call needs to
know what a user's `avg_purchase_30d` was three months ago. Storing only
the current value per `(user_id, feature_name)` pair, overwritten on each
write (as `self.online.setdefault(...)[feature_name] = value` does), keeps
online lookups O(1) and storage proportional to the number of *entities*
rather than the number of *historical events* — a deliberately narrower
data structure than the offline store's full append-only history, chosen
specifically because it matches the online path's actual access pattern
(read latest, never read history) rather than trying to serve both access
patterns from one general-purpose structure.

## Exercise

Add a third write for user 1 (`event_time="2024-03-01"`,
`avg_purchase_30d=60.2`) to `store`. Then simulate assembling a *training
set* of three labeled examples for user 1 with label timestamps
`2024-01-15`, `2024-02-10`, and `2024-03-15`, using `get_training_features`
for each. Report the three feature values retrieved and confirm none of
them uses a value that postdates its label's timestamp — the concrete
guarantee point-in-time correctness is meant to provide.
