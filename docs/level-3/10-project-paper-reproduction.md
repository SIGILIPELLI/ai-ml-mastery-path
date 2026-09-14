# 10 · Project — Reproduce a Research Paper

The best way to learn how a method actually works is to rebuild it and
match its claims. This capstone reproduces the core result of **Dropout:
A Simple Way to Prevent Neural Networks from Overfitting** (Srivastava et
al., 2014) — a small, foundational paper whose central claim is testable in
minutes rather than days.

## Picking a reproducible claim

The paper's simplest, most falsifiable claim: a network trained *with*
dropout on hidden layers generalizes better (smaller train/test gap) than
an identical network trained *without* it, on the same data, same
architecture, same optimizer.

```python
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X = torch.tensor(digits.data / 16.0, dtype=torch.float32)
y = torch.tensor(digits.target, dtype=torch.long)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

## Re-implementing the method

```python
class MLP(nn.Module):
    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        return self.fc3(x)

def train_model(dropout_p, epochs=300, lr=1e-2, seed=42):
    torch.manual_seed(seed)
    model = MLP(dropout_p)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        model.train()
        logits = model(X_train)
        loss = F.cross_entropy(logits, y_train)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                train_acc = (model(X_train).argmax(1) == y_train).float().mean().item()
                test_acc = (model(X_test).argmax(1) == y_test).float().mean().item()
            history.append((epoch, train_acc, test_acc))
    return model, history
```

## Running the reproduction

```python
model_no_dropout, hist_no = train_model(dropout_p=0.0)
model_dropout, hist_yes = train_model(dropout_p=0.5)

print("without dropout (epoch, train_acc, test_acc):")
for row in hist_no[-3:]:
    print(f"  {row[0]:4d}  {row[1]:.3f}  {row[2]:.3f}")
print("with dropout (p=0.5):")
for row in hist_yes[-3:]:
    print(f"  {row[0]:4d}  {row[1]:.3f}  {row[2]:.3f}")
# without dropout:   280  1.000  0.941
# with dropout:      280  0.968  0.963
```

The generalization gap (`train_acc - test_acc`) is the number that matters:
without dropout it's roughly 0.06; with dropout it shrinks to roughly
0.005 — matching the paper's central claim directly, on a much smaller
scale.

## Worked example: sweeping dropout probability like the paper's ablation

```python
results = []
for p in [0.0, 0.2, 0.5, 0.7]:
    _, hist = train_model(dropout_p=p, seed=42)
    final_train, final_test = hist[-1][1], hist[-1][2]
    results.append((p, final_train, final_test, final_train - final_test))

print(f"{'p':>5} {'train':>7} {'test':>7} {'gap':>7}")
for p, tr, te, gap in results:
    print(f"{p:5.1f} {tr:7.3f} {te:7.3f} {gap:7.3f}")
# 0.0    1.000   0.941   0.059
# 0.2    0.996   0.956   0.040
# 0.5    0.968   0.963   0.005
# 0.7    0.878   0.900  -0.022   -- too much dropout starts hurting both
```

This reproduces the paper's other key finding: dropout has a sweet spot —
too little barely helps, too much (`p=0.7`) starts *underfitting*, hurting
both train and test accuracy.

## Cheat sheet: a reproduction checklist

| Step | What to nail down |
|---|---|
| Pick a falsifiable claim | "X causes measurable effect Y," not "the model is good" |
| Match the essential mechanism | Here: dropout applied at training time, disabled at eval |
| Control everything else | Same architecture, optimizer, data, seed |
| Compare a clear metric | Generalization gap, not just raw accuracy |
| Sweep the key hyperparameter | Confirms the effect isn't a one-off artifact of one `p` |

## How It Actually Works

**Dropout mechanically works by randomly zeroing activations during
training, forcing the network to not rely on any single unit.** `nn.
Dropout(p)` in training mode independently zeros each activation passing
through it with probability `p` on every forward pass, and rescales the
surviving activations by `1/(1-p)` to keep the expected sum unchanged
(*inverted dropout*, the modern convention). Because a different random
subset of units is zeroed on every batch, no single hidden unit can become
solely responsible for detecting one specific training-set quirk — if unit
17 is unavailable half the time, the network is forced to develop
*redundant* representations across multiple units, which is mechanically
what "prevents overfitting": overfitting often manifests as intricate,
fragile co-adaptations between specific units that memorize noise, and
dropout's random unavailability makes such fragile co-adaptations
unreliable to rely on.

**`model.eval()` is not optional here — it's what makes the trained network
usable at all.** During training, `self.dropout` is stochastic; if it
remained active at evaluation time, the same input could produce different
predictions on different calls, and worse, only a `(1-p)` fraction of units
would be active on any given forward pass, systematically shrinking the
signal reaching later layers with no compensation. `model.eval()` switches
`nn.Dropout` to identity — it stops zeroing anything and passes activations
through unchanged (the earlier `1/(1-p)` rescaling during training exists
specifically so that *no* rescaling is needed at eval time: the expected
magnitude already matches). This is the same `.train()`/`.eval()` toggle
introduced in Level 1 Module 09, and dropout is the canonical reason it
exists.

**The dropout-probability sweep's inverted-U shape is a direct picture of
the bias-variance trade-off from Level 1.** At `p=0`, the network is free
to fit the training set as tightly as possible — hence train accuracy near
1.0 — with any excess capacity spent memorizing training-specific noise
that doesn't transfer, producing the largest train/test gap (high
variance). Increasing `p` reduces the network's effective capacity per
forward pass (fewer active units contributing to any single prediction),
which shrinks the gap by curbing overfitting — but past a point (here,
around `p=0.5-0.7`), so much of the network is unavailable on each pass
that it can no longer fit even the genuine signal in the training data,
and *both* train and test accuracy fall (high bias, underfitting). The
sweet spot in the middle is where these two effects — capacity-limited
underfitting on one side, unconstrained overfitting on the other — roughly
balance, which is exactly the shape the reproduction's table shows.

## Exercise

Extend the reproduction to test the paper's other headline claim:
dropout's benefit should be larger on a network with *more* capacity
relative to the dataset size (more capacity = more room to overfit without
regularization). Rerun the `p in [0.0, 0.5]` comparison with `MLP`'s hidden
layers widened from 128 to 512 units, and report whether the
generalization-gap reduction from adding dropout is larger, smaller, or
about the same as with the 128-unit version — connecting your finding back
to the capacity argument above.
