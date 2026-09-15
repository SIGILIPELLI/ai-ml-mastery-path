---
description: "Transfer Learning Basics — Training a CNN from scratch (Module 05) needs a lot of data to learn good low-level filters. Transfer learning reuses a network…"
---

# 06 · Transfer Learning Basics

Training a CNN from scratch (Module 05) needs a lot of data to learn good
low-level filters. **Transfer learning** reuses a network already trained
on a huge dataset (ImageNet, ~1.2M images) and adapts it to a new, smaller
task — usually with far less data and training time.

## Loading a pretrained network

```python
import torch
from torch import nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
print(model.fc)   # Linear(in_features=512, out_features=1000)  -- ImageNet's 1000 classes
```

`ResNet18` was trained to classify 1000 ImageNet categories; its
convolutional layers learned general-purpose visual features (edges,
textures, shapes, parts) that transfer well to almost any image task.

## Freezing the backbone, replacing the head

```python
for param in model.parameters():
    param.requires_grad = False              # freeze every existing weight

num_classes = 3   # e.g. cats / dogs / other
model.fc = nn.Linear(model.fc.in_features, num_classes)   # fresh, trainable head

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"{trainable:,} trainable / {total:,} total")   # 1,539 trainable / 11,178,051 total
```

Only the new final layer (`512 * 3 + 3 = 1539` parameters) trains; the
other ~11.2M parameters stay fixed at their ImageNet-trained values.

## Preprocessing to match pretraining

```python
preprocess = weights.transforms()   # exact resize/crop/normalize ResNet18 was trained with
print(preprocess)
# ImageClassification(
#     crop_size=[224], resize_size=[256], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
# )
```

Using the *same* normalization statistics the network was pretrained with is
not optional — feeding it images scaled differently shifts every activation
away from the ranges its filters were tuned for, silently destroying most of
the transferred benefit.

## Training just the head

```python
train_ds = datasets.ImageFolder("data/pets/train", transform=preprocess)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)   # only the head's params
loss_fn = nn.CrossEntropyLoss()

model.train()
for epoch in range(5):
    total_loss = 0.0
    for xb, yb in train_loader:
        logits = model(xb)
        loss = loss_fn(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    print(f"epoch {epoch}  loss {total_loss / len(train_ds):.4f}")
```

Even though `loss.backward()` computes gradients through the *entire*
network (it has to, to reach the head from the loss), `optimizer.step()`
only updates `model.fc.parameters()` — the frozen layers' `requires_grad =
False` means autograd doesn't even bother computing their gradients.

## Fine-tuning: unfreezing later layers

Once the new head has stabilized, a common second stage unfreezes the last
few blocks with a much smaller learning rate, adapting high-level features
to the new domain without destroying the low-level ones:

```python
for name, param in model.named_parameters():
    if name.startswith("layer4"):        # last residual block
        param.requires_grad = True

optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad], lr=1e-5   # 100x smaller than head-only lr
)
```

## Cheat sheet

| Stage | What trains | Typical `lr` |
|---|---|---|
| Feature extraction | Only the new head | `1e-3` |
| Fine-tuning | Head + last few blocks | `1e-5`–`1e-4` |
| Full fine-tuning | Everything | `1e-5` or lower, only with enough data |

## How It Actually Works

**Why ImageNet-pretrained filters transfer at all.** A CNN's early
convolutional layers (Module 05's mechanism: small filters detecting local
patterns) learn to respond to edges, corners, color blobs, and textures —
patterns that recur in virtually every natural image regardless of what the
final classification task is. Layers deeper in the network combine these
into progressively more task-specific shapes (wheels, fur patterns, eyes).
Because the *early* representations are close to universal, reusing them
for a new task only requires relearning the *mapping from high-level
features to your specific classes* — which is exactly what a fresh
`nn.Linear(512, num_classes)` head does, taking the pretrained network's
512-dimensional summary of "what's in this image" and learning a new linear
combination for your 3 classes instead of ImageNet's 1000.

**`requires_grad = False` mechanically prunes the autograd graph, not just
the optimizer step.** Recall from Module 09 that `backward()` walks a
computation graph, applying the chain rule at each node to compute
gradients. When a tensor has `requires_grad = False`, PyTorch does not
attach it to that graph's differentiable leaves — during the backward pass,
computation still flows *through* frozen layers (gradients must pass back
through them to reach earlier layers, if any were trainable), but no
`.grad` is accumulated on the frozen parameters themselves, and the
optimizer, restricted to `model.fc.parameters()`, has nothing to update for
them regardless. This is why freezing saves memory and compute for the
backward pass on the frozen weights specifically, even though the forward
pass still runs the full 11.2M-parameter network.

**Why fine-tuning uses a much smaller learning rate than head-only
training.** The pretrained weights already sit at a good optimum for
general visual features — large gradient updates (the `lr=1e-3` fine for a
*freshly initialized* linear head) would take large steps away from that
optimum, effectively erasing the useful pretraining before the new head has
had a chance to guide the network toward the new task's specifics. A
`lr=1e-5` update moves those weights only slightly per step — enough, over
many steps, to specialize `layer4`'s high-level features toward the new
domain's specific shapes, but not enough to catastrophically overwrite what
ImageNet training spent millions of images learning. This is the same
gradient-descent update-size argument from Module 09 (`L(w - lr·grad) ≈
L(w) - lr·grad²`), applied deliberately asymmetrically across a network
whose different layers start at very different distances from where they
need to end up.

## Exercise

Using the two-stage recipe above (freeze-and-train-head, then unfreeze
`layer4` with a smaller `lr`), compare final validation accuracy against
training only the head for all epochs (never unfreezing `layer4`). Then try
initializing `resnet18` with `weights=None` (random weights, no
pretraining) and train the whole network on the same small dataset —
report how much worse it performs and connect the gap back to the
"universal early features" argument above.
