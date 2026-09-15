---
description: "CNNs for Image Classification — A plain MLP (Module 09) treats an image as a flat list of pixels, throwing away the fact that nearby pixels are related.…"
---

# 05 · CNNs for Image Classification

A plain MLP (Module 09) treats an image as a flat list of pixels, throwing
away the fact that nearby pixels are related. **Convolutional neural
networks (CNNs)** exploit that spatial structure directly, and are the
backbone of essentially all modern computer vision. This module builds and
trains a small CNN on `FashionMNIST` with PyTorch.

## Loading image data

```python
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor()])   # scales pixels to [0, 1]
train_ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_ds = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

image, label = train_ds[0]
print(image.shape, label)   # torch.Size([1, 28, 28]) 9  -- 1 channel (grayscale), 28x28
```

Image tensors are `(channels, height, width)` — one channel here (grayscale);
color photos would have 3 (RGB).

## A small CNN

```python
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # 1 -> 16 feature maps
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 16 -> 32 feature maps
        self.pool = nn.MaxPool2d(2)                                # halves height & width
        self.fc = nn.Linear(32 * 7 * 7, 10)                        # 10 classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))   # 28x28 -> 14x14
        x = self.pool(torch.relu(self.conv2(x)))   # 14x14 -> 7x7
        x = x.flatten(1)                            # (batch, 32*7*7)
        return self.fc(x)                           # raw logits, 10 classes

torch.manual_seed(42)
model = SmallCNN()
print(sum(p.numel() for p in model.parameters()), "parameters")   # 21,258
```

## Training loop

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(3):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    print(f"epoch {epoch}  avg loss {total_loss / len(train_ds):.4f}")
# epoch 0  avg loss 0.5123
# epoch 1  avg loss 0.3324
# epoch 2  avg loss 0.2887

model.eval()
correct = 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        correct += (model(xb).argmax(1) == yb).sum().item()
print(f"test accuracy: {correct / len(test_ds):.3f}")   # ~0.90
```

## Worked example: visualizing a filter's receptive field

```python
sample_x, sample_y = test_ds[0]
with torch.no_grad():
    feat_maps = torch.relu(model.conv1(sample_x.unsqueeze(0)))
print(feat_maps.shape)   # torch.Size([1, 16, 28, 28]) -- 16 different edge/texture detectors
print(feat_maps[0, 0].mean().item(), feat_maps[0, 5].mean().item())  # different filters activate differently
```

Each of the 16 output channels of `conv1` is a different learned filter,
each producing its own activation map over the image — one might respond to
vertical edges, another to horizontal ones, purely from data, never
hand-specified.

## Cheat sheet

| Layer | Purpose |
|---|---|
| `nn.Conv2d(in_c, out_c, kernel_size, padding)` | Learn local spatial filters |
| `nn.MaxPool2d(k)` | Downsample, add translation tolerance |
| `.flatten(1)` | Turn feature maps into a vector before `nn.Linear` |
| `padding=1` with `kernel_size=3` | Keeps spatial size unchanged after the conv |

## How It Actually Works

**A convolution is a small weight matrix slid across the image, reusing the
same weights everywhere.** `Conv2d(1, 16, kernel_size=3)` learns 16 separate
`3×3` filters (plus a bias each). For a given filter, its output at
position `(i, j)` is `Σ` over the `3×3` neighborhood of the input centered
at `(i, j)`, multiplying each of the 9 input pixels by the filter's
corresponding weight and summing — exactly a dot product between the filter
and that local patch. Sliding the *same* 9 weights across every position in
the image (rather than learning separate weights per pixel, as a fully
connected layer would) is what gives convolutions two properties MLPs lack:
drastically fewer parameters (9 weights detect an edge anywhere in the
image, not just in one location), and **translation equivariance** — shift
the input pattern and the output feature map shifts by the same amount,
because it's the literal same arithmetic operation applied at a different
offset.

**`padding=1` and stride mechanically determine output size, and pooling
mechanically halves it.** With `kernel_size=3` and `padding=1` (one pixel of
zeros added around the border), a `28×28` input produces a `28×28` output
feature map — the padding exactly compensates for the 1-pixel shrinkage a
`3×3` kernel would otherwise cause at each edge. `MaxPool2d(2)` then takes
non-overlapping `2×2` blocks and keeps only the maximum value in each,
mechanically producing a `14×14` output from a `28×28` input — a factor-of-4
reduction in the number of values, and a factor-of-2 reduction in each
spatial dimension. This is why the flattened size before `fc` is exactly
`32 * 7 * 7`: two `MaxPool2d(2)` layers take `28 → 14 → 7`, and the second
conv produces 32 channels at that resolution — the fully-connected layer's
input size is a direct arithmetic consequence of the architecture above it,
not a free parameter.

**Why stacking conv layers builds increasingly large receptive fields.** A
single `3×3` filter in `conv1` only ever looks at a 3-pixel-wide
neighborhood of the raw input — it cannot "see" a whole shoe or shirt shape.
But `conv2`'s `3×3` filter operates on `conv1`'s *output* feature map, where
each position already summarizes a 3-pixel neighborhood of the original
image; after the intervening pooling (which also aggregates a 2×2 block),
one unit in `conv2`'s output is influenced by roughly a `8×8` patch of the
original 28×28 image. This growing **receptive field** with depth is the
literal mechanism by which CNNs build from local edge/texture detectors in
early layers to shape/part detectors in later layers — not a metaphor, but
a direct consequence of composing local, sliding-window operations.

## Exercise

Add a third conv block (`Conv2d(32, 64, 3, padding=1)` + `ReLU` + a third
`MaxPool2d(2)`) to `SmallCNN`, updating the flattened size for `fc`
accordingly (work out the new spatial dimensions using the halving rule
above before running it). Train for 3 epochs and compare test accuracy and
parameter count against the two-block version. Report whether the deeper
network is worth the added parameters on this dataset.
