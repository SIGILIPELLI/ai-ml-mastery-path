# 09 · Advanced Computer Vision

Module 05 (Level 2) classified whole images. Real vision tasks often need
more: *where* is the object (detection), or *which pixels* belong to it
(segmentation). This module covers both architectures at the level of how
they actually work, using pretrained models via `torchvision`.

## Object detection: boxes plus classes

```python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights)
model.eval()

preprocess = weights.transforms()
image = read_image("street_scene.jpg")   # (3, H, W) uint8 tensor
batch = [preprocess(image)]

with torch.no_grad():
    predictions = model(batch)[0]

keep = predictions["scores"] > 0.7
boxes = predictions["boxes"][keep]
labels = predictions["labels"][keep]
scores = predictions["scores"][keep]
categories = weights.meta["categories"]
for box, label, score in zip(boxes, labels, scores):
    print(f"{categories[label]:12s} {score:.2f}  box={box.tolist()}")
# person       0.98  box=[102.3, 45.1, 210.7, 380.2]
# car          0.93  box=[400.5, 150.0, 620.1, 310.4]
```

Unlike classification, the output isn't one label per image — it's a
variable-length list of `(box, label, score)` triples, one per detected
object.

## Semantic segmentation: a label per pixel

```python
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

seg_weights = DeepLabV3_ResNet50_Weights.DEFAULT
seg_model = deeplabv3_resnet50(weights=seg_weights)
seg_model.eval()

seg_preprocess = seg_weights.transforms()
batch = seg_preprocess(image).unsqueeze(0)

with torch.no_grad():
    output = seg_model(batch)["out"]   # (1, 21, H, W) -- 21 classes, one score map each
pred_mask = output.argmax(1).squeeze(0)   # (H, W) -- the winning class index per pixel
print(pred_mask.shape, pred_mask.unique())   # torch.Size([520, 780]) tensor([0, 7, 15])
```

`argmax(1)` collapses the 21-channel score map to a single integer per
pixel — the class with the highest score at that exact spatial location.

## Worked example: computing IoU for a detected box

```python
def iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0

ground_truth_box = [100.0, 44.0, 215.0, 385.0]
predicted_box = boxes[0].tolist()
print(f"IoU: {iou(ground_truth_box, predicted_box):.3f}")   # e.g. 0.91
```

**Intersection-over-Union (IoU)** is the standard metric for "how good is
this box": a common threshold like `IoU >= 0.5` counts a detection as a
true positive when computing detection precision/recall.

## Cheat sheet

| Task | Output shape | Model family |
|---|---|---|
| Classification | 1 label per image | ResNet, ViT |
| Detection | N boxes + labels + scores per image | Faster R-CNN, YOLO |
| Segmentation | 1 label per pixel | DeepLabV3, U-Net |
| Box quality metric | IoU (0 to 1) | — |

## How It Actually Works

**Faster R-CNN's two-stage design mirrors its name: propose regions, then
classify them.** The first stage (a Region Proposal Network) slides a small
network over the CNN backbone's feature map and, at each spatial location,
scores a set of predefined "anchor boxes" of different sizes/aspect ratios
for "does this look like it contains *any* object" — producing a few
hundred candidate regions likely to contain something, out of what would
otherwise be an intractable number of possible boxes across the image. The
second stage takes each proposed region, crops and resizes its
corresponding features (ROI pooling/align), and runs it through a small
classifier head that assigns an actual class label and a confidence score,
plus a refined box regression. This is mechanically why detection output
is variable-length and includes a `scores` tensor — the first stage
generates a variable number of candidates, and the confidence threshold
(`scores > 0.7`) is a downstream filter applied to the second stage's
independent classification confidence for each surviving candidate.

**Semantic segmentation's per-pixel output comes from an encoder that
downsamples spatial resolution while gaining semantic context, followed by
a decoder that upsamples it back.** Like the CNN in Level 2 Module 05,
DeepLabV3's backbone repeatedly halves spatial resolution through strided
convolutions/pooling while increasing channel depth, trading spatial
precision for larger receptive fields and richer per-region features (the
same receptive-field-growth mechanism from that module, pushed much
further). Because the final classification needs a prediction at *every
original pixel*, the decoder then upsamples this coarse, semantically rich
feature map back to the input resolution (bilinear interpolation plus
learned refinement), producing the `(21, H, W)` score tensor — 21 separate
per-pixel score maps, one per class, exactly analogous to how a
classification head outputs one logit per class, just computed
independently at every spatial position rather than once for the whole
image. `argmax(1)` then does per-pixel what `argmax(dim=1)` did per-image
in Level 1's classification modules: pick the highest-scoring class at each
location.

**IoU is a purely geometric ratio, and its threshold-based use for
"correct or not" comes from balancing two failure modes.** The formula
`intersection_area / union_area` ranges from 0 (no overlap) to 1 (identical
boxes) by construction — it penalizes both under-coverage (predicted box
too small or misplaced, shrinking the numerator) and over-prediction
(predicted box far larger than needed, inflating the denominator) in a
single number, unlike measuring only overlap area or only size difference.
A threshold like 0.5 is a calibration choice: too strict (e.g. 0.9) would
reject boxes that are visually "close enough" as false negatives even when
the object was genuinely detected in the right place, while too loose
(e.g. 0.1) would accept boxes overlapping only marginally as correct
detections — the standard detection benchmarks (COCO, Pascal VOC) settled
on thresholds like 0.5 (and report averages across several thresholds,
"mAP@[0.5:0.95]") as a practical compromise between these two error modes.

## Exercise

Using the `iou` function above, compute IoU between the ground-truth box
and *every* box in `predictions["boxes"]` (before the `scores > 0.7`
filter), then sort by IoU descending. Report whether the single highest-
confidence detection (as filtered in the main example) is also the single
highest-IoU box against ground truth, or whether a lower-confidence box
happens to have better spatial overlap — a mismatch that's common in
practice and explains why detection pipelines often use both confidence
and IoU-based non-maximum suppression together.
