# Independent YOLO11 work

This directory is a work in progress, not a drop-in replacement for
Ultralytics. It loads the bundled `models/yolo11n.pt` detection checkpoint,
runs image detection, and exports raw predictions to ONNX without importing
Ultralytics. Tracking and training are not implemented. Existing scripts such
as `tracking.py` still depend on Ultralytics.

## Image detection and ONNX export

Install the runtime dependencies with `pip install -r yolo11/requirements.txt`.
Inference accepts a single image path, a BGR `uint8` NumPy array, or a list of
these. `stream=True` returns an iterator instead of a list. For example:

```python
from yolo11 import YOLO

model = YOLO("models/yolo11n.pt")
results = model.predict("images/couple-3687274_960_720.jpg", imgsz=640)
for box in results[0].boxes:
    print(box.xyxy, box.conf, box.cls)

onnx_path = model.export(format="onnx", imgsz=640, dynamic=True)
```

The loader supports the serialized YOLO11 detection model format represented
by the bundled `yolo11n.pt`; segmentation, pose, and other model formats are
not supported. It uses a restricted unpickler but checkpoints are complex
binary inputs: load only files from a trusted source. Unsupported checkpoint
classes are rejected. No checkpoint is downloaded automatically.

The supported image API is intentionally limited: videos, URLs, tensors, PIL
images, plotting, and most Ultralytics options are not implemented. Image sizes
must be positive multiples of 32. Preprocessing uses fixed-size letterboxing,
so results may differ from Ultralytics' default rectangular batching. The
exported ONNX input is a normalized RGB tensor in NCHW order; its output is
`[batch, 4 + classes, anchors]` with pixel-space center-x, center-y, width,
height and class probabilities. It does not include image preprocessing or
non-maximum suppression. Dynamic exports support different batch sizes and
spatial sizes divisible by 32.

## COCO detection data preparation

The converter copies train and validation images into a new dataset directory
and writes normalized detection labels and `data.yaml`:

```sh
python -m yolo11.coco \
  --train-annotations /data/annotations/instances_train.json \
  --train-images /data/train \
  --val-annotations /data/annotations/instances_val.json \
  --val-images /data/val \
  --output /data/yolo-dataset
```

Train and validation category IDs and names must match. Categories are mapped
to contiguous zero-based indices sorted by COCO category ID; the mapping is
also saved in `category_ids.json`. Bounding boxes are clipped to image bounds;
boxes entirely outside the image are rejected. Images without annotations are
retained. Crowd annotations are omitted with a warning: the resulting labels
cannot represent COCO's ignore regions, so crowd objects may be treated as
background. Segmentation and keypoints are not converted.

The output directory must not already exist. The `path` in `data.yaml` is
absolute; update it if the dataset is moved. Conversion requires Python 3.9+
and has no third-party dependencies. If a filesystem error occurs while
installing the prepared output, the newly created output directory may be
incomplete and must be removed before retrying.
