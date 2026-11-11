# Independent YOLO11 work

This directory is a work in progress, not a drop-in replacement for
Ultralytics. It loads the bundled `models/yolo11n.pt` detection checkpoint,
runs image detection, and exports raw predictions to ONNX without importing
Ultralytics. ByteTrack supports image sequences and local videos. Training is
not implemented.

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

The prediction API is intentionally limited: videos, URLs, tensors, PIL
images, plotting, and most Ultralytics options are not implemented. Image sizes
must be positive multiples of 32. Preprocessing uses fixed-size letterboxing,
so results may differ from Ultralytics' default rectangular batching. The
exported ONNX input is a normalized RGB tensor in NCHW order; its output is
`[batch, 4 + classes, anchors]` with pixel-space center-x, center-y, width,
height and class probabilities. It does not include image preprocessing or
non-maximum suppression. Dynamic exports support different batch sizes and
spatial sizes divisible by 32.

## ByteTrack tracking

Track a local video, writing an annotated MP4:

```sh
python -m yolo11.track_video input.mp4 output.mp4 \
  --weights models/yolo11n.pt --tracker bytetrack.yaml
```

Or consume results in Python:

```python
from yolo11 import YOLO

model = YOLO("models/yolo11n.pt")
for result in model.track("input.mp4", tracker="bytetrack.yaml", stream=True):
    for box in result.boxes:
        print(int(box.id[0]), int(box.cls[0]), float(box.conf[0]), box.xyxy[0])
```

`track` accepts a video path, image path, BGR NumPy frame, or list of image
paths/frames. Streaming video is decoded one frame at a time; close the
generator if iteration stops early. Results contain confirmed tracks only,
including an empty `(0, 7)` boxes tensor on frames without confirmed tracks.
Each tracked box is `[x1, y1, x2, y2, track_id, confidence, class]`.

The bundled `bytetrack.yaml` is used when that name is given and no matching
file exists in the working directory. Supply a path to use a custom YAML
configuration. The default detection confidence is the smaller of 0.1 and
`track_low_thresh` so that low-confidence detections reach the tracker; an
explicit `conf` can override it. The tracker performs high/low score
association, Kalman prediction, lost-track reactivation and buffered expiry.
`track_buffer` is measured in frames at 30 FPS and is scaled to the input
video's frame rate. Pass `frame_rate` when tracking individual frames or when
a video's FPS metadata is invalid.

Each call starts a new sequence by default. For successive individual frames,
use `persist=True` on every call; call `model.reset_tracker()` when the
sequence ends. Persistent calls must use the same tracker configuration and
frame rate. A model instance cannot process concurrent tracking streams.
Video support is limited to local files with the extensions listed in
`yolo11/api.py`; camera indices and URLs are not supported.

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
