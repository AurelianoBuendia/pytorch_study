# Independent YOLO11 work

This directory is a work in progress, not an independent YOLO11 model. It does
not yet provide inference, checkpoint loading, tracking, ONNX export or
training. In particular, it is not currently a replacement for Ultralytics.

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
