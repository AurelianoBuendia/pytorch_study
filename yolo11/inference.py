"""Image preprocessing and detection postprocessing."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms, nms


class Boxes:
    def __init__(self, data, orig_shape):
        self.data = data
        self.orig_shape = orig_shape

    @property
    def xyxy(self):
        return self.data[:, :4]

    @property
    def conf(self):
        return self.data[:, -2]

    @property
    def cls(self):
        return self.data[:, -1]

    @property
    def id(self):
        return self.data[:, 4] if self.is_track else None

    @property
    def is_track(self):
        return self.data.shape[1] == 7

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for row in self.data:
            yield Boxes(row.unsqueeze(0), self.orig_shape)


class Results:
    def __init__(self, image, path, names, boxes):
        self.orig_img = image
        self.orig_shape = image.shape[:2]
        self.path = str(path) if path is not None else ""
        self.names = names
        self.boxes = Boxes(boxes, self.orig_shape)


def _read_image(source):
    if isinstance(source, (str, Path)):
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(f"could not read image: {source}")
        return image, source
    if isinstance(source, np.ndarray):
        if source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
            raise ValueError("NumPy images must have shape (height, width, 3) and dtype uint8 (BGR)")
        return source, None
    raise TypeError("source must be an image path or BGR NumPy image")


def _letterbox(image, size):
    height, width = image.shape[:2]
    ratio = min(size[0] / height, size[1] / width)
    new_width, new_height = round(width * ratio), round(height * ratio)
    if new_width <= 0 or new_height <= 0:
        raise ValueError("image dimensions are too extreme to resize")
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pad_width, pad_height = size[1] - new_width, size[0] - new_height
    left, top = round(pad_width / 2 - 0.1), round(pad_height / 2 - 0.1)
    padded = cv2.copyMakeBorder(
        resized, top, pad_height - top, left, pad_width - left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return padded, ratio, (left, top)


def _size(imgsz):
    if isinstance(imgsz, int):
        imgsz = (imgsz, imgsz)
    if not isinstance(imgsz, (list, tuple)) or len(imgsz) != 2:
        raise ValueError("imgsz must be an integer or (height, width)")
    if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 or n % 32 for n in imgsz):
        raise ValueError("image dimensions must be positive multiples of 32")
    return tuple(imgsz)


def _postprocess(prediction, shape, ratio, padding, conf, iou, max_det, classes, agnostic):
    rows = prediction.transpose(0, 1)
    scores, labels = rows[:, 4:].max(1)
    mask = scores > conf
    if classes is not None:
        if isinstance(classes, int):
            classes = [classes]
        mask &= torch.isin(labels, torch.as_tensor(classes, device=labels.device))
    boxes = rows[mask, :4]
    scores, labels = scores[mask], labels[mask]
    corners = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), 1)
    if not agnostic:
        keep = batched_nms(corners, scores, labels, iou)[:max_det]
    else:
        keep = nms(corners, scores, iou)[:max_det]
    corners = corners[keep]
    corners[:, [0, 2]] = (corners[:, [0, 2]] - padding[0]) / ratio
    corners[:, [1, 3]] = (corners[:, [1, 3]] - padding[1]) / ratio
    corners[:, [0, 2]] = corners[:, [0, 2]].clamp(0, shape[1])
    corners[:, [1, 3]] = corners[:, [1, 3]].clamp(0, shape[0])
    return torch.cat((corners, scores[keep, None], labels[keep, None].to(corners.dtype)), 1)


def predict_image(model, source, imgsz=640, conf=0.25, iou=0.7,
                  max_det=300, classes=None, agnostic_nms=False):
    if not 0 <= conf <= 1 or not 0 <= iou <= 1 or not isinstance(max_det, int) or max_det < 1:
        raise ValueError("conf and iou must be in [0, 1] and max_det must be a positive integer")
    image, path = _read_image(source)
    padded, ratio, padding = _letterbox(image, _size(imgsz))
    array = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1))
    device = next(model.parameters()).device
    tensor = torch.from_numpy(array).to(device=device, dtype=torch.float32).unsqueeze(0) / 255
    with torch.inference_mode():
        prediction = model(tensor)[0]
        boxes = _postprocess(prediction, image.shape, ratio, padding,
                             conf, iou, max_det, classes, agnostic_nms)
    return Results(image, path, model.names, boxes)
