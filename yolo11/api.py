"""Public interface to the independent YOLO11 detection model."""

from pathlib import Path

import cv2
import numpy as np
import torch

from .byte_tracker import BYTETracker, load_config
from .inference import Boxes, _size, predict_image
from .model import load_model

_VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}


class YOLO:
    def __init__(self, model, device="cpu"):
        self.path = Path(model)
        self.model = load_model(self.path, device=device)
        self.names = self.model.names
        self._tracker = None
        self._tracking = False

    def predict(self, source, *, imgsz=640, conf=0.25, iou=0.7, max_det=300,
                classes=None, agnostic_nms=False, stream=False):
        sources = source if isinstance(source, (list, tuple)) else [source]
        def generate():
            for item in sources:
                yield predict_image(self.model, item, imgsz, conf, iou, max_det, classes, agnostic_nms)
        results = generate()
        return results if stream else list(results)

    __call__ = predict

    def reset_tracker(self):
        if self._tracking:
            raise RuntimeError("cannot reset while a tracking stream is active")
        self._tracker = None

    def track(self, source, *, tracker="bytetrack.yaml", persist=False, stream=False,
              imgsz=640, conf=None, iou=0.7, max_det=300, classes=None,
              agnostic_nms=False, frame_rate=None):
        """Track an image, a sequence of images, or a local video.

        A tracking call starts a new sequence unless ``persist=True``.
        For a video, FPS is read from the file unless ``frame_rate`` is provided.
        """
        config = load_config(tracker)
        if conf is None:
            conf = min(0.1, config.track_low_thresh)
        if frame_rate is not None and (
            isinstance(frame_rate, bool) or not isinstance(frame_rate, (int, float))
            or not np.isfinite(frame_rate) or frame_rate <= 0
        ):
            raise ValueError("frame_rate must be positive and finite")

        def generate():
            if self._tracking:
                raise RuntimeError("another tracking stream is active on this model")
            self._tracking = True
            capture = None
            try:
                video = isinstance(source, (str, Path)) and Path(source).suffix.lower() in _VIDEO_SUFFIXES
                fps = frame_rate
                if video:
                    capture = cv2.VideoCapture(str(source))
                    if not capture.isOpened():
                        raise ValueError(f"could not open video: {source}")
                    if fps is None:
                        fps = capture.get(cv2.CAP_PROP_FPS)
                        if not np.isfinite(fps) or fps <= 0:
                            raise ValueError(f"video has invalid frame rate; provide frame_rate: {source}")
                    def frames():
                        while True:
                            ok, frame = capture.read()
                            if not ok:
                                break
                            yield frame
                    inputs = frames()
                else:
                    inputs = iter(source if isinstance(source, (list, tuple)) else [source])
                fps = fps if fps is not None else 30
                if persist and self._tracker is not None:
                    if self._tracker.config != config or self._tracker.frame_rate != fps:
                        raise ValueError("persistent tracking requires the same tracker configuration and frame rate; "
                                         "call reset_tracker() to start a new sequence")
                    tracker_instance = self._tracker
                else:
                    tracker_instance = BYTETracker(config, frame_rate=fps)
                    self._tracker = tracker_instance if persist else None
                for image in inputs:
                    result = predict_image(self.model, image, imgsz, conf, iou, max_det, classes, agnostic_nms)
                    detections = result.boxes.data
                    valid = (detections[:, 2] > detections[:, 0]) & (detections[:, 3] > detections[:, 1])
                    tracked = tracker_instance.update(detections[valid].detach().cpu().numpy())
                    data = torch.as_tensor(tracked, device=detections.device, dtype=detections.dtype)
                    data[:, [0, 2]] = data[:, [0, 2]].clamp(0, result.orig_shape[1])
                    data[:, [1, 3]] = data[:, [1, 3]].clamp(0, result.orig_shape[0])
                    result.boxes = Boxes(data, result.orig_shape)
                    if video:
                        result.path = str(source)
                    yield result
            finally:
                if capture is not None:
                    capture.release()
                self._tracking = False

        results = generate()
        return results if stream else list(results)

    def export(self, *, format="onnx", imgsz=640, dynamic=False, opset=18, path=None):
        if format != "onnx":
            raise ValueError("only ONNX export is supported")
        height, width = _size(imgsz)
        if not isinstance(opset, int) or opset < 12:
            raise ValueError("opset must be an integer of at least 12")
        destination = Path(path) if path is not None else self.path.with_suffix(".onnx")
        if destination.resolve() == self.path.resolve():
            raise ValueError("export destination must not overwrite the checkpoint")
        device = next(self.model.parameters()).device
        sample = torch.zeros(1, 3, height, width, device=device)
        axes = {
            "images": {0: "batch", 2: "height", 3: "width"},
            "output0": {0: "batch", 2: "anchors"},
        } if dynamic else None
        with torch.inference_mode():
            torch.onnx.export(self.model, sample, str(destination), opset_version=opset,
                              input_names=["images"], output_names=["output0"],
                              dynamic_axes=axes)
        return str(destination)
