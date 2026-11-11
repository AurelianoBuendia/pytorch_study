"""Public interface to the independent YOLO11 detection model."""

from pathlib import Path

import torch

from .inference import _size, predict_image
from .model import load_model


class YOLO:
    def __init__(self, model, device="cpu"):
        self.path = Path(model)
        self.model = load_model(self.path, device=device)
        self.names = self.model.names

    def predict(self, source, *, imgsz=640, conf=0.25, iou=0.7, max_det=300,
                classes=None, agnostic_nms=False, stream=False):
        sources = source if isinstance(source, (list, tuple)) else [source]
        def generate():
            for item in sources:
                yield predict_image(self.model, item, imgsz, conf, iou, max_det, classes, agnostic_nms)
        results = generate()
        return results if stream else list(results)

    __call__ = predict

    def export(self, *, format="onnx", imgsz=640, dynamic=False, opset=17, path=None):
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
