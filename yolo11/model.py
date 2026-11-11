"""YOLO11 detection network and restricted loader for compatible checkpoints."""

import collections
import pickle
import types

import torch
from torch import nn


class Conv(nn.Module):
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(Conv):
    pass


class Bottleneck(nn.Module):
    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3k(nn.Module):
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k2(nn.Module):
    def forward(self, x):
        parts = list(self.cv1(x).chunk(2, 1))
        for module in self.m:
            parts.append(module(parts[-1]))
        return self.cv2(torch.cat(parts, 1))


class SPPF(nn.Module):
    def forward(self, x):
        parts = [self.cv1(x)]
        for _ in range(3):
            parts.append(self.m(parts[-1]))
        return self.cv2(torch.cat(parts, 1))


class Attention(nn.Module):
    def forward(self, x):
        batch, channels, height, width = x.shape
        count = height * width
        qkv = self.qkv(x).reshape(batch, self.num_heads, 2 * self.key_dim + self.head_dim, count)
        q, k, v = qkv.split((self.key_dim, self.key_dim, self.head_dim), 2)
        weights = ((q.transpose(-2, -1) @ k) * self.scale).softmax(-1)
        attended = (v @ weights.transpose(-2, -1)).reshape(batch, channels, height, width)
        position = self.pe(v.reshape(batch, channels, height, width))
        return self.proj(attended + position)


class PSABlock(nn.Module):
    def forward(self, x):
        attended = self.attn(x)
        x = x + attended if self.add else attended
        fed = self.ffn(x)
        return x + fed if self.add else fed


class C2PSA(nn.Module):
    def forward(self, x):
        left, right = self.cv1(x).split((self.c, self.c), 1)
        return self.cv2(torch.cat((left, self.m(right)), 1))


class Concat(nn.Module):
    def forward(self, inputs):
        return torch.cat(inputs, self.d)


class DFL(nn.Module):
    def forward(self, x):
        batch, _, count = x.shape
        return self.conv(x.view(batch, 4, self.c1, count).transpose(1, 2).softmax(1)).view(batch, 4, count)


class Detect(nn.Module):
    def forward(self, inputs):
        outputs = [torch.cat((self.cv2[i](x), self.cv3[i](x)), 1) for i, x in enumerate(inputs)]
        if self.training:
            return outputs
        batch = outputs[0].shape[0]
        predictions = torch.cat([x.reshape(batch, self.no, -1) for x in outputs], 2)
        distances, scores = predictions.split((self.reg_max * 4, self.nc), 1)
        anchors, strides = [], []
        for i, feature in enumerate(outputs):
            height, width = feature.shape[-2:]
            y, x = torch.meshgrid(
                torch.arange(height, device=feature.device, dtype=feature.dtype) + 0.5,
                torch.arange(width, device=feature.device, dtype=feature.dtype) + 0.5,
                indexing="ij",
            )
            anchors.append(torch.stack((x, y), -1).reshape(-1, 2))
            strides.append(torch.full_like(x.reshape(-1, 1), (8, 16, 32)[i]))
        anchors = torch.cat(anchors, 0).transpose(0, 1).unsqueeze(0)
        strides = torch.cat(strides, 0).transpose(0, 1).unsqueeze(0)
        left_top, right_bottom = self.dfl(distances).chunk(2, 1)
        top_left, bottom_right = anchors - left_top, anchors + right_bottom
        boxes = torch.cat(((top_left + bottom_right) / 2, bottom_right - top_left), 1) * strides
        return torch.cat((boxes, scores.sigmoid()), 1)


class DetectionModel(nn.Module):
    def forward(self, x):
        saved = []
        for layer in self.model:
            if layer.f != -1:
                if isinstance(layer.f, int):
                    x = saved[layer.f]
                else:
                    x = [x if index == -1 else saved[index] for index in layer.f]
            x = layer(x)
            saved.append(x)
        return x


_MODEL_CLASSES = {
    ("ultralytics.nn.tasks", "DetectionModel"): DetectionModel,
    **{("ultralytics.nn.modules.conv", cls.__name__): cls for cls in (Conv, DWConv, Concat)},
    **{("ultralytics.nn.modules.block", cls.__name__): cls for cls in (
        Bottleneck, C3k, C3k2, SPPF, Attention, PSABlock, C2PSA, DFL
    )},
    ("ultralytics.nn.modules.head", "Detect"): Detect,
}

_ALLOWED = {
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("__builtin__", "set"): set,
    ("builtins", "set"): set,
    ("torch", "Size"): torch.Size,
    ("torch", "HalfStorage"): torch.HalfStorage,
    ("torch", "FloatStorage"): torch.FloatStorage,
    ("torch", "LongStorage"): torch.LongStorage,
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
    ("torch._utils", "_rebuild_parameter"): torch._utils._rebuild_parameter,
    ("torch.nn.modules.container", "Sequential"): nn.Sequential,
    ("torch.nn.modules.container", "ModuleList"): nn.ModuleList,
    ("torch.nn.modules.conv", "Conv2d"): nn.Conv2d,
    ("torch.nn.modules.batchnorm", "BatchNorm2d"): nn.BatchNorm2d,
    ("torch.nn.modules.activation", "SiLU"): nn.SiLU,
    ("torch.nn.modules.linear", "Identity"): nn.Identity,
    ("torch.nn.modules.pooling", "MaxPool2d"): nn.MaxPool2d,
    ("torch.nn.modules.upsampling", "Upsample"): nn.Upsample,
}


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        key = (module, name)
        if key in _MODEL_CLASSES:
            return _MODEL_CLASSES[key]
        if key in _ALLOWED:
            return _ALLOWED[key]
        raise pickle.UnpicklingError(f"unsupported checkpoint global: {module}.{name}")


_PICKLE = types.SimpleNamespace(
    __name__="yolo11.restricted_pickle", Unpickler=_CheckpointUnpickler,
    load=pickle.load, loads=pickle.loads, dump=pickle.dump, dumps=pickle.dumps,
)


def load_model(path, device="cpu"):
    """Load a compatible YOLO11 detection checkpoint using a restricted unpickler.

    Only the original detection checkpoint object format is supported. A
    checkpoint is still a complex binary input: use trusted files only.
    """
    checkpoint = torch.load(path, map_location="cpu", pickle_module=_PICKLE, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a mapping")
    model = checkpoint.get("ema")
    if model is None:
        model = checkpoint.get("model")
    if not isinstance(model, DetectionModel) or not isinstance(model.model[-1], Detect):
        raise ValueError("checkpoint is not a supported YOLO11 detection model")
    yaml = getattr(model, "yaml", {})
    if not str(yaml.get("yaml_file", "")).startswith("yolo11"):
        raise ValueError("checkpoint is not identified as a YOLO11 model")
    return model.float().to(device).eval()
