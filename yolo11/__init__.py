"""Components for an independent YOLO11 implementation."""


def __getattr__(name):
    if name == "YOLO":
        from .api import YOLO
        return YOLO
    raise AttributeError(name)


__all__ = ["YOLO"]
