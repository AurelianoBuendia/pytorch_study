"""Components for an independent YOLO11 implementation.
This code implements lazy loading (also known as PEP 562 module-level __getattr__) for the YOLO class inside
a Python package's __init__.py file.
Instead of importing YOLO immediately when the package is loaded, it delays the import until the user actually
tries to access it.
def __getattr__(name): This is a special Python hook introduced in Python 3.7. Python automatically calls this
function whenever someone tries to access an attribute from this module (e.g., from yolo11 import YOLO or
yolo11.YOLO) that hasn't already been defined or imported at the top level.

"""
from typing import TYPE_CHECKING
from .byte_tracker import load_config


if TYPE_CHECKING:
    # IDEs read this block, but Python skips it at runtime (always False)
    from .api import YOLO
    from .byte_tracker import BYTETracker


def __getattr__(name):
    if name == "YOLO":
        from .api import YOLO
        return YOLO
    if name == "BYTETracker":
        from .byte_tracker import BYTETracker
        return BYTETracker
    # Ensures standard Python behavior if someone tries to import something that doesn't exist
    raise AttributeError(name)


__all__ = ["YOLO", "BYTETracker", "load_config"]
