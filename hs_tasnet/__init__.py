"""HS-TasNet training and streaming inference APIs.

Load training dependencies only when a training API is requested, so importing
the ONNX streaming module does not initialize PyTorch or dataset libraries.
"""

from importlib import import_module

__all__ = ["HSTasNet", "Trainer", "MusDB18HQ"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = ".hs_tasnet" if name == "HSTasNet" else ".trainer"
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
