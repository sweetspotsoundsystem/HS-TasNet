"""Self-contained lossless tensor packing; no parent checkpoint is required."""
from __future__ import annotations

import math
import zlib

import numpy as np
import torch

CODEC = "byte-planes-zlib3-v1"
DTYPES = {str(dtype): dtype for dtype in (torch.float32, torch.float64, torch.int64, torch.uint8)}
MAX_TENSOR_BYTES = 1_000_000_000


def pack(value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        if str(tensor.dtype) not in DTYPES:
            raise ValueError(f"Unsupported checkpoint tensor dtype: {tensor.dtype}")
        width = tensor.element_size()
        data = tensor.reshape(-1).view(torch.uint8).numpy().reshape(-1, width).T.tobytes()
        compressed = torch.from_numpy(np.frombuffer(zlib.compress(data, level=3), dtype=np.uint8).copy())
        return {"__tensor_codec__": CODEC, "dtype": str(tensor.dtype), "shape": list(tensor.shape),
                "data": compressed}
    if isinstance(value, dict):
        return {key: pack(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(pack(item) for item in value)
    return value


def unpack(value, *, budget=None, depth=0):
    if budget is None:
        budget = {"bytes": 0, "nodes": 0}
    budget["nodes"] += 1
    if depth > 32 or budget["nodes"] > 1_000_000:
        raise ValueError("Checkpoint metadata exceeds its decoding bound")
    if isinstance(value, dict) and "__tensor_codec__" in value:
        if (set(value) != {"__tensor_codec__", "dtype", "shape", "data"}
                or value["__tensor_codec__"] != CODEC or value["dtype"] not in DTYPES
                or not isinstance(value["shape"], list) or len(value["shape"]) > 8
                or not all(type(size) is int and 0 <= size <= MAX_TENSOR_BYTES for size in value["shape"])
                or not isinstance(value["data"], torch.Tensor) or value["data"].dtype != torch.uint8
                or value["data"].device.type != "cpu" or value["data"].ndim != 1
                or not value["data"].is_contiguous()):
            raise ValueError("Malformed packed checkpoint tensor")
        dtype = DTYPES[value["dtype"]]
        width = torch.empty(0, dtype=dtype).element_size()
        expected = math.prod(value["shape"]) * width
        budget["bytes"] += expected
        if budget["bytes"] > MAX_TENSOR_BYTES:
            raise ValueError("Checkpoint tensors exceed their decoding bound")
        decoder = zlib.decompressobj()
        try:
            raw = decoder.decompress(value["data"].numpy(), expected + 1)
        except zlib.error as error:
            raise ValueError("Corrupt packed checkpoint tensor") from error
        if (len(raw) != expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail):
            raise ValueError("Packed checkpoint tensor has the wrong byte length")
        if expected == 0:
            return torch.empty(value["shape"], dtype=dtype)
        array = np.frombuffer(raw, dtype=np.uint8).reshape(width, -1).T.copy().reshape(-1)
        return torch.from_numpy(array).view(dtype).reshape(value["shape"])
    if isinstance(value, dict):
        return {key: unpack(item, budget=budget, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(unpack(item, budget=budget, depth=depth + 1) for item in value)
    if isinstance(value, torch.Tensor):
        raise ValueError("Unexpected unpacked tensor in a compressed checkpoint")
    return value
