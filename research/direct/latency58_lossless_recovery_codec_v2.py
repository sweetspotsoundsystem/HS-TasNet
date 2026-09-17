"""Lossless packed recovery of the existing raw/Adam/EMA snapshot schema.

Only representation changes. Decoded snapshots pass the original generation,
optimizer, EMA, RNG, journal and training-schedule audits before use.
"""
from __future__ import annotations

import io
import math
import sys
import zlib

import numpy as np
import torch

from research.direct.run_latency58_quality import require
from research.direct.train_latency58 import state_sha256
from research.direct.latency58_branch_memory_checkpoint import load_model
from research.direct.latency58_grouped_vocal_recovery import audit_snapshot
from research.direct.profile_latency58_lossless_checkpoint import encode_tree, tensor_bytes


SCHEMA = "latency58-lossless-packed-grouped-recovery-v1"
TENSOR_CODEC = "byte-planes-zlib3-xor-v1"
MAX_FILE_BYTES = 380_000_000
MAX_TENSOR_BYTES = 550_000_000
MAX_METADATA_BYTES = 30_000_000
MAX_METADATA_NODES = 400_000
DTYPES = {"torch.float32": torch.float32, "torch.int64": torch.int64, "torch.uint8": torch.uint8}
DESCRIPTOR_KEYS = {"codec", "dtype", "shape", "element_size", "xor_base", "data"}


def policy():
    return {"schema": SCHEMA, "tensor_codec": TENSOR_CODEC, "byteorder": "little",
            "maximum_file_bytes": MAX_FILE_BYTES, "maximum_decoded_tensor_bytes": MAX_TENSOR_BYTES,
            "maximum_decoded_metadata_bytes": MAX_METADATA_BYTES,
            "maximum_metadata_nodes": MAX_METADATA_NODES,
            "metadata_accounting": "Recursive Python object sizes, container storage and keys; 512 bytes per tensor leaf",
            "raw_reference": "Authenticated retained parent model tensors",
            "ema_reference": "Raw tensors decoded from the same snapshot",
            "optimizer_and_rng": "Lossless byte planes without XOR",
            "training_snapshot_schema_and_audits": "Unchanged grouped rolling recovery schema",
            "inference_and_numeric_precision_changed": False}


def parent_state(plan):
    model, payload = load_model(plan["parent_checkpoint"])
    require(state_sha256(payload["model"]) == plan["parent_model_state_sha256"], "Packed recovery parent state differs")
    del model
    return payload["model"]


def metadata_inventory(value, *, packed=False):
    """Bound decoded metadata, including keys, numeric leaves and containers."""
    budget = {"metadata_bytes": 0, "metadata_nodes": 0}
    def visit(item, depth):
        require(depth <= 32, "Packed metadata nesting exceeds its bound")
        budget["metadata_nodes"] += 1
        tensor = isinstance(item, torch.Tensor)
        descriptor = packed and type(item) is dict and item.get("codec") == TENSOR_CODEC
        if descriptor or tensor:
            require(descriptor or not packed, "Unexpected unpacked tensor in encoded snapshot")
            size = 512
        else:
            require(item is None or type(item) in (dict, list, tuple, str, bytes, int, float, bool),
                    "Unsupported packed metadata value")
            size = sys.getsizeof(item)
        budget["metadata_bytes"] += size
        require(budget["metadata_bytes"] <= MAX_METADATA_BYTES and budget["metadata_nodes"] <= MAX_METADATA_NODES,
                "Packed metadata exceeds its declared bound")
        if descriptor or tensor:
            return
        if type(item) is dict:
            require(all(type(key) in (str, int) for key in item), "Unsupported packed metadata key")
            for key, child in item.items():
                visit(key, depth + 1)
                visit(child, depth + 1)
        elif type(item) in (list, tuple):
            for child in item:
                visit(child, depth + 1)
    visit(value, 0)
    return budget


def validate_descriptor(value, base, budget):
    require(type(value) is dict and set(value) == DESCRIPTOR_KEYS and value["codec"] == TENSOR_CODEC
            and value["dtype"] in DTYPES and type(value["shape"]) is list
            and len(value["shape"]) <= 8
            and all(type(size) is int and 0 <= size <= MAX_TENSOR_BYTES for size in value["shape"])
            and type(value["element_size"]) is int and type(value["xor_base"]) is bool,
            "Malformed packed tensor descriptor")
    dtype = DTYPES[value["dtype"]]
    width = torch.empty(0, dtype=dtype).element_size()
    count = math.prod(value["shape"])
    expected = count * width
    data = value["data"]
    require(value["element_size"] == width and 0 <= expected <= MAX_TENSOR_BYTES
            and isinstance(data, torch.Tensor) and data.device.type == "cpu"
            and data.dtype == torch.uint8 and data.ndim == 1 and data.is_contiguous()
            and 0 < data.numel() <= MAX_FILE_BYTES, "Invalid packed tensor size, dtype or device")
    budget["tensor_bytes"] += expected
    budget["packed_bytes"] += data.numel()
    budget["tensor_count"] += 1
    require(budget["tensor_bytes"] <= MAX_TENSOR_BYTES and budget["packed_bytes"] <= MAX_FILE_BYTES
            and budget["tensor_count"] <= 512, "Packed tensor inventory exceeds its declared bound")
    if value["xor_base"]:
        require(isinstance(base, torch.Tensor) and base.device.type == "cpu"
                and base.dtype == dtype and list(base.shape) == value["shape"], "Packed XOR reference differs")
    else:
        require(base is None, "Declared raw/EMA reference is unexpectedly unused")
    return dtype, width, expected


def decode_tree(value, *, path=(), bases, budget, depth=0):
    require(depth <= 32, "Packed metadata nesting exceeds its bound")
    if isinstance(value, dict) and value.get("codec") == TENSOR_CODEC:
        base = bases.get(path)
        dtype, width, expected = validate_descriptor(value, base, budget)
        decoder = zlib.decompressobj()
        try:
            planes = decoder.decompress(value["data"].numpy(), expected + 1)
        except zlib.error as error:
            raise RuntimeError("Invalid packed tensor compression stream") from error
        require(len(planes) == expected and decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
                "Packed tensor stream has wrong length, missing end or trailing bytes")
        matrix = np.frombuffer(planes, dtype=np.uint8).reshape(width, -1).T.copy()
        if base is not None:
            matrix = np.bitwise_xor(matrix, np.frombuffer(tensor_bytes(base), dtype=np.uint8).reshape(-1, width))
        if not expected:
            return torch.empty(value["shape"], dtype=dtype)
        return torch.from_numpy(matrix.reshape(-1)).view(dtype).reshape(value["shape"])
    require(not isinstance(value, torch.Tensor), "Unexpected unpacked tensor in encoded snapshot")
    if isinstance(value, dict):
        require(all(type(key) in (str, int) for key in value), "Unsupported packed metadata key")
        return {key: decode_tree(item, path=(*path, key), bases=bases, budget=budget, depth=depth + 1)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [decode_tree(item, path=(*path, index), bases=bases, budget=budget, depth=depth + 1)
                 for index, item in enumerate(value)]
        return tuple(items) if isinstance(value, tuple) else items
    require(value is None or type(value) in (str, bytes, int, float, bool), "Unsupported packed metadata value")
    return value


def pack_snapshot(snapshot, plan, plan_sha):
    require(sys.byteorder == "little" and plan["packed_recovery"] == policy(), "Unprepared lossless recovery policy")
    metadata = metadata_inventory(snapshot)
    audited = audit_snapshot(snapshot, plan, plan_sha)
    del audited
    parent = parent_state(plan)
    bases = {("raw", "model", name): value for name, value in parent.items()}
    bases.update({("average", "model", name): value for name, value in snapshot["raw"]["model"].items()})
    rows = []
    encoded = encode_tree(snapshot, bases=bases, rows=rows)
    require(sum(row["uncompressed_bytes"] for row in rows) <= MAX_TENSOR_BYTES
            and len(rows) <= 512 and all(row["dtype"] in DTYPES for row in rows), "Unsupported snapshot tensor inventory")
    envelope = {"schema": SCHEMA, "policy": policy(), "parent_checkpoint": plan["parent_checkpoint"],
                "parent_model_state_sha256": plan["parent_model_state_sha256"], "plan_sha256": plan_sha,
                "step": snapshot["step"], "raw_model_state_sha256": snapshot["raw"]["model_state_sha256"],
                "ema_model_state_sha256": snapshot["average"]["model_state_sha256"],
                "snapshot": encoded}
    with io.BytesIO() as stream:
        torch.save(envelope, stream)
        require(0 < stream.tell() <= MAX_FILE_BYTES, "Packed recovery exceeds its file-size ceiling")
        serialized = stream.getvalue()
    return serialized, {"file_bytes": len(serialized), "tensor_count": len(rows),
                        "uncompressed_tensor_bytes": sum(row["uncompressed_bytes"] for row in rows),
                        "compressed_tensor_bytes": sum(row["compressed_bytes"] for row in rows), **metadata}


def unpack_snapshot(serialized, plan, plan_sha):
    require(sys.byteorder == "little" and plan["packed_recovery"] == policy()
            and type(serialized) is bytes and 0 < len(serialized) <= MAX_FILE_BYTES,
            "Invalid packed recovery input or policy")
    envelope = torch.load(io.BytesIO(serialized), map_location="cpu", weights_only=True)
    require(type(envelope) is dict and set(envelope) == {
        "schema", "policy", "parent_checkpoint", "parent_model_state_sha256", "plan_sha256",
        "step", "raw_model_state_sha256", "ema_model_state_sha256", "snapshot"}
        and envelope["schema"] == SCHEMA and envelope["policy"] == policy()
        and envelope["parent_checkpoint"] == plan["parent_checkpoint"]
        and envelope["parent_model_state_sha256"] == plan["parent_model_state_sha256"]
        and envelope["plan_sha256"] == plan_sha and type(envelope["step"]) is int
        and 0 < envelope["step"] <= plan["config"]["steps"], "Packed recovery envelope identity differs")
    encoded = envelope["snapshot"]
    require(type(encoded) is dict and "raw" in encoded and "average" in encoded,
            "Packed recovery lacks its raw and EMA roles")
    metadata = metadata_inventory(encoded, packed=True)
    parent = parent_state(plan)
    bases = {("raw", "model", name): value for name, value in parent.items()}
    budget = {"tensor_bytes": 0, "packed_bytes": 0, "tensor_count": 0, **metadata}
    raw = decode_tree(encoded["raw"], path=("raw",), bases=bases, budget=budget)
    require(type(raw) is dict and type(raw.get("model")) is dict
            and state_sha256(raw["model"]) == envelope["raw_model_state_sha256"], "Decoded raw model hash differs")
    bases.update({("average", "model", name): value for name, value in raw["model"].items()})
    snapshot = {key: raw if key == "raw" else decode_tree(value, path=(key,), bases=bases, budget=budget)
                for key, value in encoded.items()}
    require(snapshot["step"] == envelope["step"]
            and state_sha256(snapshot["average"]["model"]) == envelope["ema_model_state_sha256"],
            "Decoded EMA model or step differs")
    audited = audit_snapshot(snapshot, plan, plan_sha)
    del audited
    return snapshot, budget
