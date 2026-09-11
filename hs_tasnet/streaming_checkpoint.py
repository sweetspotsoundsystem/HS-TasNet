"""Portable PyTorch checkpoints and exact import of the released ONNX weights."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

import torch

from .streaming_model import StreamingHSTasNet, VERSION, require

CHECKPOINT_SCHEMA = "hs-tasnet-streaming-checkpoint-v1"
RELEASED_STATE_SHA256 = "c204b0fcb9627ca7fecd287db42fb869a1ae6783a1bc24cf2d8864c3b4a565fb"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(state):
    """Fingerprint tensor names, shapes, dtypes and exact CPU bytes."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def save_streaming_checkpoint(model, path, *, training=None):
    """Save an inference checkpoint, optionally with optimizer and RNG state.

    Existing destinations are preserved. Only a complete same-directory file
    is published; training supplies its resume payload explicitly.
    """
    require(isinstance(model, StreamingHSTasNet), "Expected StreamingHSTasNet")
    state = cpu_tree(model.state_dict())
    require(all(value.dtype == torch.float32 and bool(torch.isfinite(value).all()) for value in state.values()),
            "Checkpoint tensors must be finite FP32")
    payload = {"schema": CHECKPOINT_SCHEMA, "architecture": VERSION, "model": state,
               "model_state_sha256": state_sha256(state)}
    if training is not None:
        payload["training"] = cpu_tree(training)
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def read_streaming_checkpoint(path):
    """Read tensors and primitive metadata with PyTorch's restricted loader."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    require(isinstance(payload, dict) and payload.get("schema") == CHECKPOINT_SCHEMA
            and payload.get("architecture") == VERSION
            and set(payload) in ({"schema", "architecture", "model", "model_state_sha256"},
                                 {"schema", "architecture", "model", "model_state_sha256", "training"}),
            "Unsupported streaming checkpoint")
    state = payload["model"]
    require(isinstance(state, dict) and all(isinstance(key, str) and isinstance(value, torch.Tensor)
            and value.dtype == torch.float32 and bool(torch.isfinite(value).all()) for key, value in state.items()),
            "Checkpoint contains invalid model tensors")
    require(state_sha256(state) == payload["model_state_sha256"], "Checkpoint tensor fingerprint differs")
    return payload


def model_from_payload(payload):
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = StreamingHSTasNet()
    expected = model.state_dict()
    state = payload["model"]
    require(set(state) == set(expected) and all(state[key].shape == value.shape for key, value in expected.items()),
            "Checkpoint parameter names or dimensions differ")
    for name in ("analysis_window", "synthesis.window", "synthesis.spectral_window",
                 "synthesis.spectral_denominator", "synthesis.waveform_window_sum"):
        require(torch.equal(state[name], expected[name]), "Checkpoint changes a fixed synthesis buffer: " + name)
    model.load_state_dict(state, strict=True)
    return model.eval()


def load_streaming_checkpoint(path):
    return model_from_payload(read_streaming_checkpoint(path))


def import_released_onnx(path):
    """Recover every trained tensor from the checksum-pinned released graph.

    This importer supports the released graph only. Its three constant-folded
    linear weights are transposed back, and source scales are reshaped. The
    unused waveform window sum is rebuilt from its fixed Hann window. The
    final fingerprint must equal the original PyTorch training checkpoint.
    """
    from .streaming import MODEL_SHA256
    import onnx
    from onnx import numpy_helper

    path = Path(path)
    require(file_sha256(path) == MODEL_SHA256, "Import requires the released ONNX model checksum")
    graph = onnx.load(str(path), load_external_data=False)
    require(all(value.data_location != onnx.TensorProto.EXTERNAL for value in graph.graph.initializer),
            "The released graph must contain all its weights")
    tensors = {value.name: torch.from_numpy(numpy_helper.to_array(value).copy())
               for value in graph.graph.initializer}
    require(len(tensors) == len(graph.graph.initializer) == 26, "Released initializer inventory differs")
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = StreamingHSTasNet()
    state = {}
    transformed = {"spec_encode.weight": "onnx::MatMul_362", "to_spec_masks.weight": "onnx::MatMul_363",
                   "to_waveform_masks.weight": "onnx::MatMul_376"}
    for name, expected in model.state_dict().items():
        if name in transformed:
            value = tensors[transformed[name]].t().contiguous()
        elif name == "output_source_scales":
            value = tensors["onnx::Mul_387"].reshape(4)
        elif name == "synthesis.waveform_window_sum":
            window = tensors["model.synthesis.window"]
            value = window[:128] + window[128:]
        else:
            value = tensors["model." + name]
        require(value.shape == expected.shape and value.dtype == torch.float32 and bool(torch.isfinite(value).all()),
                "Released tensor differs: " + name)
        state[name] = value
    require(state_sha256(state) == RELEASED_STATE_SHA256 and file_sha256(path) == MODEL_SHA256,
            "Imported parameters do not exactly reproduce the training checkpoint")
    model.load_state_dict(state, strict=True)
    return model.eval()
