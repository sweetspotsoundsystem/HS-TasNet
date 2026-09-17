"""Measure exact in-memory checkpoint packing; preserve all existing files."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import resource
import time
import zlib

import numpy as np
import torch

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs
from research.direct.run_latency58_deployed_vocal_views import budget_snapshot


def tensor_bytes(tensor):
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def encode_tree(value, *, path=(), bases=None, rows=None):
    if isinstance(value, torch.Tensor):
        require(value.device.type == "cpu", "Encode existing CPU tensor values only")
        raw = tensor_bytes(value)
        width = value.element_size()
        matrix = np.frombuffer(raw, dtype=np.uint8).reshape(-1, width)
        base = bases.get(path) if bases else None
        if base is not None:
            require(base.dtype == value.dtype and base.shape == value.shape, "XOR reference differs")
            matrix = np.bitwise_xor(matrix, np.frombuffer(tensor_bytes(base), dtype=np.uint8).reshape(-1, width))
        packed = zlib.compress(matrix.T.copy().tobytes(), level=3)
        descriptor = {"codec": "byte-planes-zlib3-xor-v1", "dtype": str(value.dtype),
                      "shape": list(value.shape), "element_size": width, "xor_base": base is not None,
                      "data": torch.from_numpy(np.frombuffer(packed, dtype=np.uint8).copy())}
        recovered = decode_tensor(descriptor, base=base)
        require(recovered == raw, "Lossless tensor reconstruction differs")
        rows.append({"path": list(path), "dtype": str(value.dtype), "shape": list(value.shape),
                     "uncompressed_bytes": len(raw), "compressed_bytes": len(packed),
                     "xor_reference": base is not None, "roundtrip_bit_exact": True})
        return descriptor
    if isinstance(value, dict):
        return {key: encode_tree(item, path=(*path, key), bases=bases, rows=rows) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [encode_tree(item, path=(*path, index), bases=bases, rows=rows)
                     for index, item in enumerate(value)]
        return tuple(converted) if isinstance(value, tuple) else converted
    require(value is None or isinstance(value, (str, bytes, int, float, bool)), "Unsupported metadata type")
    return value


def decode_tensor(descriptor, *, base):
    width = descriptor["element_size"]
    raw = zlib.decompress(descriptor["data"].numpy())
    decoded = np.frombuffer(raw, dtype=np.uint8).reshape(width, -1).T.copy()
    if descriptor["xor_base"]:
        require(base is not None, "Missing XOR base")
        decoded = np.bitwise_xor(decoded, np.frombuffer(tensor_bytes(base), dtype=np.uint8).reshape(-1, width))
    return decoded.tobytes()


def compare_tree(original, restored, *, path=(), bases=None):
    if isinstance(original, torch.Tensor):
        require(restored["dtype"] == str(original.dtype) and restored["shape"] == list(original.shape)
                and decode_tensor(restored, base=bases.get(path) if bases else None) == tensor_bytes(original),
                "Serialized packed tensor differs after weights-only load")
        return 1
    if isinstance(original, dict):
        require(list(original) == list(restored), "Serialized dictionary order or keys differ")
        return sum(compare_tree(item, restored[key], path=(*path, key), bases=bases)
                   for key, item in original.items())
    if isinstance(original, (list, tuple)):
        require(type(original) is type(restored) and len(original) == len(restored), "Serialized sequence differs")
        return sum(compare_tree(item, restored[index], path=(*path, index), bases=bases)
                   for index, item in enumerate(original))
    require(type(original) is type(restored) and original == restored, "Serialized metadata differs")
    return 0


def main():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require CUDA-hidden CPU1")
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    out = PHASE / "lossless-checkpoint-packing-profile-001"
    require(not out.exists(), "Preserve previous packing measurements")
    source_path = PHASE / "branch-grouped-vocal-013/plan.json"
    source = read(source_path)
    review_path = PHASE / "grouped-continuation-review-013/result.json"
    review = read(review_path)
    candidates = review["candidate_models"]
    parent_binding = source["parent_checkpoint"]
    raw_binding, ema_binding = (candidates[key]["checkpoint"] for key in ("raw", "ema"))
    optimizer_path = Path(raw_binding["path"]).parent / "raw-optimizer.pt"
    bindings = {str(Path(__file__).resolve()): sha(__file__), str(source_path): sha(source_path),
                str(review_path): sha(review_path), str(optimizer_path): sha(optimizer_path)}
    bindings.update({item["path"]: item["sha256"] for item in (parent_binding, raw_binding, ema_binding)})
    budget = budget_snapshot(source["storage_budget"])
    require(budget["headroom_bytes"] > 1_000_000, "Reserve scalar-only packing measurements")
    plan = {"schema": "latency58-lossless-checkpoint-packing-measurement-v1", "source_bindings": bindings,
            "raw_xor_reference": parent_binding, "ema_xor_reference": raw_binding,
            "optimizer_encoding": "Byte planes and zlib level 3, no XOR base",
            "roundtrip": "Exact tensor bytes and all metadata before and after in-memory torch weights-only serialization",
            "input_scope": "Existing final 013 raw/EMA inference payloads and raw optimizer payload",
            "new_checkpoint_files_written": False, "storage_budget": source["storage_budget"], "budget_before": budget}
    verify_inputs(plan); out.mkdir(); write(out / "plan.json", plan)
    rng = torch.get_rng_state().clone(); began = time.monotonic()
    parent = torch.load(parent_binding["path"], map_location="cpu", weights_only=True)
    raw = torch.load(raw_binding["path"], map_location="cpu", weights_only=True)
    ema = torch.load(ema_binding["path"], map_location="cpu", weights_only=True)
    optimizer = torch.load(optimizer_path, map_location="cpu", weights_only=True)
    require(state_sha256(parent["model"]) == source["parent_model_state_sha256"]
            and state_sha256(raw["model"]) == candidates["raw"]["model_state_sha256"]
            and state_sha256(ema["model"]) == candidates["ema"]["model_state_sha256"]
            and optimizer["model_state_sha256"] == raw["model_state_sha256"]
            and optimizer["step"] == raw["step"] == ema["step"] == 1000
            and optimizer["parameter_names"] == raw["parameter_names"] == ema["parameter_names"],
            "Existing model identity or optimizer ownership differs")
    payloads = {"raw": raw, "ema": ema, "optimizer": optimizer}
    bases = {("raw", "model", key): value for key, value in parent["model"].items()}
    bases.update({("ema", "model", key): value for key, value in raw["model"].items()})
    rows = []
    encoded = {}
    for role, payload in payloads.items():
        encoded[role] = encode_tree(payload, path=(role,), bases=bases, rows=rows)
        role_rows = [row for row in rows if row["path"][0] == role]
        print(json.dumps({"event": "packed_payload_verified", "role": role,
                          "tensor_count": len(role_rows),
                          "uncompressed_tensor_bytes": sum(row["uncompressed_bytes"] for row in role_rows),
                          "compressed_tensor_bytes": sum(row["compressed_bytes"] for row in role_rows),
                          "elapsed_seconds": time.monotonic() - began}), flush=True)
    stream = io.BytesIO(); torch.save(encoded, stream); serialized_bytes = stream.tell(); stream.seek(0)
    restored = torch.load(stream, map_location="cpu", weights_only=True)
    tensor_count = compare_tree(payloads, restored, bases=bases)
    require(tensor_count == len(rows) and torch.equal(rng, torch.get_rng_state())
            and not torch.cuda.is_initialized(), "Roundtrip inventory, RNG or device changed")
    verify_inputs(plan)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    require(rss < 8_000_000_000, "In-memory packing exceeded the CPU memory allowance")
    result = {"schema": plan["schema"], "status": "pass", "plan_sha256": sha(out / "plan.json"),
              "source_bindings_unchanged": True, "all_tensor_and_metadata_roundtrips_exact": True,
              "tensor_count": tensor_count, "tensor_rows": rows, "packed_torch_archive_bytes": serialized_bytes,
              "original_three_files_bytes": sum(Path(p).stat().st_size for p in
                                                (raw_binding["path"], ema_binding["path"], str(optimizer_path))),
              "elapsed_seconds": time.monotonic() - began, "peak_rss_bytes": rss,
              "new_checkpoint_files_written": False, "gpu_used": False, "quality_measured": False,
              "budget_after": budget_snapshot(source["storage_budget"]),
              "limitations": ["Existing endpoint measurements do not bound the size of future trained tensors.",
                              "This is an in-memory codec profile, not a qualified atomic recovery writer or model loader.",
                              "The external XOR parent must remain authenticated and available.",
                              "Training journals, additional RNG state, recovery receipts and file publication overhead need a separate allowance."]}
    write(out / "result.json", result)
    print(json.dumps({key: result[key] for key in ("status", "tensor_count", "packed_torch_archive_bytes",
                                                  "original_three_files_bytes", "elapsed_seconds", "peak_rss_bytes")}), flush=True)


if __name__ == "__main__":
    main()
