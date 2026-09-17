"""Generate portable native-test expectations without importing ONNX Runtime."""
from __future__ import annotations

import argparse
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys


class RejectRuntimeImport(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "onnxruntime" or fullname.startswith("onnxruntime."):
            # PyTorch inspects optional dependencies with find_spec. Permit
            # that read-only lookup while preventing actual module execution.
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise RuntimeError("The fixture oracle must not import ONNX Runtime")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", default="attention-qkv-int8-fixture-001")
    args = parser.parse_args()
    from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
    FOLLOWUP = ROOT / "research/m4_followup_20260916"
    require("onnxruntime" not in sys.modules and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require CUDA-hidden CPU1 with no runtime oracle")
    sys.meta_path.insert(0, RejectRuntimeImport())
    import numpy as np
    import onnx
    import torch
    from research.direct.train_latency58 import verify_inputs, state_sha256
    from research.direct.latency58_branch_memory_checkpoint import load_model
    from research.direct.latency58_attention_qkv_int8_reference import make_reference
    from research.direct.latency58_m4_followup_budget import POLICY, snapshot
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    short_root, long_root = FOLLOWUP / "attention-qkv-int8-screen-001", FOLLOWUP / "attention-qkv-int8-long-001"
    short, long = read(short_root / "result.json"), read(long_root / "result.json")
    receipts = (FOLLOWUP / "attention-qkv-int8-execution.json", FOLLOWUP / "attention-qkv-long-execution.json")
    for directory, result, receipt in zip((short_root, long_root), (short, long), receipts, strict=True):
        execution = read(receipt)
        require(result["status"] == "pass" and result["source_bindings_unchanged"]
                and execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                and not execution["timed_out"], "Numerical evidence is incomplete")
    require(short["strict_parity_passed"] and short["graph_sha256"] == long["graph_sha256"]
            == "08424ca91feae8d4746442a35ebf70489dea70ea6e81401b39483cf02d497748", "Wrong candidate graph")
    graph_path, parent_path = Path(short["saved_graph_path"]), PHASE / "m4-inference-diagnostics-plugin-001/model/model.onnx"
    parent_proof_path = PHASE / "branch-plugin-screen-001/graph.json"
    checkpoint = short["source_checkpoint"]
    # Retain the qualified inference sources; training audio/optimizers are not
    # inputs to this synthetic fixture. The prior audits retain that lineage.
    bindings = {p: h for p, h in long["source_bindings"].items() if Path(p).suffix == ".py"}
    ten_path = Path(read(PHASE / "deployed-vocal-views-001/plan.json")["checkpoint"]["path"])
    fourteen_path = PHASE / "branch-gru-int8-plugin-001/model/model.onnx"
    sixteen_proof_path = PHASE / "branch-output-int8-screen-003/graph.json"
    fourteen_proof_path = PHASE / "branch-gru-int8-screen-002/graph.json"
    paths = [*receipts, POLICY, fourteen_path, sixteen_proof_path, ten_path, fourteen_proof_path, ROOT / "research/direct/latency58_attention_qkv_int8_reference.py", Path(__file__).resolve(), graph_path, parent_path, parent_proof_path,
             short_root / "graph.json", Path(checkpoint["path"])]
    paths.extend(directory / name for directory in (short_root, long_root)
                 for name in ("plan.json", "result.json"))
    paths.extend(ROOT / "research/direct" / name for name in (
        "latency58_m4_followup_budget.py", "latency58_weighted_storage.py", "run_latency58_quality.py", "train_latency58.py"))
    bindings.update({str(p): sha(p) for p in paths})
    require(bindings[str(graph_path)] == short["graph_sha256"]
            and bindings[str(parent_path)] == short["parent_graph_sha256"]
            and bindings[checkpoint["path"]] == checkpoint["sha256"], "Saved inference input changed")
    verify_inputs({"source_bindings": bindings})
    before = snapshot()
    require(args.output_prefix and all(c.isalnum() or c in "-_" for c in args.output_prefix), "Invalid prefix")
    out = FOLLOWUP / args.output_prefix
    require(not out.exists(), "Preserve prior fixture evidence")
    model, payload = load_model(checkpoint)
    fingerprint = state_sha256(model.state_dict())
    require(fingerprint == long["source_model_state_sha256"], "Wrong native source tensors")
    reference, independent = make_reference(model, onnx.load(ten_path),
        read(parent_proof_path)["conversion"], onnx.load(fourteen_path), read(fourteen_proof_path)["conversion"],
        onnx.load(parent_path), read(sixteen_proof_path)["conversion"],
        onnx.load(graph_path), read(short_root / "graph.json")["conversion"])
    require(len(independent) == 17, "Incomplete independently reconstructed projections")
    lengths = (1, 127, 128, 129, 255, 256, 257, 16521)
    t = np.arange(max(lengths), dtype=np.float64) / 44100.
    left = .19 * np.cos(2 * np.pi * 30 * t) + .08 * np.sin(2 * np.pi * 731 * t)
    right = -.14 * np.cos(2 * np.pi * 43 * t) + .07 * np.sin(2 * np.pi * (190 * t + 260 * t*t))
    left[2047:2051] += (.2, -.3, .15, -.1)
    right[8191:8195] += (-.13, .27, -.19, .09)
    audio = np.stack((left, right)).astype("<f4")
    expected_cases, closure = [], 0.
    with torch.inference_mode():
        for length in lengths:
            state = model.initial_state(1)
            padded = torch.zeros(1, 2, ((length + 127) // 128) * 128)
            padded[..., :length] = torch.from_numpy(audio[:, :length].copy())
            hops = []
            for offset in range(0, padded.shape[-1] + 128, 128):
                chunk = padded[..., offset:offset+128] if offset < padded.shape[-1] else torch.zeros(1, 2, 128)
                values = reference(chunk, *state)
                require(all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in values),
                        "Nonfinite output/state or changed public precision")
                state = values[1:]
                if offset:
                    hops.append(values[0])
            expected = torch.cat(hops, dim=-1)[0, ..., :length].numpy().astype("<f4")
            require(expected.shape == (4, 2, length), "Wrong recovered fixture length")
            closure = max(closure, float(np.abs(expected.sum(0) - audio[:, :length]).max()))
            expected_cases.append(expected)
    require(closure <= 1e-6 and state_sha256(model.state_dict()) == fingerprint
            and "onnxruntime" not in sys.modules and not torch.cuda.is_initialized(), "Oracle scope or source changed")
    out.mkdir()
    plan = {"schema": "latency58-seventeen-projection-fixture-plan-v1", "source_bindings": bindings,
            "source_checkpoint": checkpoint, "graph_sha256": short["graph_sha256"],
            "frames": list(lengths), "budget_before": before, "output_allowance_bytes": 2_000_000,
            "native_host_qualified": False, "quality_selected": False}
    write(out / "plan.json", plan)
    binary = out / "cropped1024-pytorch.bin"
    with binary.open("xb") as stream:
        stream.write(b"SGRTG001" + struct.pack("<I", len(lengths)))
        for length, expected in zip(lengths, expected_cases, strict=True):
            stream.write(struct.pack("<I", length))
            stream.write(audio[:, :length].tobytes(order="C"))
            stream.write(expected.tobytes(order="C"))
    # Independently parse the portable byte format and require an exact EOF.
    with binary.open("rb") as stream:
        require(stream.read(8) == b"SGRTG001" and struct.unpack("<I", stream.read(4))[0] == len(lengths),
                "Wrong binary header")
        for length, expected in zip(lengths, expected_cases, strict=True):
            require(struct.unpack("<I", stream.read(4))[0] == length, "Wrong binary case length")
            observed_input = np.frombuffer(stream.read(2 * length * 4), dtype="<f4").reshape(2, length)
            observed_output = np.frombuffer(stream.read(8 * length * 4), dtype="<f4").reshape(4, 2, length)
            require(np.array_equal(observed_input, audio[:, :length]) and np.array_equal(observed_output, expected),
                    "Binary round trip changed native samples")
        require(stream.read() == b"", "Unexpected trailing fixture bytes")
    provenance = {"format": "SGRTG001: LE uint32 case count, then per case LE uint32 frames, planar input[2,T] and deployed[4,2,T] LE float32",
        "fixture_sha256": sha(binary), "generator_sha256": sha(__file__), "plan_sha256": sha(out / "plan.json"),
        "checkpoint_sha256": checkpoint["sha256"], "deployment_graph_sha256": short["graph_sha256"],
        "independent_integer_projections": independent, "state_names": list(model.initial_state(1)._fields),
        "state_shapes": [list(v.shape) for v in model.initial_state(1)], "model_state_sha256": fingerprint,
        "torch": torch.__version__, "numpy": np.__version__, "sample_rate": 44100, "hop": 128,
        "frames": list(lengths), "source_order": ["drums", "bass", "vocals", "other"],
        "reference": "Independent CPU PyTorch declared seventeen-projection signed integer inference, reconstructed from source checkpoint weights. Eight FP32 public states; zero initial states; one zero flush; crop to real length. ONNX parsing authenticates reconstructed weights; runtime imports are actively rejected."}
    write(binary.with_suffix(".json"), provenance)
    verify_inputs(plan)
    require(sum(p.stat().st_size for p in out.iterdir()) + 100_000 < plan["output_allowance_bytes"], "Fixture allowance exceeded")
    write(out / "result.json", {"status": "pass", "plan_sha256": sha(out / "plan.json"),
        "fixture_sha256": sha(binary), "fixture_bytes": binary.stat().st_size,
        "checkpoint": checkpoint, "graph_sha256": short["graph_sha256"], "source_model_state_sha256": fingerprint,
        "source_bindings_unchanged": True, "maximum_reconstruction_error": closure,
        "binary_round_trip_exact": True, "onnxruntime_imported": False, "runtime_imports_actively_blocked": True,
        "budget_before": before, "budget_after": snapshot(), "native_tests_executed": False,
        "quality_selected": False, "native_host_qualified": False})
    print(json.dumps({"status": "pass", "output": str(out), "fixture_sha256": sha(binary),
                      "fixture_bytes": binary.stat().st_size, "maximum_reconstruction_error": closure}))


if __name__ == "__main__":
    main()
