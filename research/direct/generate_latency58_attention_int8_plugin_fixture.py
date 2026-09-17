#!/usr/bin/env python3
"""Regenerate the synthetic oracle from independently reconstructed signed inference.

Requires the HS-TasNet research checkout with its retained checkpoint/artifacts.
Run on CPU with numpy, PyTorch and ONNX parsing; this never imports or executes ONNX Runtime.
The checked-in fixture allows ordinary plugin CI to validate without PyTorch.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
    from research.direct.train_latency58 import verify_inputs, state_sha256
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT, "Use the reviewed research workspace")
    require_space(read(PHASE / "temporal-attention-001/plan.json"), 2_000_000)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    from research.direct.check_latency58_best_onnx_memory import selected_endpoint
    model, payload, training, checkpoint, quality_path, review_path = selected_endpoint()
    require("onnxruntime" not in sys.modules, "The fixture oracle must not execute ONNX Runtime")
    import onnx
    from research.direct.latency58_attention_int8 import make_reference
    saved_root = PHASE / "best-model-onnx-saved-001"
    saved = read(saved_root / "result.json")
    graph_path = Path(saved["onnx"]["path"])
    require(saved["status"] == "pass" and saved["checkpoint"] == checkpoint
            and sha(graph_path) == saved["onnx"]["sha256"], "Require the checked saved integer graph")
    screen_graph_path = PHASE / "attention-int8-screen-001/graph.json"
    screen_graph = read(screen_graph_path)
    require(screen_graph["graph_sha256"] == saved["onnx"]["sha256"], "Oracle graph differs from checked integer model")
    graph = onnx.load(graph_path)
    reference, independent = make_reference(model, graph, screen_graph["conversion"])
    deployment_quality_path = PHASE / "attention-int8-full14-memory-001/result.json"
    deployment_quality = read(deployment_quality_path)
    require(deployment_quality["status"] == "pass" and deployment_quality["quality_handoff_gate_passed"]
            and deployment_quality["graph_sha256"] == saved["onnx"]["sha256"], "Require the deployment quality gate")
    require("onnxruntime" not in sys.modules, "Independent integer oracle must not import ORT")
    model_state_sha = state_sha256(model.state_dict())
    bindings = {**training["source_bindings"], str(Path(__file__).resolve()): sha(__file__),
                str(review_path): sha(review_path), str(quality_path): sha(quality_path),
                checkpoint["path"]: checkpoint["sha256"]}
    paths = [saved_root / "plan.json", saved_root / "result.json", graph_path, screen_graph_path,
             deployment_quality_path, ROOT / "research/direct/latency58_attention_int8.py",
             ROOT / "research/direct/latency58_int8_precise_float.py", ROOT / "research/direct/latency58_int8_reference.py",
             ROOT / "research/direct/latency58_best_onnx.py", ROOT / "export_onnx.py"]
    bindings.update({str(p): sha(p) for p in paths})
    verify_inputs({"source_bindings": bindings})
    out = PHASE / "attention-int8-plugin-fixture-001"
    require(not out.exists(), "Preserve fixture evidence")
    out.mkdir()
    output = out / "cropped1024-pytorch.bin"
    write(out / "plan.json", {"source_bindings": bindings, "checkpoint": checkpoint,
          "oracle": "Independent NumPy signed reconstruction and PyTorch int32 products; declared FP64 quantizer ancestors and FP32 public values; six persistent states; no ORT execution"})
    lengths = (1, 127, 128, 129, 255, 256, 257, 16521)
    t = np.arange(max(lengths), dtype=np.float64) / 44100.0
    # Distinct channels, low bass, higher partials, chirp and short transients.
    # The initial peak opens the existing writer confidence envelope fully.
    left = 0.19 * np.cos(2 * np.pi * 30 * t) + 0.08 * np.sin(2 * np.pi * 731 * t)
    right = -0.14 * np.cos(2 * np.pi * 43 * t) + 0.07 * np.sin(2 * np.pi * (190 * t + 260 * t*t))
    left[2047:2051] += (0.2, -0.3, 0.15, -0.1)
    right[8191:8195] += (-0.13, 0.27, -0.19, 0.09)
    audio = np.stack((left, right)).astype("<f4")
    with output.open("xb") as stream, torch.inference_mode():
        stream.write(b"SGRTG001" + struct.pack("<I", len(lengths)))
        for length in lengths:
            state = model.initial_state(1)
            hops = []
            padded = torch.zeros(1, 2, ((length + 127) // 128) * 128)
            padded[..., :length] = torch.from_numpy(audio[:, :length].copy())
            for offset in range(0, padded.shape[-1], 128):
                values = reference(padded[..., offset:offset+128], *state)
                deployed, state = values[0], values[1:]
                if offset:
                    hops.append(deployed)
            values = reference(torch.zeros(1, 2, 128), *state)
            deployed, state = values[0], values[1:]
            hops.append(deployed)
            expected = torch.cat(hops, dim=-1)[0, ..., :length].numpy().astype("<f4")
            stream.write(struct.pack("<I", length))
            stream.write(audio[:, :length].tobytes(order="C"))
            stream.write(expected.tobytes(order="C"))
    provenance = {
        "format": "SGRTG001: LE uint32 case count, then per case LE uint32 frames, planar input[2,T] and deployed[4,2,T] LE float32",
        "fixture_sha256": digest(output),
        "generator_sha256": digest(Path(__file__)),
        "selection_review_sha256": sha(review_path),
        "checkpoint_sha256": checkpoint["sha256"],
        "source_checkpoint_quality_result_sha256": sha(quality_path),
        "deployment_quality_result_sha256": sha(deployment_quality_path),
        "deployment_graph_sha256": saved["onnx"]["sha256"],
        "independent_integer_projections": independent,
        "state_names": list(model.initial_state(1)._fields),
        "state_shapes": [list(value.shape) for value in model.initial_state(1)],
        "model_state_sha256": model_state_sha,
        "torch": torch.__version__, "numpy": np.__version__,
        "sample_rate": 44100, "hop": 128, "frames": lengths,
        "source_order": ["drums", "bass", "vocals", "other"],
        "reference": "Independent CPU PyTorch declared integer inference, reconstructed from source checkpoint weights; float32 public output/state, zero initial state, one zero flush, cropped to real length. ONNX is parsed only to authenticate all independently reconstructed weight bytes; no ONNX Runtime import or execution.",
    }
    output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    verify_inputs({"source_bindings": bindings})
    require(state_sha256(model.state_dict()) == model_state_sha and "onnxruntime" not in sys.modules
            and not torch.cuda.is_initialized(), "Fixture source or CPU oracle changed")
    write(out / "result.json", {"status": "pass", "source_bindings_unchanged": True,
          "fixture_sha256": digest(output), "fixture_bytes": output.stat().st_size,
          "checkpoint": checkpoint, "onnxruntime_imported": False,
          "counted_bytes_after": require_space(read(PHASE / "temporal-attention-001/plan.json"), 0)})
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
