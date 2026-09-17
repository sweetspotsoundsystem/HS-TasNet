#!/usr/bin/env python3
"""Generate the PR fixture from independently reconstructed branch-memory inference.

Requires the research checkpoint and verified graph; never imports ONNX Runtime.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import onnx
    import torch
    from research.direct.run_latency58_quality import PHASE, read, require, sha, write
    from research.direct.train_latency58 import verify_inputs, state_sha256
    from research.direct.latency58_branch_memory_checkpoint import load_model
    from research.direct.latency58_branch_int8 import make_reference
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    screen_root = args.screen.resolve()
    screen, plan = read(screen_root / "result.json"), read(screen_root / "plan.json")
    require(screen["status"] == "pass" and read(screen_root / "execution.json")["actual_exit_code"] == 0,
            "Require completed short parity")
    graph_path = screen_root / "model.onnx"
    require(sha(graph_path) == screen["graph_sha256"], "Graph changed")
    model, payload = load_model(screen["checkpoint"])
    graph = onnx.load(graph_path)
    reference, independent = make_reference(model, graph, read(screen_root / "graph.json")["conversion"])
    require("onnxruntime" not in sys.modules, "Never use ONNX Runtime for the fixture oracle")
    verify_inputs(plan)
    out = args.output.resolve()
    require(out.parent == PHASE and not out.exists(), "Preserve existing fixture evidence")
    out.mkdir()
    output = out / "cropped1024-pytorch.bin"
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
        "fixture_sha256": sha(output), "generator_sha256": sha(__file__),
        "selection_review_sha256": sha(plan["selection_review"]),
        "checkpoint_sha256": screen["checkpoint"]["sha256"],
        "source_checkpoint_quality_result_sha256": sha(plan["quality_result"]),
        "deployment_graph_sha256": screen["graph_sha256"],
        "independent_integer_projections": independent,
        "state_names": list(model.initial_state(1)._fields),
        "state_shapes": [list(v.shape) for v in model.initial_state(1)],
        "model_state_sha256": payload["model_state_sha256"],
        "torch": torch.__version__, "numpy": np.__version__,
        "sample_rate": 44100, "hop": 128, "frames": lengths,
        "source_order": ["drums", "bass", "vocals", "other"],
        "reference": "Independent CPU PyTorch declared signed integer inference, reconstructed from source checkpoint weights. Eight float32 public states; zero initial states; one zero flush; crop to real length. ONNX parsing only authenticates independently reconstructed weights. No ONNX Runtime import or execution.",
    }
    write(output.with_suffix(".json"), provenance)
    require(state_sha256(model.state_dict()) == payload["model_state_sha256"]
            and "onnxruntime" not in sys.modules and not torch.cuda.is_initialized(), "Oracle scope changed")
    write(out / "result.json", {"status": "pass", "fixture_sha256": sha(output),
          "fixture_bytes": output.stat().st_size, "checkpoint": screen["checkpoint"],
          "graph_sha256": screen["graph_sha256"], "onnxruntime_imported": False,
          "source_bindings_unchanged": True})
    print(json.dumps({"status": "pass", "fixture_sha256": sha(output), "output": str(out)}), flush=True)


if __name__ == "__main__":
    main()
