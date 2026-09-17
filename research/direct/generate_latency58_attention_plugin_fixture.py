#!/usr/bin/env python3
"""Regenerate the synthetic oracle from the authenticated research checkpoint.

Requires the HS-TasNet research checkout with its retained checkpoint/artifacts.
Run on CPU with numpy and PyTorch; this never imports or executes ONNX Runtime.
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
    model_state_sha = state_sha256(model.state_dict())
    bindings = {**training["source_bindings"], str(Path(__file__).resolve()): sha(__file__),
                str(review_path): sha(review_path), str(quality_path): sha(quality_path),
                checkpoint["path"]: checkpoint["sha256"]}
    verify_inputs({"source_bindings": bindings})
    out = PHASE / "attention-plugin-fixture-001"
    require(not out.exists(), "Preserve fixture evidence")
    out.mkdir()
    output = out / "cropped1024-pytorch.bin"
    write(out / "plan.json", {"source_bindings": bindings, "checkpoint": checkpoint,
          "oracle": "Native CPU FP32 PyTorch with six independent persistent states; no export-copy or ORT execution"})
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
                deployed, state = model.forward_chunk(padded[..., offset:offset+128], state)
                if offset:
                    hops.append(deployed)
            deployed, state = model.flush(state)
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
        "quality_result_sha256": sha(quality_path),
        "state_names": list(model.initial_state(1)._fields),
        "state_shapes": [list(value.shape) for value in model.initial_state(1)],
        "model_state_sha256": model_state_sha,
        "torch": torch.__version__, "numpy": np.__version__,
        "sample_rate": 44100, "hop": 128, "frames": lengths,
        "source_order": ["drums", "bass", "vocals", "other"],
        "reference": "Original CPU FP32 PyTorch deployed outputs, zero state, one zero flush, cropped to real length. No ONNX execution.",
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
