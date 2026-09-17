"""CPU weight-spectrum diagnostic for the preserved C204 inference graph.

This measures optimal matrix approximation error, not separation quality or
runtime. It saves no transformed weights and never uses validation audio.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs


def spectrum_report(matrix):
    import numpy as np
    import torch

    require(matrix.ndim == 2 and matrix.dtype == np.float32
            and np.isfinite(matrix).all(), "Require a finite FP32 matrix")
    rows, columns = matrix.shape
    singular = torch.linalg.svdvals(torch.from_numpy(matrix.copy()).double()).numpy()
    energy = singular ** 2
    total = float(np.sum(matrix.astype(np.float64) ** 2))
    require(total > 0 and np.isfinite(singular).all()
            and np.all(singular[:-1] >= singular[1:])
            and np.isclose(energy.sum(), total, rtol=1e-10), "Invalid singular spectrum")
    captured = np.cumsum(energy) / total
    ranks = sorted(set(min(rank, len(singular)) for rank in
                       (64, 128, 256, 384, 512, 768, 1000, 1536, len(singular))))
    return {
        "shape": [rows, columns], "elements": rows * columns,
        "weight_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
        "frobenius_energy": total, "singular_values": singular.tolist(),
        "relative_energy_check_error": abs(float(energy.sum()) - total) / total,
        "rank_for_energy_fraction": {str(fraction): int(np.searchsorted(captured, fraction) + 1)
                                     for fraction in (.9, .95, .99, .999)},
        "rank_screens": [{"rank": rank,
                          "relative_frobenius_error": float(np.sqrt(max(0., 1 - captured[rank - 1]))),
                          "factor_elements": rank * (rows + columns),
                          "factor_element_ratio": rank * (rows + columns) / (rows * columns)}
                         for rank in ranks],
    }


def main():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CPU1 with CUDA hidden")
    import numpy as np
    import onnx
    from onnx import numpy_helper
    import torch
    from research.direct.latency58_sdr_checkpoint import require_space

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source_path = PHASE / "full-magnitude-sdr-001/plan.json"
    source = read(source_path)
    verify_inputs(source)
    counted_before = require_space(source, 371_000_000)
    graph_path = Path("/home/axel/autoresearch/codex/stemgen-rt-hop128-5ms/model/model.onnx")
    require(sha(graph_path) == "b8574ac2e67bcd1df533e3fc7464c0659d4bfa6744cfbb4389c0967271594fe3",
            "Preserved C204 graph changed")
    out = PHASE / "m4-matrix-ranks-001"
    require(not out.exists(), "Preserve completed diagnostics")
    bindings = {**source["source_bindings"], str(source_path): sha(source_path),
                str(graph_path): sha(graph_path), str(Path(__file__).resolve()): sha(__file__)}
    out.mkdir()
    write(out / "plan.json", {"source_bindings": bindings, "counted_bytes_before": counted_before,
          "pending_training_save_reserved_bytes": 370_000_000, "cpu_threads": 1,
          "matrix_svd_dtype": "float64", "training_or_validation_audio_used": False,
          "transformed_weights_saved": False, "quality_or_timing_measured": False})
    graph = onnx.load(graph_path, load_external_data=False)
    require(not any(t.external_data for t in graph.graph.initializer), "Require self-contained graph")
    initializers = {t.name: numpy_helper.to_array(t) for t in graph.graph.initializer}
    selected = {
        "conv_encode": ("model.conv_encode.weight", (3000, 2048)),
        "basis_to_embed": ("model.basis_to_embed.weight", (500, 1500)),
        "spec_encode": ("onnx::MatMul_362", (2052, 500)),
        "to_spec_masks": ("onnx::MatMul_363", (500, 8208)),
        "to_waveform_masks": ("onnx::MatMul_376", (500, 6000)),
        **{f"gru_{kind}_l{layer}": (f"model.fusion_branch.weight_{kind}_l{layer}", (3000, 1000))
           for layer in range(2) for kind in ("ih", "hh")},
    }
    require(all(name in initializers and initializers[name].size == int(np.prod(shape))
                for name, shape in selected.values()), "Projection inventory changed")
    began, results = time.monotonic(), []
    with (out / "progress.jsonl").open("x", buffering=1) as journal:
        for label, (name, shape) in selected.items():
            matrix = np.ascontiguousarray(initializers[name].reshape(shape))
            row = {"projection": label, "initializer": name, **spectrum_report(matrix),
                   "elapsed_seconds": time.monotonic() - began}
            results.append(row)
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps({k: row[k] for k in ("projection", "shape", "rank_for_energy_fraction", "elapsed_seconds")}), flush=True)
    require(not torch.cuda.is_initialized(), "Diagnostic initialized CUDA")
    verify_inputs({"source_bindings": bindings})
    result = {"status": "complete", "projections": results,
              "source_bindings_unchanged": True, "gpu_used": False,
              "quality_or_timing_measured": False, "transformed_weights_saved": False,
              "elapsed_seconds": time.monotonic() - began,
              "counted_bytes_after": require_space(source, 370_000_000),
              "limitation": "Weight spectra alone do not establish activation error, recurrent stability, separation quality, or M4 speed. A factorization must have fewer total elements to reduce matrix traffic."}
    write(out / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "projections"}), flush=True)


if __name__ == "__main__":
    main()
