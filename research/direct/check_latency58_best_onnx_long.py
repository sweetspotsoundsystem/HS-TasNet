"""Check 30 seconds of native/export/ORT recurrence from the selected model."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs


def main():
    import numpy as np
    import onnxruntime as ort
    import soundfile as sf
    import torch
    from research.direct.latency58_best_onnx_export import build
    from research.direct.latency58_best_onnx import interface
    from research.direct.latency58_best_onnx_verify import _run_case
    from research.direct.check_latency58_best_onnx_memory import selected_endpoint
    from research.direct.check_latency58_fused_gru import session_for
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "" and ort.__version__ == "1.26.0"
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require CPU1 and shipping ORT")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    budget = read(PHASE / "temporal-attention-001/plan.json")
    require_space(budget, 5_000_000)
    screen_root = PHASE / "best-model-onnx-memory-001"
    screen, screen_plan = read(screen_root / "result.json"), read(screen_root / "plan.json")
    execution_path = PHASE / "best-model-onnx-memory-stage-001/execution.json"
    execution = read(execution_path)
    require(screen["status"] == "pass" and screen["source_bindings_unchanged"]
            and execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
            and not execution["timed_out"], "Require the completed raw and optimized export checks")
    verify_inputs(screen_plan)
    model, payload, training, checkpoint, quality_path, _ = selected_endpoint()
    fingerprint = state_sha256(model.state_dict())
    wrapper, graph = build(model, payload, training, checkpoint, quality_path)
    data = graph.SerializeToString()
    require(hashlib.sha256(data).hexdigest() == screen["graph_sha256"], "Rebuilt graph differs")
    del graph
    session = session_for(data)
    path = Path("/home/axel/HS-TasNet/data/musdb18hq/train/Actions - One Minute Smile/mixture.wav")
    audio, rate = sf.read(path, frames=30 * 44100 + 37, always_2d=True, dtype="float32")
    require(rate == 44100 and audio.shape == (30 * 44100 + 37, 2), "Music fixture changed")
    paths = [Path(__file__).resolve(), path, screen_root / "result.json", screen_root / "plan.json", execution_path,
             ROOT / "research/direct/check_latency58_fused_gru.py"]
    bindings = {**screen_plan["source_bindings"], **{str(p): sha(p) for p in paths}}
    verify_inputs({"source_bindings": bindings})
    out = PHASE / "best-model-onnx-long-001"
    require(not out.exists(), "Preserve long checks")
    out.mkdir()
    write(out / "plan.json", {"source_bindings": bindings, "checkpoint": checkpoint,
          "graph_sha256": screen["graph_sha256"], "samples": len(audio), "repetitions": 2,
          "graph_saved": False, "native_host_qualified": False})
    case = {"name": "recorded_music_30_seconds_partial", "audio": np.ascontiguousarray(audio.T),
            "initial_states": [np.zeros(shape, np.float32) for shape in interface(model)["state_shapes"]],
            "source": {"path": str(path), "sha256": sha(path)}}
    began, rows = time.monotonic(), []
    for repeat in range(2):
        row = _run_case(model, wrapper, session, case)
        rows.append(row)
        write(out / f"repeat-{repeat + 1}.json", row)
        print(json.dumps({"repeat": repeat + 1, "passed": row["passed"],
                          "native_vs_ort": row["comparisons"]["native_vs_ort"]}), flush=True)
    exact = rows[0]["all_output_and_state_trajectory_sha256"] == rows[1]["all_output_and_state_trajectory_sha256"]
    verify_inputs({"source_bindings": bindings})
    require(state_sha256(model.state_dict()) == fingerprint and not torch.cuda.is_initialized(), "Native model changed")
    passed = exact and all(row["passed"] for row in rows)
    write(out / "result.json", {"status": "pass" if passed else "fail", "repetitions": rows,
          "reset_replay_all_outputs_and_states_bit_exact": exact, "checkpoint": checkpoint,
          "graph_sha256": screen["graph_sha256"], "graph_saved": False,
          "source_bindings_unchanged": True, "native_host_qualified": False,
          "elapsed_seconds": time.monotonic() - began, "counted_bytes_after": require_space(budget, 0)})
    require(passed, "Long selected-model recurrence failed")


if __name__ == "__main__":
    main()
