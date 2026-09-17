"""Thirty-second recurrence and exact reset replay for the declared integer variant."""
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
    from research.direct.check_latency58_best_onnx_memory import selected_endpoint
    from research.direct.latency58_attention_int8 import build, make_reference
    from research.direct.latency58_attention_int8_verify import session_for, run_case
    from research.direct.latency58_best_onnx import interface, TOLERANCES
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and ort.__version__ == "1.26.0" and all(os.environ.get(k) == "1" for k in
                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CPU1 and shipping ORT")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    budget = read(PHASE / "temporal-attention-001/plan.json")
    require_space(budget, 5_000_000)
    screen_root = PHASE / "attention-int8-screen-001"
    screen, plan = read(screen_root / "result.json"), read(screen_root / "plan.json")
    execution_path = PHASE / "attention-int8-screen-stage-001/execution.json"
    execution = read(execution_path)
    require(screen["status"] == "pass" and screen["strict_parity_passed"] and screen["source_bindings_unchanged"]
            and execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
            and not execution["timed_out"], "Short parity must close successfully")
    verify_inputs(plan)
    native, payload, training, checkpoint, quality_path, _ = selected_endpoint()
    require(checkpoint == screen["checkpoint"], "Selected checkpoint changed")
    fingerprint, contract = state_sha256(native.state_dict()), interface(native)
    integer, graph, conversion = build(native, payload, training, checkpoint, quality_path)
    reference, _ = make_reference(native, integer, conversion)
    data = graph.SerializeToString()
    require(hashlib.sha256(data).hexdigest() == screen["graph_sha256"], "Rebuilt graph differs")
    del graph, integer
    session = session_for(data, contract)
    music = Path("/home/axel/HS-TasNet/data/musdb18hq/train/Actions - One Minute Smile/mixture.wav")
    audio, rate = sf.read(music, frames=30 * 44100 + 37, always_2d=True, dtype="float32")
    require(rate == 44100 and audio.shape == (30 * 44100 + 37, 2), "Long source changed")
    paths = [Path(__file__).resolve(), music, screen_root / "plan.json", screen_root / "result.json", execution_path]
    bindings = {**plan["source_bindings"], **{str(p): sha(p) for p in paths}}
    verify_inputs({"source_bindings": bindings})
    out = PHASE / "attention-int8-long-001"
    require(not out.exists(), "Preserve long tests")
    out.mkdir()
    write(out / "plan.json", {"source_bindings": bindings, "checkpoint": checkpoint,
          "graph_sha256": screen["graph_sha256"], "tolerances": TOLERANCES,
          "physical_samples": len(audio), "repetitions": 2,
          "reference": "Independent declared signed-integer inference; no ORT-generated oracle",
          "parity_to_unquantized_fp32_source_claimed": False, "graph_saved": False})
    case = {"name": "recorded_music_30_seconds_partial", "audio": np.ascontiguousarray(audio.T),
            "initial_states": [np.zeros(shape, np.float32) for shape in contract["state_shapes"]],
            "source": {"path": str(music), "sha256": sha(music)}}
    rows, began = [], time.monotonic()
    for repeat in range(2):
        row = run_case(reference, session, case, contract)
        rows.append(row)
        write(out / f"repeat-{repeat + 1}.json", row)
        print(json.dumps({"repeat": repeat + 1, "passed": row["passed"],
                          "errors": row["maximum_errors_in_physical_units"]}), flush=True)
    exact = rows[0]["all_output_and_state_trajectory_sha256"] == rows[1]["all_output_and_state_trajectory_sha256"]
    verify_inputs({"source_bindings": bindings})
    require(state_sha256(native.state_dict()) == fingerprint and not torch.cuda.is_initialized(), "Source changed")
    passed = exact and all(row["passed"] for row in rows)
    write(out / "result.json", {"status": "pass" if passed else "fail", "repetitions": rows,
          "reset_replay_all_outputs_and_states_bit_exact": exact, "checkpoint": checkpoint,
          "graph_sha256": screen["graph_sha256"], "graph_saved": False, "tolerances_changed": False,
          "source_bindings_unchanged": True, "parity_to_unquantized_fp32_source_claimed": False,
          "native_host_qualified": False, "elapsed_seconds": time.monotonic() - began,
          "counted_bytes_after": require_space(budget, 0)})
    require(passed, "Long integer recurrence failed")


if __name__ == "__main__":
    main()
