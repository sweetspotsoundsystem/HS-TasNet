"""Thirty-second independent recurrence and reset replay for the saved PR graph."""
import argparse
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import onnx
    import onnxruntime as ort
    import soundfile as sf
    import torch
    from research.direct.latency58_branch_memory_checkpoint import load_model
    from research.direct.latency58_branch_int8 import make_reference
    from research.direct.latency58_branch_int8_verify import session_for, run_case
    from research.direct.latency58_branch_onnx import interface, TOLERANCES
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and ort.__version__ == "1.26.0" and all(os.environ.get(k) == "1" for k in
                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CPU1 and shipping ORT")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    screen_root = args.screen.resolve()
    screen, plan = read(screen_root / "result.json"), read(screen_root / "plan.json")
    require(screen["status"] == "pass" and read(screen_root / "execution.json")["actual_exit_code"] == 0,
            "Complete the actual short parity execution")
    graph_path = screen_root / "model.onnx"
    require(sha(graph_path) == screen["graph_sha256"], "Saved graph changed")
    native, payload = load_model(screen["checkpoint"])
    require(payload["model_state_sha256"] == screen["model_state_sha256"], "Source model changed")
    graph = onnx.load(graph_path)
    reference, independent = make_reference(native, graph, read(screen_root / "graph.json")["conversion"])
    contract = interface(native)
    session = session_for(graph.SerializeToString(), contract)
    music = Path("/home/axel/HS-TasNet/data/musdb18hq/train/Actions - One Minute Smile/mixture.wav")
    audio, rate = sf.read(music, frames=30 * 44100 + 37, always_2d=True, dtype="float32")
    require(rate == 44100 and audio.shape == (30 * 44100 + 37, 2), "Long audio source changed")
    paths = [Path(__file__).resolve(), music, graph_path, screen_root / "graph.json", screen_root / "plan.json",
             screen_root / "result.json", screen_root / "execution.json"]
    bindings = {**plan["source_bindings"], **{str(p): sha(p) for p in paths}}
    verify_inputs({"source_bindings": bindings})
    budget = read(Path(plan["quality_result"]).parents[1] / "plan.json")
    require_space(budget, 600_000_000 + 5_000_000)
    out = args.output.resolve()
    require(out.parent == PHASE and not out.exists(), "Preserve long parity evidence")
    out.mkdir()
    write(out / "plan.json", {"source_bindings": bindings, "checkpoint": screen["checkpoint"],
          "graph_sha256": screen["graph_sha256"], "tolerances": TOLERANCES,
          "physical_samples": len(audio), "repetitions": 2,
          "reference": "Independent declared signed integer inference; no ORT-generated oracle",
          "parity_to_unquantized_fp32_source_claimed": False, "graph_saved": True})
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
    require(state_sha256(native.state_dict()) == payload["model_state_sha256"]
            and not torch.cuda.is_initialized(), "Source changed or GPU used")
    passed = exact and all(row["passed"] for row in rows)
    write(out / "result.json", {"status": "pass" if passed else "fail", "repetitions": rows,
          "reset_replay_all_outputs_and_states_bit_exact": exact, "checkpoint": screen["checkpoint"],
          "graph_sha256": screen["graph_sha256"], "graph_saved": True, "tolerances_changed": False,
          "source_bindings_unchanged": True, "parity_to_unquantized_fp32_source_claimed": False,
          "native_host_qualified": False, "elapsed_seconds": time.monotonic() - began,
          "counted_bytes_after": require_space(budget, 600_000_000)})
    require(passed, "Long integer recurrence failed")


if __name__ == "__main__":
    main()
