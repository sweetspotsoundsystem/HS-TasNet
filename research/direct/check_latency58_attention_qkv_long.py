"""Long carried-state checks of the saved seventeen-product QKV candidate."""
from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import time

from research.direct.latency58_m4_followup_budget import ROOT, POLICY, snapshot
from research.direct.run_latency58_quality import read, require, sha, write

PHASE = ROOT / "research/direct/runs/latency58"
FOLLOWUP = ROOT / "research/m4_followup_20260916"


def main():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and all(os.environ.get(k) == "1"
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CPU1")
    import onnx
    import onnxruntime as ort
    import torch
    from research.direct.latency58_branch_memory_checkpoint import load_model
    from research.direct.latency58_attention_qkv_int8_reference import make_reference
    from research.direct.latency58_branch_onnx import interface, TOLERANCES
    from research.direct.latency58_branch_int8_verify import session_for, run_case
    from research.direct.check_latency58_branch_int8_plugin import cases
    from research.direct.train_latency58 import verify_inputs, state_sha256

    require(ort.__version__ == "1.26.0", "Require shipping ORT version")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    before = snapshot()
    out = FOLLOWUP / "attention-qkv-int8-long-001"
    require(not out.exists(), "Preserve earlier long checks")
    short_root = FOLLOWUP / "attention-qkv-int8-screen-001"
    short = read(short_root / "result.json")
    execution = read(FOLLOWUP / "attention-qkv-int8-execution.json")
    native_root = FOLLOWUP / "attention-qkv-native-001"
    native = read(native_root / "result.json")
    native_execution = read(FOLLOWUP / "attention-qkv-native-execution.json")
    require(short["status"] == "pass" and short["strict_parity_passed"] and short["local_speed_gate"]["passed"]
            and execution["actual_exit_code"] == 0 and not execution["timed_out"], "Short screen incomplete")
    require(native["status"] == "pass" and native["relative_speed_gate"]["passed"]
            and native_execution["actual_exit_code"] == 0 and not native_execution["timed_out"]
            and native["plan_sha256"] == sha(native_root / "plan.json"), "Native comparison incomplete")
    verify_inputs(short)
    verify_inputs(read(native_root / "plan.json"))
    baseline_path = PHASE / "deployed-vocal-views-001/plan.json"
    baseline = read(baseline_path)
    checkpoint = short["source_checkpoint"]
    graph_path = Path(short["saved_graph_path"])
    ten_path = Path(baseline["checkpoint"]["path"])
    fourteen_path = PHASE / "branch-gru-int8-plugin-001/model/model.onnx"
    parent_path = PHASE / "m4-inference-diagnostics-plugin-001/model/model.onnx"
    ten_proof_path = PHASE / "branch-plugin-screen-001/graph.json"
    fourteen_proof_path = PHASE / "branch-gru-int8-screen-002/graph.json"
    sixteen_proof_path = PHASE / "branch-output-int8-screen-003/graph.json"
    bindings = dict(short["source_bindings"])
    paths = [Path(__file__).resolve(), POLICY, baseline_path, graph_path, ten_path,
        fourteen_path, parent_path, ten_proof_path, fourteen_proof_path, sixteen_proof_path,
        short_root / "result.json", short_root / "graph.json", FOLLOWUP / "attention-qkv-int8-execution.json",
        native_root / "plan.json", native_root / "result.json", FOLLOWUP / "attention-qkv-native-execution.json"]
    bindings.update({str(p): sha(p) for p in paths})
    require(sha(graph_path) == short["graph_sha256"] and sha(parent_path) == short["parent_graph_sha256"],
            "Saved graphs changed")
    model, payload = load_model(checkpoint)
    fingerprint = state_sha256(model.state_dict())
    require(fingerprint == short["source_model_state_sha256"], "Saved source changed")
    reference, independent = make_reference(model, onnx.load(ten_path), read(ten_proof_path)["conversion"],
        onnx.load(fourteen_path), read(fourteen_proof_path)["conversion"], onnx.load(parent_path),
        read(sixteen_proof_path)["conversion"], onnx.load(graph_path), read(short_root / "graph.json")["conversion"])
    contract = interface(model)
    require(json.loads(json.dumps(contract)) == baseline["interface"], "Public interface changed")
    out.mkdir()
    write(out / "plan.json", {"source_bindings": bindings, "source_checkpoint": checkpoint,
        "saved_graph_path": str(graph_path), "graph_sha256": sha(graph_path), "interface": contract,
        "tolerances": TOLERANCES, "signal_hops": 2048, "replays": 2,
        "budget_before": before, "independent_reference_projections": independent,
        "validation_music_used": False, "native_host_qualified": False})
    session = session_for(graph_path.read_bytes(), contract)
    fixtures = list(cases(model, hops=2048))
    fixtures = [fixtures[0], {**fixtures[4], "name": "nonzero_all_states_long_dc_nyquist_signal",
                             "audio": fixtures[2]["audio"]}]
    began, reports = time.monotonic(), []
    for case in fixtures:
        first = run_case(reference, session, case, contract)
        write(out / (case["name"] + "-first.json"), first)
        print(json.dumps({"case": case["name"], "replay": 1, "passed": first["passed"],
            "maximum_errors": first["maximum_errors_in_physical_units"]}), flush=True)
        second = run_case(reference, session, case, contract)
        write(out / (case["name"] + "-second.json"), second)
        exact = first["all_output_and_state_trajectory_sha256"] == second["all_output_and_state_trajectory_sha256"]
        reports.append({"case": case["name"], "first": first, "second": second,
            "reset_replay_bit_exact": exact, "passed": first["passed"] and second["passed"] and exact})
        print(json.dumps({"case": case["name"], "replay": 2, "passed": reports[-1]["passed"],
            "reset_replay_bit_exact": exact}), flush=True)
    verify_inputs({"source_bindings": bindings})
    require(state_sha256(model.state_dict()) == fingerprint == payload["model_state_sha256"]
            and not torch.cuda.is_initialized(), "Native source or CPU scope changed")
    passed = all(r["passed"] for r in reports)
    write(out / "result.json", {"status": "pass" if passed else "fail", "reports": reports,
        "graph_sha256": sha(graph_path), "saved_graph_path": str(graph_path),
        "source_checkpoint": checkpoint, "source_model_state_sha256": fingerprint,
        "source_bindings": bindings, "source_bindings_unchanged": True, "runtime": ort.__version__,
        "tolerances_changed": False, "quality_measured": False, "native_host_qualified": False,
        "gpu_used": False, "graph_plus_host_delay_samples": 256,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "elapsed_seconds": time.monotonic() - began, "budget_before": before, "budget_after": snapshot()})
    print(json.dumps({"status": "pass" if passed else "fail"}), flush=True)
    require(passed, "Long independent numerical parity or reset replay failed")


if __name__ == "__main__":
    main()
