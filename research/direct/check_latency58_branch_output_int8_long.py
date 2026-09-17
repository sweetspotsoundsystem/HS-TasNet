"""Long carried-state checks of the saved sixteen-projection experiment."""
from __future__ import annotations

from pathlib import Path
import json
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs, state_sha256


def main():
    import onnx
    import onnxruntime as ort
    import torch
    from research.direct.run_latency58_deployed_vocal_views import require_cpu, budget_snapshot
    from research.direct.latency58_branch_memory_checkpoint import load_model
    from research.direct.latency58_branch_output_int8_reference import make_reference
    from research.direct.latency58_branch_onnx import interface, TOLERANCES
    from research.direct.latency58_branch_int8_verify import session_for, run_case
    from research.direct.check_latency58_branch_int8_plugin import cases
    baseline_path = PHASE / "deployed-vocal-views-001/plan.json"
    baseline = read(baseline_path)
    require_cpu(baseline)
    budget = {**baseline, "diagnostic_artifact_allowance_bytes": 250_000_000}
    before = budget_snapshot(budget)
    short_root = PHASE / "branch-output-int8-screen-003"
    short, execution = read(short_root / "result.json"), read(short_root / "execution.json")
    require(short["status"] == "pass" and short["strict_parity_passed"]
            and short["source_bindings_unchanged"] and execution["actual_exit_code"] == 0
            and not execution["timed_out"], "Complete the actual short numerical screen first")
    checkpoint = short["source_checkpoint"]
    graph_path = Path(short["saved_graph_path"])
    parent_path = Path(short["parent_graph"]["path"])
    parent_proof_path = PHASE / "branch-plugin-screen-001/graph.json"
    # This synthetic inference check consumes the saved tensors, graphs and
    # code. Training audio/optimizer files remain authenticated by the completed
    # short screen; they are not inputs to this new numerical computation.
    bindings = {p: h for p, h in short["source_bindings"].items() if Path(p).suffix == ".py"}
    ten_path = Path(baseline["checkpoint"]["path"])
    fourteen_proof_path = PHASE / "branch-gru-int8-screen-002/graph.json"
    paths = [ten_path, fourteen_proof_path, ROOT / "research/direct/latency58_branch_output_int8_reference.py", Path(__file__).resolve(), baseline_path, short_root / "result.json",
        short_root / "execution.json", short_root / "plan.json", short_root / "graph.json",
        graph_path, parent_path, parent_proof_path, Path(checkpoint["path"]), Path(ort.__file__).resolve(),
        *sorted((Path(ort.__file__).resolve().parent / "capi").glob("*.so*"))]
    bindings.update({str(p): sha(p) for p in paths})
    require(bindings[str(graph_path)] == short["graph_sha256"]
            and bindings[str(parent_path)] == short["parent_graph"]["sha256"]
            and bindings[checkpoint["path"]] == checkpoint["sha256"], "Saved inference inputs changed")
    verify_inputs({"source_bindings": bindings})
    model, payload = load_model(checkpoint)
    fingerprint = state_sha256(model.state_dict())
    parent_proof = read(parent_proof_path)
    require(fingerprint == parent_proof["conversion"]["source_model_state_sha256"], "Native source tensor identity changed")
    graph, parent = onnx.load(graph_path), onnx.load(parent_path)
    reference, independent = make_reference(model, onnx.load(ten_path), parent_proof["conversion"], parent, read(fourteen_proof_path)["conversion"], graph,
                                            read(short_root / "graph.json")["conversion"])
    contract = interface(model)
    require(json.loads(json.dumps(contract)) == baseline["interface"], "Saved public interface differs")
    out = PHASE / "branch-output-int8-long-001"
    require(not out.exists(), "Preserve previous long numerical evidence")
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
        print({"case": case["name"], "replay": 1, "passed": first["passed"],
            "maximum_errors": first["maximum_errors_in_physical_units"]}, flush=True)
        second = run_case(reference, session, case, contract)
        write(out / (case["name"] + "-second.json"), second)
        exact = first["all_output_and_state_trajectory_sha256"] == second["all_output_and_state_trajectory_sha256"]
        reports.append({"case": case["name"], "first": first, "second": second,
            "reset_replay_bit_exact": exact, "passed": first["passed"] and second["passed"] and exact})
        print({"case": case["name"], "replay": 2, "passed": reports[-1]["passed"],
            "reset_replay_bit_exact": exact}, flush=True)
    verify_inputs({"source_bindings": bindings})
    require(state_sha256(model.state_dict()) == fingerprint == payload["model_state_sha256"]
            and not torch.cuda.is_initialized(), "Native source or CPU scope changed")
    passed = all(r["passed"] for r in reports)
    write(out / "result.json", {"status": "pass" if passed else "fail", "reports": reports,
        "graph_sha256": sha(graph_path), "saved_graph_path": str(graph_path),
        "source_checkpoint": checkpoint, "source_model_state_sha256": fingerprint,
        "source_bindings": bindings, "source_bindings_unchanged": True,
        "tolerances_changed": False, "quality_measured": False, "native_host_qualified": False,
        "gpu_used": False, "graph_plus_host_delay_samples": 256,
        "elapsed_seconds": time.monotonic() - began, "budget_before": before,
        "budget_after": budget_snapshot(budget)})
    print({"status": "pass" if passed else "fail"}, flush=True)
    require(passed, "Long independent numerical parity or reset replay failed")


if __name__ == "__main__":
    main()
