"""Screen two additional branch-output integer products against the PR #15 parent."""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import time

from research.direct.run_latency58_quality import ROOT, PHASE, require, read, sha, write
from research.direct.train_latency58 import verify_inputs, state_sha256


def main():
    import numpy as np
    import onnx
    import torch
    from research.direct.run_latency58_deployed_vocal_views import require_cpu, budget_snapshot
    from research.direct.latency58_branch_plugin_endpoint import selected_endpoint, export_bindings
    from research.direct.latency58_branch_output_int8 import build
    from research.direct.latency58_branch_output_int8_reference import make_reference
    from research.direct.latency58_branch_int8_verify import session_for, run_case
    from research.direct.check_latency58_branch_int8_plugin import cases
    from research.direct.latency58_branch_onnx import TOLERANCES, interface
    from research.direct.latency58_branch_memory_checkpoint import load_model

    baseline_path = PHASE / "deployed-vocal-views-001/plan.json"
    baseline = read(baseline_path)
    require_cpu(baseline)
    budget = {**baseline, "diagnostic_artifact_allowance_bytes": 250_000_000}
    before = budget_snapshot(budget)
    profile_path = PHASE / "branch-gru-int8-profile-001/results/result.json"
    profile_execution_path = PHASE / "branch-gru-int8-profile-001/execution.json"
    profile, execution = read(profile_path), read(profile_execution_path)
    require(profile["status"] == "pass" and profile["source_bindings_unchanged"]
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and profile["checkpoint"]["sha256"] == "878c74694fa4c558de1c5a75837893a0afeadcf57f6e3b860d5904cab04e9fc9", "Profile does not identify the released graph")
    verify_inputs(profile)
    print({"stage": "authenticate_saved_source"}, flush=True)
    first_out = PHASE / "branch-output-int8-screen-001"
    first_plan, failed = read(first_out / "plan.json"), read(first_out / "execution.json")
    require(failed["actual_exit_code"] == 1 and failed["source_bindings_unchanged"] and not failed["parity_executed"], "Unexpected prior screen")
    checkpoint = first_plan["checkpoint"]
    model, payload = load_model(checkpoint)
    bindings = {p: h for p, h in first_plan["source_bindings"].items() if Path(p).suffix == ".py"}
    bindings[checkpoint["path"]] = checkpoint["sha256"]
    require(sha(checkpoint["path"]) == checkpoint["sha256"], "Saved source changed")
    contract = interface(model)
    require(json.loads(json.dumps(contract)) == baseline["interface"], "Native interface differs")
    require(checkpoint == baseline["checkpoint"]["source_checkpoint"], "Wrong source checkpoint")
    fingerprint = state_sha256(model.state_dict())
    ten_path = Path(baseline["checkpoint"]["path"])
    parent_path = PHASE / "branch-gru-int8-plugin-001/model/model.onnx"
    require(sha(parent_path) == "878c74694fa4c558de1c5a75837893a0afeadcf57f6e3b860d5904cab04e9fc9", "Released graph bytes changed")
    parent = onnx.load(parent_path)
    parent_proof_path = PHASE / "branch-plugin-screen-001/graph.json"
    parent_proof = read(parent_proof_path)
    require(parent_proof["graph_sha256"] == sha(ten_path)
            and parent_proof["conversion"]["source_model_state_sha256"] == fingerprint, "Source/graph lineage changed")
    fourteen_proof_path = PHASE / "branch-gru-int8-screen-002/graph.json"
    paths = [Path(__file__).resolve(), ROOT / "research/direct/latency58_branch_output_int8.py", ROOT / "research/direct/latency58_branch_gru_int8.py",
        ROOT / "research/direct/latency58_branch_int8.py", ROOT / "research/direct/latency58_int8_precise_float.py",
        ROOT / "research/direct/latency58_branch_int8_verify.py", ROOT / "research/direct/check_latency58_branch_int8_plugin.py",
        ROOT / "research/direct/run_latency58_deployed_vocal_views.py",
        profile_path, profile_execution_path, baseline_path, parent_path, parent_proof_path, ten_path, fourteen_proof_path]
    paths.append(ROOT / "research/direct/latency58_branch_output_int8_reference.py")
    paths.extend((PHASE / "branch-output-int8-screen-002") / n for n in ("plan.json", "execution.json"))
    paths.extend(first_out / n for n in ("plan.json", "execution.json", "graph.json", "model.onnx"))
    bindings = {**bindings, **{str(p): sha(p) for p in paths}}
    verify_inputs({"source_bindings": bindings})
    out = PHASE / "branch-output-int8-screen-003"
    require(not out.exists(), "Preserve previous experiment")
    out.mkdir()
    write(out / "plan.json", {"source_bindings": bindings, "checkpoint": checkpoint,
        "parent_graph": profile["checkpoint"], "interface": baseline["interface"], "tolerances": TOLERANCES,
        "profile": str(profile_path), "budget_before": before,
        "purpose": "Test two branch output products that consumed 9.57% of current local kernel time",
        "quality_selection": False, "native_host_qualification": False})
    began = time.monotonic()
    graph_path = first_out / "model.onnx"
    original = read(first_out / "graph.json")
    require(sha(graph_path) == original["graph_sha256"], "Preserved graph changed")
    graph = onnx.load(graph_path)
    rebuilt, conversion = build(parent)
    require(rebuilt.SerializeToString() == graph.SerializeToString() and conversion == original["conversion"], "Candidate is not reproducible")
    del rebuilt
    data = graph.SerializeToString()
    write(out / "graph.json", {"graph_sha256": sha(graph_path), "graph_bytes": len(data),
        "metadata": {v.key: v.value for v in graph.metadata_props}, "conversion": conversion})
    reference, independent = make_reference(model, onnx.load(ten_path), parent_proof["conversion"], parent, read(fourteen_proof_path)["conversion"], graph, conversion)
    write(out / "independent-reference.json", independent)
    print({"stage": "independent_numerical_parity", "graph_bytes": len(data)}, flush=True)
    reports = []
    for optimization in ("disabled", "all"):
        session = session_for(data, contract, optimization)
        for case in cases(model, hops=64):
            first = run_case(reference, session, case, contract)
            replay = run_case(reference, session, case, contract)
            exact = first["all_output_and_state_trajectory_sha256"] == replay["all_output_and_state_trajectory_sha256"]
            first.update(optimization=optimization, reset_replay_bit_exact=exact,
                         passed=first["passed"] and replay["passed"] and exact)
            write(out / (optimization + "-" + case["name"] + ".json"), first)
            reports.append(first)
            print({"optimization": optimization, "case": case["name"], "passed": first["passed"],
                "maximum_errors": first["maximum_errors_in_physical_units"]}, flush=True)
    passed = all(r["passed"] for r in reports)
    timings = []
    if passed:
        sessions = {"parent": session_for(parent_path.read_bytes(), baseline["interface"]),
                    "candidate": session_for(data, baseline["interface"])}
        signal = np.random.default_rng(20260913).normal(0, .03, (64, 1, 2, 128)).astype(np.float32)
        # Alternating ABBA blocks reduce order bias. They remain a local,
        # concurrent-load screen and cannot establish target-Mac speed.
        for repeat in range(2):
            for label in ("parent", "candidate", "candidate", "parent"):
                states = [np.zeros(shape, np.float32) for shape in baseline["interface"]["state_shapes"]]
                durations = []
                for audio in signal:
                    feed = dict(zip(baseline["interface"]["input_names"], [audio, *states], strict=True))
                    start = time.perf_counter_ns()
                    values = sessions[label].run(baseline["interface"]["output_names"], feed)
                    durations.append((time.perf_counter_ns() - start) / 1e6)
                    require(all(np.isfinite(v).all() for v in values), "Nonfinite timing output")
                    states = values[1:]
                timings.append({"repeat": repeat, "label": label, "warmup_hops": 16, "measured_hops": 48,
                    "mean_ms": statistics.mean(durations[16:]), "median_ms": statistics.median(durations[16:]),
                    "maximum_ms": max(durations[16:])})
    verify_inputs({"source_bindings": bindings})
    require(state_sha256(model.state_dict()) == fingerprint == payload["model_state_sha256"]
            and sha(graph_path) == read(out / "graph.json")["graph_sha256"]
            and not torch.cuda.is_initialized(), "Native tensors, candidate bytes or CPU scope changed")
    write(out / "result.json", {"status": "pass" if passed else "fail", "strict_parity_passed": passed,
        "reports": reports, "graph_sha256": sha(graph_path), "graph_bytes": len(data), "saved_graph_path": str(graph_path),
        "parent_graph": profile["checkpoint"], "source_checkpoint": checkpoint,
        "source_bindings": bindings, "source_bindings_unchanged": True,
        "tolerances_changed": False, "gpu_used": False, "quality_measured": False,
        "quality_selected": False, "native_host_qualified": False, "graph_plus_host_delay_samples": 256,
        "local_abba_timing_blocks": timings, "concurrent_training_and_validation": True,
        "budget_before": before, "budget_after": budget_snapshot(budget),
        "elapsed_seconds": time.monotonic() - began,
        "limitations": "Short synthetic strict parity to independently reconstructed sixteen-projection inference. This is not parity to the unquantized source or the PR #15 fourteen-projection graph. Full-mixture quality, counterfactual leakage, long-stream parity and target-Mac timing remain unmeasured."})
    print({"status": "pass" if passed else "fail", "local_abba_timing_blocks": timings}, flush=True)
    require(passed, "Two-output integer screen failed independent numerical parity")


if __name__ == "__main__":
    main()
