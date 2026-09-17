"""Screen a fused FP64 attention QKV product against the exact PR #17 parent."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import time

from research.direct.latency58_m4_followup_budget import ROOT, POLICY, snapshot
from research.direct.run_latency58_quality import require, read, sha, write

PHASE = ROOT / "research/direct/runs/latency58"
OUT = ROOT / "research/m4_followup_20260916/attention-qkv-screen-001"


def main():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and all(os.environ.get(k) == "1"
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CPU1")
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from research.direct.latency58_attention_qkv_fusion import build, PARENT_SHA
    from research.direct.latency58_branch_output_int8_reference import make_reference
    from research.direct.latency58_branch_int8_verify import session_for, run_case
    from research.direct.check_latency58_branch_int8_plugin import cases
    from research.direct.latency58_branch_onnx import TOLERANCES, interface
    from research.direct.latency58_branch_memory_checkpoint import load_model
    from research.direct.train_latency58 import verify_inputs, state_sha256

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    require(ort.__version__ == "1.26.0", "Require shipping ORT version")
    before = snapshot()
    require(not OUT.exists(), "Preserve earlier screen evidence")
    began = time.monotonic()
    short_root = PHASE / "branch-output-int8-screen-003"
    parent_result = read(short_root / "result.json")
    parent_execution = read(short_root / "execution.json")
    require(parent_result["status"] == "pass" and parent_result["strict_parity_passed"]
            and parent_execution["actual_exit_code"] == 0 and not parent_execution["timed_out"],
            "PR #17 source screen did not finish cleanly")
    baseline_path = PHASE / "deployed-vocal-views-001/plan.json"
    baseline = read(baseline_path)
    checkpoint = parent_result["source_checkpoint"]
    parent_path = PHASE / "m4-inference-diagnostics-plugin-001/model/model.onnx"
    require(sha(parent_path) == PARENT_SHA and parent_result["graph_sha256"] == PARENT_SHA,
            "Exact deployed parent changed")
    profile_root = ROOT / "research/m4_followup_20260916"
    profile_path = profile_root / "pr17-operator-profile-001/result.json"
    profile_execution_path = profile_root / "pr17-profile-execution.json"
    profile, profile_execution = read(profile_path), read(profile_execution_path)
    require(profile["status"] == "pass" and profile["graph_sha256"] == PARENT_SHA
            and profile_execution["actual_exit_code"] == 0 and not profile_execution["timed_out"],
            "Current graph profile did not finish cleanly")
    verify_inputs(profile)
    ten_path = Path(baseline["checkpoint"]["path"])
    fourteen_path = Path(parent_result["parent_graph"]["path"])
    ten_proof_path = PHASE / "branch-plugin-screen-001/graph.json"
    fourteen_proof_path = PHASE / "branch-gru-int8-screen-002/graph.json"
    source_paths = [Path(__file__).resolve(), POLICY,
        ROOT / "research/direct/latency58_m4_followup_budget.py",
        ROOT / "research/direct/latency58_attention_qkv_fusion.py",
        ROOT / "research/direct/latency58_branch_output_int8_reference.py",
        baseline_path, short_root / "result.json", short_root / "execution.json", short_root / "graph.json",
        parent_path, ten_path, fourteen_path, ten_proof_path, fourteen_proof_path,
        Path(checkpoint["path"]), profile_path, profile_execution_path,
        Path(ort.__file__).resolve(), *sorted((Path(ort.__file__).resolve().parent / "capi").glob("*.so*"))]
    bindings = {p: h for p, h in parent_result["source_bindings"].items() if Path(p).suffix == ".py"}
    bindings.update({str(p): sha(p) for p in source_paths})
    verify_inputs({"source_bindings": bindings})
    require(bindings[checkpoint["path"]] == checkpoint["sha256"], "Saved source checkpoint changed")
    print(json.dumps({"stage": "load_saved_source_and_build_candidate"}), flush=True)
    model, payload = load_model(checkpoint)
    fingerprint = state_sha256(model.state_dict())
    contract = interface(model)
    require(json.loads(json.dumps(contract)) == baseline["interface"], "Public interface changed")
    ten_proof = read(ten_proof_path)
    require(fingerprint == ten_proof["conversion"]["source_model_state_sha256"], "Source lineage changed")
    parent = onnx.load(parent_path)
    graph, conversion = build(parent)
    data = graph.SerializeToString()
    OUT.mkdir(parents=True)
    write(OUT / "plan.json", {"source_bindings": bindings, "checkpoint": checkpoint,
        "parent_graph": {"path": str(parent_path), "sha256": PARENT_SHA}, "interface": contract,
        "tolerances": TOLERANCES, "profile": str(profile_path), "budget_before": before,
        "purpose": "Fuse QKV projections with unchanged stored weights and FP64 precision; preserve query last-frame slice",
        "short_cases": 5, "signal_hops": 64, "optimizations": ["disabled", "all"], "reset_replays": 2,
        "local_speed_gate": {"minimum_median_block_p50_reduction": 0.03,
            "minimum_faster_pairs": 3, "pairs": 4, "order": "ABBAABBA"},
        "concurrent_gpu_training": True, "native_host_qualification": False, "quality_selection": False})
    graph_path = OUT / "model.onnx"
    with graph_path.open("xb") as stream:
        stream.write(data)
    require(sha(graph_path) == hashlib.sha256(data).hexdigest(), "Saved graph differs")
    write(OUT / "graph.json", {"graph_sha256": sha(graph_path), "graph_bytes": len(data),
        "metadata": {v.key: v.value for v in graph.metadata_props}, "conversion": conversion})
    reference, independent = make_reference(model, onnx.load(ten_path), ten_proof["conversion"],
        onnx.load(fourteen_path), read(fourteen_proof_path)["conversion"], parent,
        read(short_root / "graph.json")["conversion"])
    write(OUT / "independent-reference.json", independent)
    reports = []
    fixtures = list(cases(model, hops=64))
    require(len(fixtures) == 5, "Short fixture set changed")
    print(json.dumps({"stage": "independent_parity", "graph_bytes": len(data)}), flush=True)
    for optimization in ("disabled", "all"):
        session = session_for(data, contract, optimization)
        for case in fixtures:
            first = run_case(reference, session, case, contract)
            replay = run_case(reference, session, case, contract)
            exact = first["all_output_and_state_trajectory_sha256"] == replay["all_output_and_state_trajectory_sha256"]
            report = {"optimization": optimization, "case": case["name"], "first": first, "replay": replay,
                      "reset_replay_bit_exact": exact,
                      "passed": first["passed"] and replay["passed"] and exact}
            write(OUT / (optimization + "-" + case["name"] + ".json"), report)
            reports.append(report)
            print(json.dumps({"optimization": optimization, "case": case["name"], "passed": report["passed"],
                "maximum_errors": first["maximum_errors_in_physical_units"]}), flush=True)
        del session
    passed = all(r["passed"] for r in reports)
    timings, gate = [], {"evaluated": False, "passed": False}
    if passed:
        sessions = {"parent": session_for(parent_path.read_bytes(), contract),
                    "candidate": session_for(data, contract)}
        signal = np.random.default_rng(20260916).normal(0, .03, (64, 1, 2, 128)).astype(np.float32)
        for repeat in range(2):
            for label in ("parent", "candidate", "candidate", "parent"):
                states = [np.zeros(shape, np.float32) for shape in contract["state_shapes"]]
                durations = []
                for audio in signal:
                    feed = dict(zip(contract["input_names"], [audio, *states], strict=True))
                    start = time.perf_counter_ns()
                    values = sessions[label].run(list(contract["output_names"]), feed)
                    durations.append((time.perf_counter_ns() - start) / 1e6)
                    require(all(np.isfinite(v).all() for v in values), "Nonfinite timing output")
                    states = values[1:]
                measured = durations[16:]
                timings.append({"repeat": repeat, "label": label, "warmup_hops": 16, "measured_hops": 48,
                    "durations_ms": measured, "mean_ms": statistics.mean(measured),
                    "median_ms": statistics.median(measured), "p95_ms": float(np.percentile(measured, 95)),
                    "p99_ms": float(np.percentile(measured, 99)), "maximum_ms": max(measured),
                    "over_hop_budget": sum(v > 1000 * 128 / 44100 for v in measured)})
        medians = {label: statistics.median(v["median_ms"] for v in timings if v["label"] == label)
                   for label in sessions}
        faster = sum(timings[c]["median_ms"] < timings[p]["median_ms"]
                     for p, c in ((0, 1), (3, 2), (4, 5), (7, 6)))
        reduction = 1 - medians["candidate"] / medians["parent"]
        gate = {"evaluated": True, "median_of_block_p50_ms": medians,
                "reduction_fraction": reduction, "faster_pairs": faster, "total_pairs": 4,
                "passed": reduction >= .03 and faster >= 3}
    verify_inputs({"source_bindings": bindings})
    require(state_sha256(model.state_dict()) == fingerprint == payload["model_state_sha256"]
            and not torch.cuda.is_initialized(), "Native source or CPU execution scope changed")
    require(sha(graph_path) == read(OUT / "graph.json")["graph_sha256"], "Saved candidate changed")
    write(OUT / "result.json", {"status": "pass" if passed else "fail", "strict_parity_passed": passed,
        "reports": reports, "saved_graph_path": str(graph_path), "graph_sha256": sha(graph_path),
        "graph_bytes": len(data), "parent_graph_sha256": PARENT_SHA, "source_checkpoint": checkpoint,
        "source_model_state_sha256": fingerprint, "source_bindings": bindings, "source_bindings_unchanged": True,
        "local_abba_timing_blocks": timings, "local_speed_gate": gate,
        "graph_plus_host_delay_samples": 256, "tolerances_changed": False, "gpu_used": False,
        "quality_measured": False, "quality_selected": False, "native_host_qualified": False,
        "elapsed_seconds": time.monotonic() - began, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "budget_before": before, "budget_after": snapshot(),
        "limitations": "Independent original sixteen-projection reference. Stored weights and matrix precision are unchanged; FP64 reduction order may differ. Long-stream numerical checks, full14 and source-view quality, native timing and physical M4 acceptance remain required. Linux Python timings include allocation and concurrent GPU training load."})
    print(json.dumps({"status": "pass" if passed else "fail", "local_speed_gate": gate}), flush=True)
    require(passed, "Fused attention candidate failed independent numerical checks")


if __name__ == "__main__":
    main()
