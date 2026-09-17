"""Authenticate the packed trial's retained comparison panel without inference."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.report_latency58_grouped_vocal_parent import paired_root, full_report, bind
from research.direct.report_latency58_vocal_focus import load_views
from research.direct.report_latency58_grouped_continuation import compare_endpoint, fingerprint
from research.direct.report_latency58_branch_output_int8 import require_close
from research.direct.run_latency58_weighted_quality_v2 import transport_proof
from research.direct.latency58_weighted_storage import snapshot

MAIN = PHASE / "branch-weighted-vocal-014"
OUT = MAIN / "quality-reference-qualification"


def main():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require CUDA-hidden CPU1 reference qualification")
    require(not OUT.exists(), "Preserve previous reference qualifications")
    began = time.monotonic()
    source = read(MAIN / "plan.json")
    decision_binding = source["scientific_decision"]
    require(sha(decision_binding["path"]) == decision_binding["sha256"], "Scientific decision changed")
    decision = read(decision_binding["path"])
    previous = PHASE / "grouped-continuation-review-013"
    previous_plan, previous_result, previous_execution = (read(previous / name)
        for name in ("plan.json", "result.json", "execution.json"))
    require(previous_result["status"] == "pass" and previous_result["source_bindings_unchanged"]
            and previous_execution["actual_exit_code"] == 0 and not previous_execution["timed_out"]
            and previous_execution["source_bindings_unchanged"]
            and previous_execution["result_sha256"] == sha(previous / "result.json")
            and previous_execution["plan_sha256"] == previous_result["plan_sha256"] == sha(previous / "plan.json"),
            "Existing independent comparison is incomplete")
    bindings = {}
    bind(bindings, [Path(__file__).resolve(), MAIN / "plan.json", Path(decision_binding["path"]),
        MAIN / "evaluation-adapter-source-proof-v2.json", ROOT / "research/direct/run_latency58_weighted_quality_v2.py",
        *(previous / name for name in ("plan.json", "result.json", "execution.json"))])
    proof = read(MAIN / "evaluation-adapter-source-proof-v2.json")
    require(proof["status"] == "pass" and proof["quality_controller"]["sha256"]
            == sha(ROOT / "research/direct/run_latency58_weighted_quality_v2.py"), "Controller changed after source proof")
    geometry = ("manifest", "config", "track_indices", "track_intervals", "protocol_version")
    template = read(PHASE / "paired-vocal-grouped-013/plan.json")
    endpoints, roots = {}, {}
    from research import evaluate as legacy
    for label, directory, role, selection_key in (
        ("starting_parent", previous_plan["reference_source_views"]["starting_parent"], "ema", "starting_parent"),
        ("retained_best", previous_plan["reference_source_views"]["retained_best"], "ema", "retained_best"),
        ("raw", previous_plan["candidate_source_views"], "raw", "raw"),
        ("ema", previous_plan["candidate_source_views"], "ema", "ema")):
        directory = Path(directory)
        if directory not in roots:
            roots[directory] = paired_root(directory, bindings)
        prepared, completed = roots[directory]
        require(all(prepared[k] == template[k] for k in geometry), "Comparison physical panel changed")
        views = load_views(directory / role, bindings)
        expected = decision["preserved_models"][selection_key]
        require(views["model"] == completed["models"][role] == expected, "Comparison role differs from selected scientific decision")
        full_binding = expected["original_full_mixture_report"]
        require(sha(full_binding["path"]) == full_binding["sha256"], "Reference full14 changed")
        full, _ = full_report(Path(full_binding["path"]).parent, expected["checkpoint"], expected["model_state_sha256"], bindings)
        require_close(full["aggregate"], legacy._aggregate_tracks(full["tracks"]), label + " aggregate")
        endpoints[label] = {"full": full, "views": views}
        print(json.dumps({"event": "reference_authenticated", "role": label,
                          "full_sdr_db": full["aggregate"]["full_sdr_db"]}), flush=True)
    reproduced = {}
    for reference in ("starting_parent", "retained_best"):
        for role in ("raw", "ema"):
            comparison = compare_endpoint(endpoints[reference], endpoints[role])
            expected = previous_result["comparisons"][reference][role]
            require_close(comparison, expected, reference + "/" + role + " complete stored review")
            reproduced[reference + "/" + role] = {"fingerprint": fingerprint(comparison),
                "track_stem_cells": sum(map(len, comparison["all_track_stem_cells"].values())),
                "paired_windows": len(comparison["paired_windows"]["all_windows"])}
    peer = compare_endpoint(endpoints["raw"], endpoints["ema"])
    require_close(peer, previous_result["comparisons"]["ema_vs_raw"], "complete stored peer review")
    from research.direct.compare_latency58_vocal_views import compare_reports
    from research.direct.report_latency58_branch_gru_int8 import compare_windows
    released_records = template["released_reference"]
    require(all(sha(v["path"]) == v["sha256"] for v in released_records.values()), "Released source-view reference changed")
    bind(bindings, [Path(v["path"]) for v in released_records.values()])
    released_plan, released_result, released_execution = (read(released_records[k]["path"])
                                                         for k in ("plan", "result", "execution"))
    argv = released_execution["argv"]
    require(released_result["status"] == "pass" and released_result["source_bindings_unchanged"]
            and released_execution["actual_exit_code"] == 0 and not released_execution["timed_out"]
            and released_execution["source_bindings_unchanged"]
            and released_result["plan_sha256"] == released_records["plan"]["sha256"]
            and argv[argv.index("--plan") + 1] == released_records["plan"]["path"]
            and argv[argv.index("--plan-sha256") + 1] == released_records["plan"]["sha256"]
            and all(released_plan[k] == template[k] for k in geometry), "Released reference execution or protocol differs")
    released = {"version": released_plan["protocol_version"], "model": released_result["checkpoint"],
                "tracks": released_result["tracks"], "aggregate": released_result["aggregate"]}
    stored_views = roots[Path(previous_plan["candidate_source_views"])][1]
    for role in ("raw", "ema"):
        comparison = {"aggregate": compare_reports(released, endpoints[role]["views"]),
                      "windows": compare_windows(released["tracks"], endpoints[role]["views"]["tracks"])}
        require_close(comparison, stored_views["comparisons"][role + "_vs_released"], "stored released comparison/" + role)
        reproduced[role + "_vs_released"] = {"fingerprint": fingerprint(comparison),
                                             "paired_windows": len(comparison["windows"]["all_windows"])}
    ast_proof = transport_proof()
    verify_inputs({"source_bindings": bindings})
    before = snapshot(source)
    OUT.mkdir()
    prepared = {"schema": "latency58-weighted-quality-reference-qualification-plan-v1", "source_bindings": bindings,
        "training_plan": {"path": str(MAIN / "plan.json"), "sha256": sha(MAIN / "plan.json")},
        "comparison_roles": list(endpoints), "cpu_only": True, "inference_executed": False,
        "current_training_weights_inspected": False, "storage_before": before}
    write(OUT / "plan.json", prepared)
    result = {"schema": "latency58-weighted-quality-reference-qualification-result-v1", "status": "pass",
        "observed_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": sha(OUT / "plan.json"),
        "source_bindings_unchanged": True, "source_binding_count": len(bindings),
        "reference_full_sdr_db": {k: v["full"]["aggregate"]["full_sdr_db"] for k, v in endpoints.items()},
        "reproduced_complete_comparisons": reproduced, "reproduced_raw_ema_peer_fingerprint": fingerprint(peer),
        "transport_ast_proof": ast_proof, "all_four_reference_models_match_scientific_decision": True,
        "all_source_view_physical_panels_identical": True, "released_reference_actual_argv_authenticated": True,
        "all_original_full14_and_source_view_comparisons_reproduced": True,
        "storage_after": snapshot(source), "elapsed_seconds": time.monotonic() - began,
        "gpu_used": False, "inference_executed": False, "current_training_weights_inspected": False,
        "new_model_quality_measured": False, "overall_goal_complete": False}
    write(OUT / "result.json", result)
    print(json.dumps({"status": "pass", "references": list(endpoints), "reference_binding_count": len(bindings),
                      "complete_comparison_count": len(reproduced) + 1}), flush=True)


if __name__ == "__main__":
    main()
