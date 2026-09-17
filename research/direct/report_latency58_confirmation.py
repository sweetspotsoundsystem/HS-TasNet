"""Authenticate and compare the two completed, prospectively reserved evaluations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.train_latency58 import verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Summary plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-confirmation-summary-plan-v1", "Different summary schema")
    verify_inputs(plan)
    from research.direct.evaluate_latency58_sdr_confirmation_v5 import validate_selection
    from research.direct.report_latency58_sdr import load_completed
    from research.direct.report_latency58_vocal_focus import music_cells
    from research.direct.compare import compare
    from research.direct.latency58_sdr_checkpoint import require_space

    evidence, reports, execution_plans = {}, {}, {}
    reservation = selection = None
    for label, kind in (("working", "working_baseline"), ("candidate", "selected_generation")):
        directory = Path(plan["evaluations"][label])
        require(directory.parent == PHASE, "Unexpected evaluation location")
        execution_plan, report = load_completed(directory, evidence)
        evidence.update(execution_plan["source_bindings"])
        reserved, selected = validate_selection(execution_plan)
        if reservation is None:
            reservation, selection = reserved, selected
        require(reserved == reservation and selected == selection
                and execution_plan["model_kind"] == kind
                and report["primary_selection"] == execution_plan["selection"] == plan["selection"]
                and report["reservation"] == execution_plan["reservation"] == plan["reservation"]
                and report["panel"] == "reserved-development-confirmation"
                and report["inputs_unchanged"]
                and report["root_source_bindings"] == execution_plan["source_bindings"]
                and report["confirmation_material_used_for_selection"] is False
                and report["selected_model_state_sha256"] == selection["model_state_sha256"]
                and report["track_names"] == reservation["track_names"]
                and report["excerpt_count"] == 28 and len(report["results"]) == 1
                and report["results"][0]["model"]["model_state_sha256"]
                == execution_plan["expected_model_state_sha256"], "Incomplete or different confirmation")
        execution = read(directory / "execution.json")
        argv = execution["argv"]
        require(argv[argv.index("-m") + 1] == "research.direct.evaluate_latency58_sdr_confirmation_v5"
                and argv[argv.index("--plan") + 1] == str(directory / "plan.json")
                and argv[argv.index("--plan-sha256") + 1] == sha(directory / "plan.json"),
                "Execution receipt identifies another evaluator")
        require(not report["cuda_initialized"] and report["cpu_rng_unchanged"]
                and report["device"] == "cpu" and report["threads"] == report["interop_threads"] == 1
                and report["intended_total_latency_samples"] == 256
                and report["graph_alignment_samples"] == 128
                and report["host_queue_implemented"] is False and report["host_qualified"] is False,
                "Confirmation runtime or latency claim differs")
        require([e["start_seconds"] for e in report["excerpts"]] == reservation["excerpt_starts"]
                and all(e["duration_seconds"] == reservation["duration_seconds"] for e in report["excerpts"]),
                "Different reserved intervals")
        reports[label], execution_plans[label] = report, execution_plan
    working, candidate = reports["working"], reports["candidate"]
    protocol_fields = ("manifest_sha256", "output_policy", "precision", "metrics", "metric_source_sha256",
                       "evaluator_sha256", "config_sha256", "streaming_state", "normalization",
                       "vocal_gain_override", "alignment_samples", "graph_alignment_samples",
                       "intended_total_latency_samples", "host_queue_samples", "unroll_hops", "io_block_hops")
    require(all(working[k] == candidate[k] for k in protocol_fields), "Paired confirmation protocols differ")
    left, right = working["results"][0], candidate["results"][0]
    require(right["model"]["model_state_sha256"] == selection["model_state_sha256"],
            "Candidate changed after selection")
    comparison, cells = compare(left, right), music_cells(left, right)
    rows = {}
    for track, stems in cells.items():
        for stem, item in stems.items():
            for metric, value in item["metrics"].items():
                if value["delta"] is not None:
                    rows.setdefault(metric, []).append({"track": track, "stem": stem, **value})
    cell_summary = {}
    for metric, values in rows.items():
        lower_better = metric == "absent_fp_dbfs"
        ordered = sorted(values, key=lambda row: -row["delta"] if lower_better else row["delta"])
        cell_summary[metric] = {
            "eligible": len(values), "lower_is_better": lower_better,
            "improved": sum((v["delta"] < 0 if lower_better else v["delta"] > 0) for v in values),
            "unchanged": sum(v["delta"] == 0 for v in values), "worst_five": ordered[:5],
        }
    require(all(plan["source_bindings"].get(p) == s for p, s in evidence.items()),
            "Summary omitted a completed input")
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "result.json").exists(), "Preserve summary")
    counted = require_space(execution_plans["candidate"], 2_000_000)
    verify_inputs(plan)
    full = comparison["metrics"]["full_sdr_db"]
    result = {
        "schema": "latency58-confirmation-summary-v1", "status": "pass",
        "status_scope": "Complete authenticated comparison, independent of whether quality improved.",
        "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"],
        "source_bindings_unchanged": True, "primary_selection": plan["selection"],
        "reservation": plan["reservation"], "model_state_sha256": selection["model_state_sha256"],
        "comparison": comparison, "all_track_stem_band_absence": cells, "cell_summary": cell_summary,
        "full_sdr_improves": full["delta"] > 0,
        "full_sdr_paired_interval_above_zero": full["paired_track_bootstrap_95_percent"][0] > 0,
        "all_metrics_compared": True, "confirmation_material_used_for_selection": False,
        "further_checkpoint_selection": False, "new_training_updates": 0,
        "human_listening_completed": False, "native_host_qualified": False, "deployment_selected": False,
        "counted_bytes_before": counted,
        "limitations": [reservation["limitation"], comparison["uncertainty_scope"],
                        "This comparison does not establish universal isolation, audibility or M4 timing."]}
    write(out / "result.json", result)
    print(json.dumps({"status": "pass", "output": str(out / "result.json"),
                      "metrics": comparison["metrics"], "cell_summary": cell_summary}), flush=True)


if __name__ == "__main__":
    main()
