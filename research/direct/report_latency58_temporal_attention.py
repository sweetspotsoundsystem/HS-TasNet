"""Review a completed, externally observed temporal-attention endpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs


def worst_cells(cells):
    rows = [{"track": track, "stem": stem, **entry["metrics"]}
            for track, stems in cells.items() for stem, entry in stems.items()]
    require(len(rows) == 56, "The review must cover all 14 tracks and four stems")
    return {metric: sorted((row for row in rows if row[metric]["delta"] is not None),
                           key=lambda row: row[metric]["delta"], reverse=metric == "absent_fp_dbfs")[:4]
            for metric in ("full_sdr_db", "sir_db", "low_20_250", "low_20_80", "low_80_250",
                           "low_250_500", "absent_fp_dbfs")}


def main():
    from research.direct.compare import compare
    from research.direct.report_latency58_vocal_focus import music_cells
    from research.direct.latency58_sdr_checkpoint import require_space
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-root-session", required=True, type=int)
    parser.add_argument("--actual-root-exit-code", required=True, type=int, choices=[0])
    parser.add_argument("--actual-root-tool-chunk", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and args.actual_root_session > 0
            and args.actual_root_tool_chunk and all(c.isalnum() for c in args.actual_root_tool_chunk),
            "Provide the actual completed root execution evidence")
    root = PHASE / "temporal-attention-001"
    plan_path, quality_path = root / "plan.json", root / "full14/result.json"
    plan, quality = read(plan_path), read(quality_path)
    terminal, audit, training = (read(root / name) for name in
                                 ("result.json", "checkpoint-audit.json", "production-run/result.json"))
    require(terminal["status"] == "training_audit_and_full14_complete"
            and quality["status"] == training["status"] == audit["status"] == "pass"
            and quality["track_count"] == 14 and quality["excerpt_count"] == 28
            and quality["source_bindings_unchanged"] and training["source_bindings_unchanged"]
            and audit["source_bindings_unchanged"] and audit["saved_optimizer_tensor_count"] == 30
            and audit["algorithmic_latency_samples"] == 256
            and audit["step"] == training["updates"] == plan["config"]["steps"]
            and training["checkpoint_written"] and quality["graph_delay_samples"] == quality["host_queue_samples"] == 128,
            "Training, saved audit and unchanged full14 evaluation must be complete")
    paths = [plan_path, quality_path, root / "full14/plan.json", Path(__file__).resolve(),
             root / "result.json", root / "checkpoint-audit.json", root / "production-run/result.json",
             root / "root-command.json", root / "production-prefix-independent-check.json",
             ROOT / "research/direct/compare.py", ROOT / "research/direct/report_latency58_vocal_focus.py"]
    for name in ("resource-stage/execution.json", "production-stage/execution.json", "full14/execution.json"):
        path = root / name
        execution = read(path)
        require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                and not execution.get("timed_out", False), "A required stage failed or remains open")
        paths.append(path)
        if "monitor_result" in execution:
            monitor_path = Path(execution["monitor_result"])
            monitor = read(monitor_path)
            require(monitor["status"] == monitor["supervisor_health"] == "pass"
                    and monitor["child_exit_code"] == 0 and monitor["post_exit_quiet_completed"],
                    "A GPU monitor did not close successfully")
            paths.append(monitor_path)
    verify_inputs(plan)
    verify_inputs(quality)
    root_command = read(root / "root-command.json")
    verify_inputs(root_command)
    checkpoint = terminal["checkpoint"]
    generation = Path(checkpoint["path"]).parent
    receipt = read(generation / "receipt.json")
    require(checkpoint == audit["checkpoint"] == training["checkpoint"] == quality["results"][0]["checkpoint"]
            and receipt["step"] == audit["step"] and receipt["plan_sha256"] == sha(plan_path)
            and sha(quality_path) == terminal["quality_result"]["sha256"]
            and audit["model_state_sha256"] == training["final_model_state_sha256"]
            == quality["results"][0]["model"]["model_state_sha256_after"], "Endpoint identity changed")
    for name, expected in receipt["files"].items():
        path = generation / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == expected["bytes"]
                and sha(path) == expected["sha256"], "Saved generation differs from its audit")
    parent_path = Path(plan["reference_result"])
    parent = read(parent_path)["results"][0]
    candidate = quality["results"][0]
    require(parent["checkpoint"] == plan["parent_checkpoint"]
            and parent["aggregate"]["full_sdr_db"] == 4.266897232064164,
            "Use the reviewed best saved parent")
    baseline_path = PHASE / "c204-residual-share-full14-001/result.json"
    baselines = read(baseline_path)["policies"]
    c204, fixed = baselines["working_policy"], baselines["fixed_share"]
    require(c204["aggregate"]["full_sdr_db"] == 4.069078803302578
            and fixed["aggregate"]["full_sdr_db"] == 4.0846618609770395, "Preserved baselines changed")
    references = {"parent": parent, "c204": c204, "preserved_fixed_share": fixed}
    comparisons = {key: compare(value, candidate) for key, value in references.items()}
    cells = {key: music_cells(value, candidate) for key, value in references.items()}
    score = candidate["aggregate"]["full_sdr_db"]
    selected = score > parent["aggregate"]["full_sdr_db"]
    best = candidate if selected else parent
    paths.extend((parent_path, baseline_path, generation / "receipt.json", Path(checkpoint["path"])))
    if selected:
        paths.append(generation / "optimizer.pt")
    bindings = {**root_command["source_bindings"], **quality["source_bindings"],
                **{str(path): sha(path) for path in paths}}
    completion = (f"Unified execution {args.actual_root_session} returned actual exit 0 "
                  f"in tool output {args.actual_root_tool_chunk}.")
    verify_inputs({"source_bindings": bindings})
    require_space(plan, 2_000_000)
    write(root / "root-execution.json", {"actual_exit_code": 0, "actual_root_completion_evidence": completion,
          "source_bindings_unchanged": True, "root_command_sha256": sha(root / "root-command.json"),
          "plan_sha256": sha(plan_path), "result_sha256": sha(root / "result.json")})
    bindings[str(root / "root-execution.json")] = sha(root / "root-execution.json")
    track_deltas = comparisons["parent"]["metrics"]["full_sdr_db"]["per_track_macro_delta"]
    review = {"status": "selected_for_research" if selected else "not_selected",
              "actual_root_exit_code": 0, "actual_root_completion_evidence": completion,
              "reason": "Select the higher saved full14 SDR endpoint; preserve all detailed regressions and baseline weights.",
              "full14_vs_parent": comparisons["parent"], "full14_vs_c204": comparisons["c204"],
              "full14_vs_preserved_fixed_share": comparisons["preserved_fixed_share"],
              "all_track_stem_band_absence": cells,
              "worst_cells_to_review": {key: worst_cells(value) for key, value in cells.items()},
              "full_sdr_improved_tracks": sum(value > 0 for value in track_deltas.values()),
              "full_sdr_regressed_tracks": sum(value < 0 for value in track_deltas.values()),
              "best_research_checkpoint": best["checkpoint"],
              "best_research_reference_result": str(quality_path if selected else parent_path),
              "best_full_sdr_db": best["aggregate"]["full_sdr_db"],
              "target_gap_db": max(0., 5. - best["aggregate"]["full_sdr_db"]),
              "inference_checkpoint_retained": True, "optimizer_retained_at_review": True,
              "audited_optimizer_sha256": audit["optimizer_sha256"],
              "optimizer_may_be_retired_after_this_review": not selected,
              "optimizer_availability_scope": "Retained at review. Any later retirement needs its own identity, inactivity and reservation checks.",
              "plugin_replaced": False, "new_weights_native_timing_measured": False,
              "additional_public_stream_states": 2, "graph_plus_host_delay_samples": 256,
              "interpretation": "Single-seed architecture trial with fresh Adam and a new sample range; not a matched causal ablation. Track-bootstrap uncertainty excludes training-seed uncertainty and selection effects.",
              "next_direction": "Finish the all-signed inference study, then assess the next quality trial from the selected saved checkpoint.",
              "source_bindings": bindings}
    verify_inputs(review)
    write(root / "selection-review.json", review)
    print(json.dumps({key: review[key] for key in ("status", "best_full_sdr_db", "target_gap_db",
          "full_sdr_improved_tracks", "full_sdr_regressed_tracks", "optimizer_may_be_retired_after_this_review")}), flush=True)


if __name__ == "__main__":
    main()
