"""Review the completed grouped-rate attention trial against its saved parent and rollback baselines."""
from __future__ import annotations

import argparse
import json
import math
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
    from research.direct.latency58_attention_grouped_checkpoint import (
        OPTIMIZER_SCHEMA, SCHEDULE_SCHEMA, optimizer_configuration)
    from research.direct.prepare_latency58_attention_grouped import RECIPE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-root-session", required=True, type=int)
    parser.add_argument("--actual-root-exit-code", required=True, type=int, choices=[0])
    parser.add_argument("--actual-root-tool-chunk", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and args.actual_root_session > 0
            and args.actual_root_tool_chunk and all(c.isalnum() for c in args.actual_root_tool_chunk),
            "Provide the actual completed root execution evidence")
    root = PHASE / "attention-grouped-001"
    required = [root / name for name in (
        "plan.json", "result.json", "checkpoint-audit.json", "production-run/result.json",
        "resource-stage/execution.json", "production-stage/execution.json", "full14/execution.json",
        "full14/result.json", "root-command.json", "production-prefix-independent-check.json")]
    require(all(path.is_file() for path in required),
            "Complete grouped training, saved audit and full14 before reviewing its endpoint")
    require(not (root / "selection-review.json").exists()
            and not (root / "root-execution.json").exists(), "Preserve existing endpoint reviews")
    plan_path, quality_path = root / "plan.json", root / "full14/result.json"
    plan, quality = read(plan_path), read(quality_path)
    terminal, audit, training = (read(root / name) for name in
                                 ("result.json", "checkpoint-audit.json", "production-run/result.json"))
    continuation_review_path = PHASE / "attention-continuation-001/selection-review.json"
    continuation_review = read(continuation_review_path)
    require(continuation_review["actual_root_exit_code"] == 0
            and continuation_review["status"] in ("selected_for_research", "not_selected")
            and 4.288099064999147 <= continuation_review["best_full_sdr_db"] < 5.0,
            "Require the completed continuation's saved-parent selection")
    verify_inputs(continuation_review)
    require(plan["schema"] == "latency58-attention-grouped-training-plan-v1"
            and plan["optimizer_schema"] == OPTIMIZER_SCHEMA and plan["optimizer_schedule"] == SCHEDULE_SCHEMA
            and plan["attention_lr_multiplier"] == RECIPE["attention_lr_multiplier"]
            and all(plan["config"][key] == RECIPE[key] for key in
                    ("steps", "lr", "min_lr", "warmup", "data_start", "seed"))
            and audit["optimizer_group_count"] == training["optimizer_group_count"] == 2
            and audit["optimizer_configuration"] == training["optimizer_configuration"]
            == optimizer_configuration(plan["attention_lr_multiplier"]),
            "Grouped optimizer or frozen trial recipe changed")
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
             ROOT / "research/direct/compare.py", ROOT / "research/direct/report_latency58_vocal_focus.py",
             continuation_review_path, ROOT / "research/direct/latency58_attention_grouped_checkpoint.py",
             ROOT / "research/direct/prepare_latency58_attention_grouped.py"]
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
    prefix = read(root / "production-prefix-independent-check.json")
    resource = read(root / "resource-run/result.json")
    verify_inputs(prefix)
    require(root_command["record_kind"] == "observed_live_root_command"
            and root_command["actual_root_session"] == prefix["actual_root_session"] == args.actual_root_session
            and prefix["status"] == "pass" and prefix["normalized_fields_equal"]
            and resource["matching_production_updates"] == training["matching_production_updates"]
            and resource["optimizer_group_count"] == 2
            and resource["optimizer_configuration"] == training["optimizer_configuration"]
            and root_command["argv"][3] == "research.direct.run_latency58_attention_grouped"
            and "--after-resource" in root_command["argv"],
            "Completed production must match the independently observed rehearsal prefix")
    with (root / "production-run/metrics.jsonl").open() as journal:
        journal_rows = [json.loads(next(journal)) for _ in range(2)]
    require(journal_rows == prefix["production_first_two_rows"]
            and [{k: v for k, v in row.items() if k not in ("elapsed_seconds", "peak_vram_gib")}
                 for row in journal_rows] == resource["matching_production_updates"],
            "The completed journal must retain the exact observed prefix")
    paths.extend((root / "resource-run/result.json", root / "production-run/metrics.jsonl"))
    checkpoint = terminal["checkpoint"]
    generation = Path(checkpoint["path"]).parent
    receipt = read(generation / "receipt.json")
    require(checkpoint == audit["checkpoint"] == training["checkpoint"] == quality["results"][0]["checkpoint"]
            and receipt["step"] == audit["step"] and receipt["plan_sha256"] == sha(plan_path)
            and receipt["optimizer_configuration"] == audit["optimizer_configuration"]
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
            == continuation_review["best_research_checkpoint"]
            and str(parent_path) == continuation_review["best_research_reference_result"]
            and parent["aggregate"]["full_sdr_db"] == continuation_review["best_full_sdr_db"]
            and plan["parent_kind"] == "saved_temporal_attention"
            and plan["parent_training_updates"] in (21250, 25250)
            and plan["optimizer_initialization"] == "fresh_adam"
            and plan["inference_architecture_changed"] is False,
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
    require(math.isfinite(score) and terminal["target_reached"] == quality["target_reached"] == (score >= 5.0),
            "The target decision must use the finite, saved-checkpoint full14 score")
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
              "candidate_saved_full_sdr_db": score,
              "saved_native_quality_target_reached": score >= 5.0,
              "target_gap_db": max(0., 5. - best["aggregate"]["full_sdr_db"]),
              "inference_checkpoint_retained": True, "optimizer_retained_at_review": True,
              "audited_optimizer_sha256": audit["optimizer_sha256"],
              "optimizer_may_be_retired_after_this_review": not selected,
              "optimizer_availability_scope": "Retained at review. Any later retirement needs its own identity, inactivity and reservation checks.",
              "plugin_replaced": False, "new_weights_native_timing_measured": False,
              "additional_public_stream_states": 0, "total_public_stream_states": 6,
              "graph_plus_host_delay_samples": 256,
              "total_training_updates": plan["parent_training_updates"] + plan["config"]["steps"],
              "parent_training_updates": plan["parent_training_updates"],
              "optimizer_group_count": 2, "optimizer_configuration": audit["optimizer_configuration"],
              "optimizer_schedule": plan["optimizer_schedule"], "recipe": RECIPE,
              "continuation_uses_trained_nonzero_attention_head": True,
              "optimizer_initialization": "fresh_adam",
              "interpretation": "Single-seed trial with fresh Adam, two parameter groups, a new schedule and a new sample range; not an exact optimizer restart or a matched causal ablation. Track-bootstrap uncertainty excludes training-seed uncertainty and selection effects.",
              "next_direction": "Review all regressions and continue quality work from the selected saved checkpoint until the full goal is proven. A native SDR result alone does not qualify new deployment graph timing.",
              "source_bindings": bindings}
    verify_inputs(review)
    write(root / "selection-review.json", review)
    print(json.dumps({key: review[key] for key in ("status", "best_full_sdr_db", "target_gap_db",
          "full_sdr_improved_tracks", "full_sdr_regressed_tracks", "optimizer_may_be_retired_after_this_review")}), flush=True)


if __name__ == "__main__":
    main()
