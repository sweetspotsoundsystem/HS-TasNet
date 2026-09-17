"""Review the saved branch-memory endpoint and every track/stem regression."""
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
    from research.direct.prepare_latency58_branch_memory import RECIPE
    from research.direct.latency58_attention_sdr_blend import VERSION, SDR_WEIGHT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-root-session", required=True, type=int)
    parser.add_argument("--actual-root-exit-code", required=True, type=int, choices=[0])
    parser.add_argument("--actual-root-tool-chunk", required=True)
    parser.add_argument("--validation-recovery", type=Path,
                        help="Completed recovery directory when the original launcher lost its exit status")
    args = parser.parse_args()
    require(Path.cwd() == ROOT and args.actual_root_session > 0
            and args.actual_root_tool_chunk and all(c.isalnum() for c in args.actual_root_tool_chunk),
            "Provide the actual completed root execution evidence")
    root = PHASE / "branch-memory-001"
    endpoint = root if args.validation_recovery is None else args.validation_recovery.resolve(strict=True)
    recovered = endpoint != root
    require(args.validation_recovery is None or endpoint == root / "validation-recovery-001",
            "Use only this trial's recorded validation recovery")
    require(not (root / "selection-review.json").exists()
            and not (root / "root-execution.json").exists(), "Preserve existing endpoint reviews")
    plan_path, quality_path = root / "plan.json", endpoint / "full14/result.json"
    plan, quality = read(plan_path), read(quality_path)
    require(plan["objective_version"] == VERSION and plan["direct_sdr_weight"] == SDR_WEIGHT == .1
            and all(plan["config"][key] == RECIPE[key] for key in
                    ("steps", "lr", "min_lr", "warmup", "data_start", "seed")), "Frozen branch-memory recipe changed")
    terminal = read(endpoint / "result.json")
    audit, training = (read(root / name) for name in ("checkpoint-audit.json", "production-run/result.json"))
    require(terminal["status"] == "training_audit_and_full14_complete"
            and quality["status"] == training["status"] == audit["status"] == "pass"
            and quality["track_count"] == 14 and quality["excerpt_count"] == 28
            and quality["source_bindings_unchanged"] and training["source_bindings_unchanged"]
            and audit["source_bindings_unchanged"] and audit["saved_optimizer_tensor_count"] == 40
            and audit["algorithmic_latency_samples"] == 256
            and audit["step"] == training["updates"] == plan["config"]["steps"]
            and training["checkpoint_written"] and quality["graph_delay_samples"] == quality["host_queue_samples"] == 128,
            "Training, saved audit and unchanged full14 evaluation must be complete")
    paths = [plan_path, quality_path, endpoint / "full14/plan.json", Path(__file__).resolve(),
             endpoint / "result.json", root / "checkpoint-audit.json", root / "production-run/result.json",
             root / "root-command.json", root / "production-prefix-independent-check.json",
             ROOT / "research/direct/compare.py", ROOT / "research/direct/report_latency58_vocal_focus.py"]
    for path in (root / "resource-stage/execution.json", root / "production-stage/execution.json",
                 endpoint / "full14/execution.json"):
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
    original_session = root_command["actual_root_session"]
    if recovered:
        recovery_intent = read(endpoint / "intent.json")
        original_attempt = read(endpoint / "original-attempt.json")
        verify_inputs(recovery_intent)
        verify_inputs(terminal)
        require(recovery_intent["original_root_session"] == original_session == 78470
                and args.actual_root_session != original_session
                and recovery_intent["original_root_actual_exit_code"] is None
                and terminal["original_root_actual_exit_code"] is None
                and terminal["recovered_validation"] and not terminal["training_repeated"]
                and original_attempt["original_processes_terminal_or_absent"]
                and original_attempt["exit_status_not_inferred_from_output"],
                "Recovery must preserve the missing original exit status and the completed training")
        old_plan = read(root / "full14/plan.json")
        new_plan = read(endpoint / "full14/plan.json")
        require({k: v for k, v in old_plan.items() if k not in ("output_directory", "source_bindings")}
                == {k: v for k, v in new_plan.items() if k not in ("output_directory", "source_bindings")},
                "Recovered evaluation changed the validation protocol")
        if original_attempt["original_result"] is not None:
            old_result = read(original_attempt["original_result"]["path"])
            require(sha(original_attempt["original_result"]["path"]) == original_attempt["original_result"]["sha256"]
                    and all(terminal["original_attempt_exact_quality_agreement"].values())
                    and all(old_result["results"][0][key] == quality["results"][0][key] for key in
                            ("checkpoint", "tracks", "aggregate", "reconstruction_max_abs")),
                    "Recovered quality does not exactly reproduce the retained original result")
        paths.extend((endpoint / "intent.json", endpoint / "original-attempt.json"))
    else:
        require(original_session == args.actual_root_session, "Wrong completed original launcher")
    require(root_command["record_kind"] == "observed_live_root_command"
            and original_session == prefix["actual_root_session"]
            and prefix["status"] == "pass" and prefix["normalized_fields_equal"]
            and resource["matching_production_updates"] == training["matching_production_updates"]
            and root_command["argv"][3] == "research.direct.run_latency58_branch_memory"
            and "--after-resource" in root_command["argv"],
            "Completed production must match the independently observed rehearsal prefix")
    with (root / "production-run/metrics.jsonl").open() as journal:
        journal_rows = [json.loads(next(journal)) for _ in range(2)]
    require(journal_rows == prefix["production_first_two_rows"]
            and [{k: v for k, v in row.items() if k not in ("elapsed_seconds", "peak_vram_gib")}
                 for row in journal_rows] == resource["matching_production_updates"],
            "The completed journal must retain the exact observed prefix")
    paths.extend((root / "resource-run/result.json", root / "production-run/metrics.jsonl"))
    all_rows = [json.loads(line) for line in (root / "production-run/metrics.jsonl").read_text().splitlines()]
    require(len(all_rows) == plan["config"]["steps"], "Incomplete blended-objective journal")
    config = plan["config"]
    for index, row in enumerate(all_rows):
        if index < config["warmup"]:
            rate = config["lr"] * (index + 1) / config["warmup"]
        else:
            phase = (index - config["warmup"]) / max(1, config["steps"] - 1 - config["warmup"])
            rate = config["min_lr"] + .5 * (config["lr"] - config["min_lr"]) * (1 + math.cos(math.pi * phase))
        require(row["step"] == index + 1 and row["lr"] == rate
                and row["first_sample_index"] == config["data_start"] + index * 16
                and row["next_sample_index"] == config["data_start"] + (index + 1) * 16
                and len(row["microbatches"]) == 1, "Training schedule or batch geometry changed")
        terms = row["microbatches"][0]
        identities = [(terms["reconstruction_loss"], terms["waveform_loss"] + .25 * terms["spectral_loss"] + .25 * terms["raw_anchor"]),
                      (terms["direct_sdr_loss"], terms["negative_sdr_db"] + .5 * terms["absence_db"] + .05 * terms["direct_raw_anchor"]),
                      (terms["loss"], terms["reconstruction_loss"] + SDR_WEIGHT * terms["direct_sdr_loss"])]
        require(all(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=2e-6, abs_tol=2e-6)
                    for actual, expected in identities) and row["loss"] == terms["loss"],
                "Logged blended loss differs from the fixed training objective")
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
            and parent["aggregate"]["full_sdr_db"] == 4.3535847721055445
            and plan["parent_kind"] == "saved_temporal_attention"
            and plan["parent_training_updates"] == 26250
            and plan["config"]["steps"] == 4000
            and plan["optimizer_initialization"] == "fresh_adam"
            and plan["inference_architecture_changed"] is True,
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
    completion_detail = {"completion_mode": "validation_recovery" if recovered else "original_launcher",
                         "actual_root_session": args.actual_root_session,
                         "original_root_session": original_session,
                         "original_root_actual_exit_code": None if recovered else 0,
                         "validation_endpoint_directory": str(endpoint),
                         "training_repeated": False}
    verify_inputs({"source_bindings": bindings})
    require_space(plan, 2_000_000)
    write(root / "root-execution.json", {"actual_exit_code": 0, "actual_root_completion_evidence": completion,
          "source_bindings_unchanged": True, "root_command_sha256": sha(root / "root-command.json"),
          "plan_sha256": sha(plan_path), "result_sha256": sha(endpoint / "result.json"), **completion_detail})
    bindings[str(root / "root-execution.json")] = sha(root / "root-execution.json")
    track_deltas = comparisons["parent"]["metrics"]["full_sdr_db"]["per_track_macro_delta"]
    review = {"status": "selected_for_research" if selected else "not_selected",
              "actual_root_exit_code": 0, "actual_root_completion_evidence": completion,
              **completion_detail,
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
              "additional_public_stream_states": 2, "total_public_stream_states": 8,
              "graph_plus_host_delay_samples": 256, "total_training_updates": 30250,
              "parent_training_updates": 26250, "objective_version": VERSION,
              "direct_sdr_weight": SDR_WEIGHT, "all_4000_blend_journal_identities_checked": True,
              "all_40_saved_optimizer_states_checked": True,
              "added_branch_memory_parameters": 3506000,
              "recipe": RECIPE,
              "continuation_uses_trained_nonzero_attention_head": True,
              "optimizer_initialization": "fresh_adam",
              "interpretation": "Single-seed trial with separate branch GRUs, the parent's blended objective, fresh Adam, a new schedule and a new sample range; not an exact optimizer restart or a matched causal ablation. Track-bootstrap uncertainty excludes training-seed uncertainty and selection effects.",
              "next_direction": "Review all regressions and continue quality work from the selected saved checkpoint until the full goal is proven. A native SDR result alone does not qualify new deployment graph timing.",
              "source_bindings": bindings}
    verify_inputs(review)
    write(root / "selection-review.json", review)
    print(json.dumps({key: review[key] for key in ("status", "best_full_sdr_db", "target_gap_db",
          "full_sdr_improved_tracks", "full_sdr_regressed_tracks", "optimizer_may_be_retired_after_this_review")}), flush=True)


if __name__ == "__main__":
    main()
