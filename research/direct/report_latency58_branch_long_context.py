"""Authenticate two-second-context raw/EMA endpoints and retain every regression."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.report_latency58_branch_sdr_continuation import worst_cells

OBSERVATION_FIELDS = ("elapsed_seconds", "peak_vram_gib", "data_wait_seconds", "compute_and_audit_seconds")


def check_journal(plan, rows, training, resource):
    from research.direct.latency58_branch_sdr_blend import SDR_WEIGHT
    from research.direct.latency58_remix_augmentation import recipe
    from research.direct.latency58_long_context_data import batch_recipes
    from research.direct.latency58_logical_batch_loss import policy as reduction_policy
    config = plan["config"]
    require(len(rows) == config["steps"] and plan["logical_batch_loss"] == reduction_policy()
            and plan["accumulation_steps"] == 2 and config["microbatch_size"] == 8
            and config["batch_size"] == 16 and config["crop_samples"] == 176384
            and plan["warmup_samples"] == 88064 and plan["scored_samples"] == 88320,
            "Incomplete or changed two-second B16 accumulated trajectory")
    for index, row in enumerate(rows):
        if index < config["warmup"]:
            rate = config["lr"] * (index + 1) / config["warmup"]
        else:
            phase = (index - config["warmup"]) / max(1, config["steps"] - 1 - config["warmup"])
            rate = config["min_lr"] + .5 * (config["lr"] - config["min_lr"]) * (1 + math.cos(math.pi * phase))
        first = config["data_start"] + index * 16
        require(row["step"] == row["ema_updates"] == index + 1 and row["lr"] == rate
                and row["first_sample_index"] == first and row["next_sample_index"] == first + 16
                and row["adam_steps_this_update"] == row["gradient_clips_this_update"] == 1
                and len(row["microbatches"]) == 2, "Update schedule or batch geometry changed")
        pitch = batch_recipes(config, first)
        remix = {key: value.tolist() for key, value in recipe(seed=config["seed"], first_sample_index=first).items()}
        require(row["pitch_tempo_recipes"] == pitch and row["augmentation_recipe"] == remix
                and row["pitch_tempo_selected_examples"] == sum(r["selected"] for r in pitch),
                "Pitch/tempo composition or full-B16 remix changed")
        for micro, terms in enumerate(row["microbatches"]):
            require(terms["augmented_examples"] == (4 if micro == 0 else 8)
                    and len(terms["active_windows"]) == len(terms["absent_windows"]) == 4
                    and all(a + b == 16 for a, b in zip(terms["active_windows"], terms["absent_windows"], strict=True)),
                    "Microbatch source composition or two-window activity counts changed")
            identities = [(terms["reconstruction_loss"], terms["waveform_loss"] + .25 * terms["spectral_loss"] + .25 * terms["raw_anchor"]),
                          (terms["direct_sdr_loss"], terms["negative_sdr_db"] + .5 * terms["absence_db"] + .05 * terms["direct_raw_anchor"]),
                          (terms["loss"], terms["reconstruction_loss"] + SDR_WEIGHT * terms["direct_sdr_loss"])]
            require(all(math.isfinite(a) and math.isclose(a, b, rel_tol=2e-6, abs_tol=2e-6) for a, b in identities),
                    "Globally normalized blended-loss identity changed")
        for name, key in (("active_windows", "logical_batch_active_windows"), ("absent_windows", "logical_batch_absent_windows")):
            require(row[key] == [sum(m[name][i] for m in row["microbatches"]) for i in range(4)],
                    "Microbatch eligibility does not sum to the original B16 references")
        require(all(a + b == 32 for a, b in zip(row["logical_batch_active_windows"], row["logical_batch_absent_windows"], strict=True))
                and row["loss"] == sum(m["loss"] for m in row["microbatches"])
                and row["negative_sdr_db"] == sum(m["negative_sdr_db"] for m in row["microbatches"]),
                "Loss contributions were averaged again or lost scored windows")
        for name in ("raw_model_state_sha256", "ema_parameters_sha256", "before_remix_audio_sha256",
                     "after_remix_audio_sha256", "augmented_inputs_sha256"):
            value = row[name]
            require(len(value) == 64 and all(c in "0123456789abcdef" for c in value), "Malformed trajectory identity")
        if index < 4:
            expected = plan["qualified_data_prefix"][index]
            require(row["before_remix_audio_sha256"] == expected["input_sha256"]
                    and row["after_remix_audio_sha256"] == expected["after_remix_sha256"],
                    "Production data prefix differs from four-second CPU qualification")
        require(all(math.isfinite(row[name]) and row[name] >= 0 for name in OBSERVATION_FIELDS)
                and math.isfinite(row["grad_norm"]), "Invalid resource observation or gradient norm")
    normalized = [{k: v for k, v in row.items() if k not in OBSERVATION_FIELDS} for row in rows[:2]]
    require(normalized == training["matching_production_updates"] == resource["matching_production_updates"]
            and rows[-1]["raw_model_state_sha256"] == training["final_raw_model_state_sha256"]
            and rows[-1]["ema_parameters_sha256"] == training["final_ema_parameters_sha256"],
            "Raw/EMA endpoints or rehearsed prefix changed")
    return {"updates_checked": len(rows), "all_loss_and_recipe_identities_checked": True,
            "whole_batch_activity_denominators_and_sum_reduction_checked": True,
            "raw_and_ema_endpoint_journal_identities_match": True,
            "selected_training_crops": sum(r["pitch_tempo_selected_examples"] for r in rows),
            "data_wait_seconds_total": sum(r["data_wait_seconds"] for r in rows),
            "compute_and_audit_seconds_total": sum(r["compute_and_audit_seconds"] for r in rows)}


def main():
    import torch
    from research.direct.compare import compare
    from research.direct.report_latency58_vocal_focus import music_cells
    from research.direct.latency58_sdr_checkpoint import require_space
    from research.direct.latency58_branch_ema_checkpoint import audit_saved, policy as ema_policy
    from research.direct.latency58_long_context_data import policy as data_policy
    from research.direct.prepare_latency58_branch_long_context import RECIPE
    from research.direct.latency58_branch_sdr_blend import VERSION, SDR_WEIGHT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-root-session", required=True, type=int)
    parser.add_argument("--actual-root-exit-code", required=True, type=int, choices=[0])
    parser.add_argument("--actual-root-tool-chunk", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and args.actual_root_session > 0 and args.actual_root_tool_chunk.isalnum(),
            "Provide actual completed root execution evidence")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = PHASE / "branch-long-context-001"
    require(not (root / "selection-review.json").exists() and not (root / "root-execution.json").exists(),
            "Preserve previous endpoint reviews")
    plan_path = root / "plan.json"
    plan = read(plan_path)
    verify_inputs(plan)
    require(plan["objective_version"] == VERSION and plan["direct_sdr_weight"] == SDR_WEIGHT == .2
            and plan["config"]["augmentation"] == data_policy() and plan["ema"] == ema_policy(.995)
            and plan["parent_full_sdr_db"] == 4.46515742201644 and plan["parent_training_updates"] == 39250
            and plan["continuation_kind"] == "selected_ema_branch_two_second_context_and_paired_ema"
            and plan["optimizer_initialization"] == "fresh_adam" and plan["parent_weight_role"] == "ema"
            and not plan["inference_architecture_changed"]
            and all(plan["config"][k] == RECIPE[k] for k in ("steps", "lr", "min_lr", "warmup", "data_start", "seed")),
            "Frozen pitch/EMA recipe or selected parent changed")
    terminal, training, resource = (read(root / name) for name in
        ("result.json", "production-run/result.json", "resource-run/result.json"))
    require(terminal["status"] == "training_audit_and_paired_full14_complete"
            and training["status"] == resource["status"] == "pass"
            and training["source_bindings_unchanged"] and resource["source_bindings_unchanged"]
            and training["updates"] == training["ema_updates"] == plan["config"]["steps"]
            and training["checkpoint_written"] and resource["updates"] == 2 and not resource["checkpoint_written"],
            "Training and resource rehearsal did not complete")
    paths = [plan_path, Path(__file__).resolve(), root / "result.json", root / "production-run/result.json",
             root / "resource-run/result.json", root / "checkpoint-audit.json", root / "root-command.json",
             root / "production-prefix-independent-check.json", root / "paired-full14-comparison.json",
             root / "production-run/metrics.jsonl"]
    parity_path = root / "resource-run/branch-long-context-gpu-parity.json"
    parity = read(parity_path)
    require(parity["status"] == parity["logical_batch_loss"]["status"] == "pass"
            and parity["original_model_state_sha256"] == plan["parent_model_state_sha256"]
            and parity["trained_parent_weights_unchanged"] and parity["cpu_and_cuda_rng_restored"]
            and parity["microbatch_size"] == 8 and parity["scored_samples"] == 88320
            and parity["warmup_input_gradient_zero"] and len(parity["all_40_gradients"]) == 40
            and all(v["maximum_error"] == 0 and v["reference_norm"] > 0 for v in parity["all_40_gradients"].values()),
            "Full two-second CUDA context or logical-loss qualification failed")
    paths.append(parity_path)
    for path in (root / "resource-stage/execution.json", root / "production-stage/execution.json",
                 root / "full14-raw/execution.json", root / "full14-ema/execution.json",
                 PHASE / "branch-long-context-preparation-stage-001/execution.json",
                 PHASE / "branch-long-context-preparation-stage-001/root-execution.json"):
        execution = read(path)
        require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                and not execution.get("timed_out", False), "A required stage failed or remains open")
        paths.append(path)
        if "monitor_result" in execution:
            path = Path(execution["monitor_result"])
            monitor = read(path)
            require(monitor["status"] == monitor["supervisor_health"] == "pass"
                    and monitor["child_exit_code"] == 0 and monitor["post_exit_quiet_completed"], "GPU monitor failed")
            paths.append(path)
    command, prefix = (read(root / name) for name in ("root-command.json", "production-prefix-independent-check.json"))
    verify_inputs(command)
    verify_inputs(prefix)
    require(command["actual_root_session"] == prefix["actual_root_session"] == args.actual_root_session
            and command["record_kind"] == "observed_live_root_command"
            and command["argv"][3] == "research.direct.run_latency58_branch_long_context"
            and "--after-resource" in command["argv"] and prefix["status"] == "pass"
            and prefix["normalized_fields_equal"], "Root launcher or independently observed prefix changed")
    rows = [json.loads(line) for line in (root / "production-run/metrics.jsonl").read_text().splitlines()]
    require(rows[:2] == prefix["production_first_two_rows"], "Observed prefix changed in completed journal")
    journal_audit = check_journal(plan, rows, training, resource)
    checkpoint = training["checkpoint"]
    audit = audit_saved(checkpoint, plan, sha(plan_path))
    previous_audit = read(root / "checkpoint-audit.json")
    require(all(previous_audit[k] == v for k, v in audit.items())
            and audit["algorithmic_latency_samples"] == 256 and audit["saved_optimizer_tensor_count"] == 40
            and audit["model_state_sha256"] == training["final_model_state_sha256"]
            and audit["raw_model_state_sha256"] == training["final_raw_model_state_sha256"], "Saved raw/EMA audit changed")
    generation = Path(checkpoint["path"]).parent
    paths.extend(generation.iterdir())
    parent_path = Path(plan["reference_result"])
    parent_report = read(parent_path)
    verify_inputs(parent_report)
    parent = parent_report["results"][0]
    selection = read(plan["parent_selection_review"])
    require(parent["checkpoint"] == plan["parent_checkpoint"] == selection["best_research_checkpoint"]
            and parent["aggregate"]["full_sdr_db"] == plan["parent_full_sdr_db"] == selection["best_full_sdr_db"],
            "Selected comparison parent changed")
    baseline_path = PHASE / "c204-residual-share-full14-001/result.json"
    baselines = read(baseline_path)["policies"]
    c204, fixed = baselines["working_policy"], baselines["fixed_share"]
    require(c204["aggregate"]["full_sdr_db"] == 4.069078803302578
            and fixed["aggregate"]["full_sdr_db"] == 4.0846618609770395, "Rollback baseline metrics changed")
    references = {"parent": parent, "c204": c204, "preserved_fixed_share": fixed}
    candidates, reports, reviews = {}, {}, {}
    bindings = {**plan["source_bindings"], **command["source_bindings"]}
    for kind in ("raw", "ema"):
        path = root / ("full14-" + kind) / "result.json"
        report = read(path)
        verify_inputs(report)
        candidate = report["results"][0]
        expected_state = training["final_raw_model_state_sha256" if kind == "raw" else "final_model_state_sha256"]
        require(report["status"] == "pass" and report["source_bindings_unchanged"]
                and report["track_count"] == 14 and report["excerpt_count"] == 28
                and report["graph_delay_samples"] == report["host_queue_samples"] == 128
                and candidate["checkpoint"] == terminal["checkpoints"][kind]
                and candidate["model"]["model_state_sha256_after"] == expected_state
                and terminal["quality_results"][kind] == {"path": str(path), "sha256": sha(path)},
                "Paired full14 protocol or endpoint identity changed")
        score = candidate["aggregate"]["full_sdr_db"]
        require(math.isfinite(score) and report["target_reached"] == (score >= 5.)
                and terminal["full_sdr_db"][kind] == score, "Target must use saved full14 SDR")
        comparisons = {key: compare(ref, candidate) for key, ref in references.items()}
        cells = {key: music_cells(ref, candidate) for key, ref in references.items()}
        reviews[kind] = {"full_sdr_db": score, "comparisons": comparisons, "all_track_stem_band_absence": cells,
                         "worst_cells_to_review": {key: worst_cells(value) for key, value in cells.items()}}
        candidates[kind], reports[kind] = candidate, path
        bindings.update(report["source_bindings"])
        paths.extend((path, path.parent / "plan.json"))
    paired = {"comparison": compare(candidates["raw"], candidates["ema"]),
              "all_track_stem_cells": music_cells(candidates["raw"], candidates["ema"]), "quality_selected": False}
    require(paired == read(root / "paired-full14-comparison.json"), "Paired raw/EMA comparison changed")
    best_kind = max(("parent", "raw", "ema"), key=lambda k: (parent if k == "parent" else candidates[k])["aggregate"]["full_sdr_db"])
    best = parent if best_kind == "parent" else candidates[best_kind]
    score = best["aggregate"]["full_sdr_db"]
    paths.extend((parent_path, baseline_path))
    bindings.update({str(path): sha(path) for path in paths})
    verify_inputs({"source_bindings": bindings})
    require_space(plan, 4_000_000)
    completion = f"Unified execution {args.actual_root_session} returned actual exit 0 in tool output {args.actual_root_tool_chunk}."
    write(root / "root-execution.json", {"actual_exit_code": 0, "actual_root_session": args.actual_root_session,
          "actual_root_completion_evidence": completion, "source_bindings_unchanged": True,
          "root_command_sha256": sha(root / "root-command.json"), "plan_sha256": sha(plan_path),
          "result_sha256": sha(root / "result.json")})
    bindings[str(root / "root-execution.json")] = sha(root / "root-execution.json")
    review = {"status": "not_selected" if best_kind == "parent" else "selected_for_research",
              "actual_root_exit_code": 0, "actual_root_session": args.actual_root_session,
              "actual_root_completion_evidence": completion, "selected_weight_role": best_kind,
              "best_research_checkpoint": best["checkpoint"], "best_full_sdr_db": score,
              "best_research_reference_result": str(parent_path if best_kind == "parent" else reports[best_kind]),
              "candidate_saved_full_sdr_db": {k: r["full_sdr_db"] for k, r in reviews.items()},
              "saved_native_quality_target_reached": score >= 5., "target_gap_db": max(0., 5. - score),
              "endpoint_reviews": reviews, "paired_raw_vs_ema": paired, "journal_audit": journal_audit,
              "all_40_saved_optimizer_states_checked": True, "raw_optimizer_owner": "raw-model.pt",
              "raw_optimizer_sha256": sha(generation / "raw-optimizer.pt"),
              "raw_and_ema_and_optimizer_retained": True, "parent_and_rollback_baselines_preserved": True,
              "total_training_updates": plan["parent_training_updates"] + plan["config"]["steps"],
              "graph_plus_host_delay_samples": 256, "total_public_stream_states": 8,
              "plugin_replaced": False, "new_weights_native_timing_measured": False,
              "artifact_cap_bytes": plan["stop_counted_bytes"] + plan["outside_roots_reservation_bytes"],
              "recipe": RECIPE, "ema": plan["ema"], "augmentation": data_policy(),
              "logical_batch_loss": plan["logical_batch_loss"], "warmup_samples": plan["warmup_samples"],
              "scored_samples": plan["scored_samples"], "logical_batch_size": 16, "microbatch_size": 8,
              "interpretation": "A single-seed continuation doubles scored context from one to two seconds and changes seed and crop range, starting from selected EMA weights with fresh Adam. Logical B16 eligibility denominators preserve the full-batch objective across two B8 microbatches. Raw versus EMA compares one shared trajectory. Paired-track intervals exclude training-seed uncertainty and repeated selection. EMA inference has no matching raw Adam optimizer; restart requires its explicit fresh-optimizer policy or the retained raw/EMA generation.",
              "source_bindings": bindings}
    verify_inputs(review)
    write(root / "selection-review.json", review)
    print(json.dumps({k: review[k] for k in ("status", "selected_weight_role", "best_full_sdr_db", "candidate_saved_full_sdr_db", "target_gap_db")}), flush=True)


if __name__ == "__main__":
    main()
