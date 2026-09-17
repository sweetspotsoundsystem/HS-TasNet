"""Freeze source-controlled continuation from the completed SDR-blend review."""
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs

RECIPE = {"steps": 1000, "lr": 3e-5, "min_lr": 3e-6, "warmup": 100,
          "data_start": 3_800_000, "seed": 20261023, "optimizer_initialization": "fresh_adam"}


def main():
    import json
    import torch
    from research.direct.latency58_branch_memory_checkpoint import load_model, audit_saved
    from research.direct.latency58_branch_memory import ADAPTERS
    from research.direct.latency58_branch_controlled_remix import AUGMENTATION
    from research.direct.check_latency58_branch_continuation_checkpoint import check as check_checkpoint
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require single-threaded CPU preparation")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    latest = PHASE / "branch-sdr-blend-001"
    review_path = latest / "selection-review.json"
    review = read(review_path)
    verify_inputs(review)
    require(review["actual_root_exit_code"] == 0 and not review["saved_native_quality_target_reached"]
            and review["status"] in ("selected_for_research", "not_selected")
            and review["best_full_sdr_db"] >= 4.391035784116766,
            "Require a completed latest review below the target")
    checkpoint = review["best_research_checkpoint"]
    parent_root = Path(checkpoint["path"]).parents[2]
    source_path = parent_root / "plan.json"
    source = read(source_path)
    verify_inputs(source)
    quality_path = Path(review["best_research_reference_result"])
    quality = read(quality_path)
    require(quality["status"] == "pass" and quality["track_count"] == 14 and quality["excerpt_count"] == 28
            and quality["results"][0]["checkpoint"] == checkpoint
            and quality["results"][0]["aggregate"]["full_sdr_db"] == review["best_full_sdr_db"],
            "The selected parent must be the exact saved full14 endpoint")
    budget = {**source, "stop_counted_bytes": 89_200_000_000, "artifact_cap_bytes": 90_000_000_000,
              "artifact_cap_authorization": "User: you can use 10 GB more; 2026-09-13 UTC"}
    require(budget["stop_counted_bytes"] + budget["outside_roots_reservation_bytes"] == budget["artifact_cap_bytes"],
            "Retain the outside-roots allowance")
    paths = [Path(__file__).resolve(), source_path, review_path, quality_path]
    for path in (latest / "root-execution.json", PHASE / "branch-sdr-blend-review-stage-001/execution.json",
                 PHASE / "branch-sdr-blend-review-stage-001/root-execution.json",
                 parent_root / "production-stage/execution.json", quality_path.parent / "execution.json",
                 PHASE / "branch-controlled-remix-functional-001/execution.json",
                 PHASE / "branch-controlled-remix-functional-001/root-execution.json"):
        execution = read(path)
        require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                and not execution.get("timed_out", False), "A required preceding execution did not close")
        paths.append(path)
    proof_path = PHASE / "branch-controlled-remix-functional-001/result.json"
    proof = read(proof_path)
    require(proof["status"] == "pass" and proof["augmentation"] == AUGMENTATION
            and proof["all_twelve_ordinary_examples_bit_exact_to_current_remix"]
            and proof["controlled_views_sum_exactly_to_their_targets"]
            and proof["current_blended_objective_has_restoring_silent_source_gradient"],
            "Source-controlled CPU qualification is incomplete")
    paths.extend((proof_path, PHASE / "branch-controlled-remix-functional-001/plan.json"))
    verify_inputs(read(PHASE / "branch-controlled-remix-functional-001/plan.json"))
    audit = audit_saved(checkpoint, source, sha(source_path))
    parent, payload = load_model(checkpoint)
    parent_sha = state_sha256(parent.state_dict())
    parent_updates = parent.provenance["training_updates"]
    require(audit["status"] == "pass" and audit["saved_optimizer_tensor_count"] == 40
            and parent_sha == audit["model_state_sha256"]
            and all(torch.count_nonzero(dict(parent.named_parameters())[name]) > 0 for name in ADAPTERS),
            "Selected parent or trained private memories failed their audit")
    weight = source["direct_sdr_weight"]
    require(weight in (.1, .2), "Keep the selected parent's blended objective")
    if weight == .2:
        from research.direct.latency58_branch_sdr_blend import VERSION
    else:
        from research.direct.latency58_attention_sdr_blend import VERSION
    require(VERSION == source["objective_version"], "Selected objective source differs")
    paths.extend(parent_root / name for name in ("checkpoint-audit.json", "production-run/checkpoint/model.pt",
                 "production-run/checkpoint/optimizer.pt", "production-run/checkpoint/receipt.json"))
    files = ("latency58_branch_controlled_remix.py", "train_latency58_branch_controlled_remix.py",
             "run_latency58_branch_controlled_remix.py", "report_latency58_branch_controlled_remix.py",
             "observe_latency58_branch_controlled_prefix.py")
    paths.extend(ROOT / "research/direct" / name for name in files)
    bindings = {**source["source_bindings"], **review["source_bindings"], **{str(p): sha(p) for p in paths}}
    verify_inputs({"source_bindings": bindings})
    out = PHASE / "branch-controlled-remix-001"
    require(not out.exists(), "Preserve earlier continuation plans")
    counted = require_space(budget, 455_000_000)
    out.mkdir()
    began = time.monotonic()
    fixture_source = {**source, "parent_checkpoint": checkpoint, "parent_model_state_sha256": parent_sha,
                      "parent_training_updates": parent_updates, "parent_kind": "saved_trained_branch_memory",
                      "inference_architecture_changed": False}
    functional = check_checkpoint(fixture_source, out)
    write(out / "current-parent-checkpoint-functional.json", functional)
    require(functional["status"] == "pass" and functional["all_40_optimizer_states_checked"]
            and functional["resumed_third_update_and_adam_moments_bit_exact"]
            and state_sha256(parent.state_dict()) == parent_sha and not torch.cuda.is_initialized()
            and parent.algorithmic_latency_samples == 256, "Current parent qualification failed")
    paths.extend(out / name for name in ("fixture-plan.json", "current-parent-checkpoint-functional.json"))
    config = {**source["config"], **{k: RECIPE[k] for k in ("steps", "lr", "min_lr", "warmup", "data_start", "seed")},
              "checkpoint_every": RECIPE["steps"], "augmentation": AUGMENTATION,
              "source_view_counts": {"ordinary": 12, "instrumental": 2, "vocals_only": 2},
              "excluded_source_gain": 0., "retained_source_gain_bounds_db": [-3, 3]}
    plan = {**budget, "name": out.name, "output_directory": str(out), "config": config,
            "source_bindings": {**bindings, **{str(p): sha(p) for p in paths}},
            "parent_checkpoint": checkpoint, "parent_kind": "saved_trained_branch_memory",
            "parent_model_state_sha256": parent_sha, "parent_training_updates": parent_updates,
            "optimizer_initialization": "fresh_adam", "reference_result": str(quality_path),
            "parent_selection_review": str(review_path), "parent_full_sdr_db": review["best_full_sdr_db"],
            "initialized_model_state_sha256": parent_sha,
            "fixed_buffers_sha256": state_sha256(dict(parent.named_buffers())),
            "parameter_names": [name for name, _ in parent.named_parameters()],
            "inference_architecture": parent.architecture_metadata, "inference_architecture_changed": False,
            "objective_version": VERSION, "direct_sdr_weight": weight, "quality_endpoints": [1000],
            "source_gain_bounds_scope": "Retained sources only; excluded sources have exactly zero multiplier",
            "excluded_source_multiplier": 0., "controlled_source_views_cover_warmup_and_scoring": True,
            "augmentation_fractions": {"untouched": .25, "same_crop_transformed": .25,
                                       "ordinary_cross_crop": .25, "instrumental_cross_crop": .125,
                                       "vocals_only_cross_crop": .125},
            "source_view_fractions": {"ordinary": .75, "instrumental": .125, "vocals_only": .125},
            "cross_crop_source_selection": "Four distinct nonzero cyclic base shifts; controlled rows then zero excluded sources",
            "continuation_uses_trained_nonzero_branch_memories": True,
            "continuation_kind": "trained_branch_memory_quarter_controlled_source_views",
            "native_cost_measured": False, "new_weights_native_cost_measured": False}
    verify_inputs(plan)
    write(out / "preparation.json", {"status": "pass", "parent_full_sdr_db": review["best_full_sdr_db"],
          "parent_audit": audit, "recipe": RECIPE, "objective_version": VERSION, "direct_sdr_weight": weight,
          "augmentation": AUGMENTATION, "source_view_counts": config["source_view_counts"],
          "current_parent_context_and_ram_checkpoint_passed": True, "all_40_tensors_trainable_in_production": True,
          "optimizer_initialization": "fresh_adam", "target_full_sdr_db": 5.0, "graph_plus_host_samples": 256,
          "total_public_stream_states": 8, "additional_parameters": 0, "inference_architecture_changed": False,
          "counted_bytes_before": counted, "reserved_training_bytes": 455_000_000,
          "forecast_including_outside_and_run": counted + budget["outside_roots_reservation_bytes"] + 455_000_000,
          "artifact_cap_bytes": budget["artifact_cap_bytes"], "cuda_initialized": False,
          "elapsed_seconds": time.monotonic() - began,
          "rationale": "The fixed training-view ablation found a small SDR benefit from the saved branch memories alongside an absent-vocal cost. Keep the selected parent's architecture and objective, preserve twelve ordinary remixes, and supervise four physical instrumental/vocal-only views to practice absent-source rejection.",
          "limitation": "Single-seed augmentation and schedule continuation with fresh Adam and a new crop range. No matched causal attribution, held-out quality gain, vocal rejection improvement or native timing is established by preparation."})
    plan["source_bindings"][str(out / "preparation.json")] = sha(out / "preparation.json")
    require_space(plan, 450_000_000)
    write(out / "plan.json", plan)
    print(json.dumps({"event": "prepared", "plan": str(out / "plan.json"), "sha256": sha(out / "plan.json"),
                      "initialized_model_state_sha256": parent_sha, "parent_full_sdr_db": review["best_full_sdr_db"]}), flush=True)


if __name__ == "__main__":
    main()
