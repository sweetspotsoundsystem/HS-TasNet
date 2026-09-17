"""Continue the retained SDR-blend parent after the controlled-remix review."""
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs

RECIPE = {"steps": 4000, "lr": 6e-5, "min_lr": 6e-6, "warmup": 100,
          "data_start": 3_900_000, "seed": 20261024, "optimizer_initialization": "fresh_adam"}


def main():
    import torch
    from research.direct.latency58_branch_memory_checkpoint import load_model, audit_saved
    from research.direct.latency58_branch_sdr_blend import VERSION, SDR_WEIGHT
    from research.direct.latency58_branch_memory import ADAPTERS
    from research.direct.latency58_remix_augmentation import AUGMENTATION
    from research.direct.check_latency58_branch_continuation_checkpoint import check as check_checkpoint
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require single-threaded CPU preparation")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    latest = PHASE / "branch-controlled-remix-001"
    review_path = latest / "selection-review.json"
    review = read(review_path)
    verify_inputs(review)
    require(review["status"] == "not_selected" and review["actual_root_exit_code"] == 0
            and not review["saved_native_quality_target_reached"]
            and review["best_full_sdr_db"] == 4.395331681151562
            and review["candidate_saved_full_sdr_db"] == 4.364384841074201,
            "Require the completed controlled-remix rejection and retained best endpoint")
    checkpoint = review["best_research_checkpoint"]
    parent_root = Path(checkpoint["path"]).parents[2]
    source_path = parent_root / "plan.json"
    source = read(source_path)
    quality_path = Path(review["best_research_reference_result"])
    quality = read(quality_path)
    verify_inputs(source)
    verify_inputs(quality)
    require(parent_root == PHASE / "branch-sdr-blend-001"
            and quality["status"] == "pass" and quality["track_count"] == 14 and quality["excerpt_count"] == 28
            and quality["results"][0]["checkpoint"] == checkpoint
            and quality["results"][0]["aggregate"]["full_sdr_db"] == review["best_full_sdr_db"]
            and source["objective_version"] == VERSION and source["direct_sdr_weight"] == SDR_WEIGHT == .2
            and source["config"]["augmentation"] == AUGMENTATION,
            "Retain the selected saved model, objective and ordinary-mixture augmentation")
    budget = {key: source[key] for key in ("counted_roots", "stop_counted_bytes", "outside_roots_reservation_bytes")}
    require(budget["stop_counted_bytes"] + budget["outside_roots_reservation_bytes"] == 90_000_000_000,
            "Retain the current artifact cap")
    paths = [Path(__file__).resolve(), source_path, review_path, quality_path]
    for path in (latest / "root-execution.json", PHASE / "branch-controlled-remix-review-stage-001/execution.json",
                 PHASE / "branch-controlled-remix-review-stage-001/root-execution.json",
                 parent_root / "root-execution.json", parent_root / "production-stage/execution.json",
                 quality_path.parent / "execution.json"):
        execution = read(path)
        require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                and not execution.get("timed_out", False), "A required prior execution remains incomplete")
        paths.append(path)
    paths.extend(parent_root / name for name in ("checkpoint-audit.json", "production-run/checkpoint/model.pt",
                 "production-run/checkpoint/optimizer.pt", "production-run/checkpoint/receipt.json"))
    audit = audit_saved(checkpoint, source, sha(source_path))
    parent, _ = load_model(checkpoint)
    parent_sha = state_sha256(parent.state_dict())
    parent_updates = parent.provenance["training_updates"]
    require(audit["status"] == "pass" and audit["saved_optimizer_tensor_count"] == 40
            and parent_sha == audit["model_state_sha256"] == "24cc71db393e3f3f1ec7cc70a9325271d5c4ec2df47b96f9ae81cc4be229301b"
            and parent_updates == source["parent_training_updates"] + source["config"]["steps"] == 31250
            and all(torch.count_nonzero(dict(parent.named_parameters())[name]) > 0 for name in ADAPTERS),
            "The selected parent's audited weights, optimizer or lineage changed")
    files = ("latency58_branch_sdr_blend.py", "check_latency58_branch_continuation_checkpoint.py",
             "train_latency58_branch_sdr_blend.py", "run_latency58_branch_sdr_blend.py",
             "report_latency58_branch_sdr_continuation.py", "observe_latency58_branch_sdr_continuation_prefix.py")
    paths.extend(ROOT / "research/direct" / name for name in files)
    require(all(source["source_bindings"].get(path, digest) == digest
                for path, digest in review["source_bindings"].items()), "Preceding evidence binds conflicting source bytes")
    bindings = {**source["source_bindings"], **review["source_bindings"], **{str(path): sha(path) for path in paths}}
    verify_inputs({"source_bindings": bindings})
    out = PHASE / "branch-sdr-continuation-001"
    require(not out.exists(), "Preserve earlier continuation plans")
    # Includes the new model, optimizer, 4000-row journal and concurrent vocal diagnostics.
    counted = require_space(budget, 465_000_000)
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
            and parent.algorithmic_latency_samples == 256, "Current-parent qualification failed")
    paths.extend(out / name for name in ("fixture-plan.json", "current-parent-checkpoint-functional.json"))
    config = {**source["config"], **{k: RECIPE[k] for k in ("steps", "lr", "min_lr", "warmup", "data_start", "seed")},
              "checkpoint_every": RECIPE["steps"]}
    plan = {**source, **budget, "name": out.name, "output_directory": str(out), "config": config,
            "source_bindings": {**bindings, **{str(path): sha(path) for path in paths}},
            "parent_checkpoint": checkpoint, "parent_kind": "saved_trained_branch_memory",
            "parent_model_state_sha256": parent_sha, "parent_training_updates": parent_updates,
            "optimizer_initialization": "fresh_adam", "reference_result": str(quality_path),
            "parent_selection_review": str(review_path), "parent_full_sdr_db": review["best_full_sdr_db"],
            "initialized_model_state_sha256": parent_sha,
            "fixed_buffers_sha256": state_sha256(dict(parent.named_buffers())),
            "parameter_names": [name for name, _ in parent.named_parameters()],
            "inference_architecture": parent.architecture_metadata, "inference_architecture_changed": False,
            "objective_version": VERSION, "direct_sdr_weight": SDR_WEIGHT, "quality_endpoints": [4000],
            "continuation_uses_trained_nonzero_branch_memories": True,
            "continuation_kind": "retained_branch_sdr_blend_longer_ordinary_mixture_training",
            "native_cost_measured": False, "new_weights_native_cost_measured": False}
    verify_inputs(plan)
    write(out / "preparation.json", {"status": "pass", "parent_full_sdr_db": review["best_full_sdr_db"],
          "parent_audit": audit, "recipe": RECIPE, "objective_version": VERSION, "direct_sdr_weight": SDR_WEIGHT,
          "augmentation": AUGMENTATION, "current_parent_context_and_ram_checkpoint_passed": True,
          "all_40_tensors_trainable_in_production": True, "optimizer_initialization": "fresh_adam",
          "target_full_sdr_db": 5.0, "graph_plus_host_samples": 256, "total_public_stream_states": 8,
          "additional_parameters": 0, "inference_architecture_changed": False,
          "counted_bytes_before": counted, "reserved_training_and_diagnostic_bytes": 465_000_000,
          "forecast_including_outside_and_run": counted + budget["outside_roots_reservation_bytes"] + 465_000_000,
          "artifact_cap_bytes": 90_000_000_000, "cuda_initialized": False,
          "elapsed_seconds": time.monotonic() - began,
          "rationale": "The controlled-remix endpoint reduced all four full-band stem means and every low-band stem mean. Continue from the retained best ordinary-mixture parent with its unchanged 0.2 blended objective and a longer 4000-update schedule at 6e-5 to 6e-6. The rejected endpoint receives separate source-view diagnostics while this trial is prepared.",
          "limitation": "Single-seed schedule continuation with fresh Adam and a new crop range. No matched causal attribution, quality gain or new native timing is established by preparation. The retained parent's interference and absence regressions remain part of the next complete review."})
    plan["source_bindings"][str(out / "preparation.json")] = sha(out / "preparation.json")
    require_space(plan, 460_000_000)
    write(out / "plan.json", plan)
    print(json.dumps({"event": "prepared", "plan": str(out / "plan.json"), "sha256": sha(out / "plan.json"),
                      "initialized_model_state_sha256": parent_sha, "parent_full_sdr_db": review["best_full_sdr_db"]}), flush=True)


if __name__ == "__main__":
    main()
