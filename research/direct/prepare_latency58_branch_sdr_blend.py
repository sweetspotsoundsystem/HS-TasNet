"""Freeze a 1000-update 0.2 SDR blend from the reviewed, trained branch memories."""
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs

RECIPE = {"steps": 1000, "lr": 3e-5, "min_lr": 3e-6, "warmup": 100,
          "data_start": 3_700_000, "seed": 20261022, "optimizer_initialization": "fresh_adam"}


def main():
    import torch
    from research.direct.latency58_branch_memory_checkpoint import load_model, audit_saved
    from research.direct.latency58_branch_sdr_blend import VERSION, SDR_WEIGHT, check as check_objective
    from research.direct.latency58_branch_memory import ADAPTERS
    from research.direct.check_latency58_branch_continuation_checkpoint import check as check_checkpoint
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require single-threaded CPU preparation")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    parent_root = PHASE / "branch-memory-001"
    endpoint = parent_root / "validation-recovery-001"
    source_path = parent_root / "plan.json"
    source, review = read(source_path), read(parent_root / "selection-review.json")
    # The user increased the total artifact allowance by 10 GB on 2026-09-13 UTC.
    # Preserve the parent's frozen 80 GB plan and apply the new cap prospectively.
    budget_source = {**source, "stop_counted_bytes": 89_200_000_000,
                     "artifact_cap_bytes": 90_000_000_000,
                     "artifact_cap_authorization": "User: you can use 10 GB more; 2026-09-13 UTC"}
    require(budget_source["stop_counted_bytes"] + source["outside_roots_reservation_bytes"]
            == budget_source["artifact_cap_bytes"], "Artifact accounting must retain the outside-roots allowance")
    terminal, quality = read(endpoint / "result.json"), read(endpoint / "full14/result.json")
    require(review["status"] == "selected_for_research" and review["actual_root_exit_code"] == 0
            and review["best_full_sdr_db"] == terminal["full_sdr_db"] == 4.391035784116766 < 5.0
            and review["optimizer_may_be_retired_after_this_review"] is False
            and review["original_root_actual_exit_code"] is None
            and review["best_research_reference_result"] == str(endpoint / "full14/result.json")
            and terminal["status"] == "training_audit_and_full14_complete"
            and quality["status"] == "pass" and quality["track_count"] == 14 and quality["excerpt_count"] == 28
            and quality["results"][0]["checkpoint"] == review["best_research_checkpoint"] == terminal["checkpoint"],
            "Require the reviewed branch-memory parent and its actual completed recovery")
    verify_inputs(source)
    verify_inputs(review)
    paths = [Path(__file__).resolve(), source_path]
    for path in (parent_root / "production-stage/execution.json", endpoint / "full14/execution.json",
                 parent_root / "root-execution.json", PHASE / "branch-memory-review-stage-001/execution.json"):
        execution = read(path)
        require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                and not execution.get("timed_out", False), "Prior execution is incomplete")
        paths.append(path)
    paths.extend(parent_root / name for name in ("selection-review.json", "checkpoint-audit.json",
                 "production-run/checkpoint/model.pt", "production-run/checkpoint/optimizer.pt",
                 "production-run/checkpoint/receipt.json"))
    paths.extend(endpoint / name for name in ("result.json", "full14/result.json"))
    audit = audit_saved(terminal["checkpoint"], source, sha(source_path))
    parent, payload = load_model(terminal["checkpoint"])
    parent_sha = state_sha256(parent.state_dict())
    require(audit["status"] == "pass" and audit["saved_optimizer_tensor_count"] == 40
            and payload["step"] == 4000 and parent.provenance["training_updates"] == 30250
            and parent_sha == "d8086c12f3ba5dfacfe94f14bebf4b9694882c6fbe803baa70bf869ca4f2c3fd"
            and all(torch.count_nonzero(dict(parent.named_parameters())[name]) > 0 for name in ADAPTERS),
            "Selected parent audit, trained memories or lineage changed")
    storage = PHASE / "branch-sdr-blend-storage-001"
    receipt = read(storage / "receipt.json")
    require(receipt["status"] == "complete" and receipt["preserved_bindings_unchanged"]
            and receipt["all_inference_models_and_source_audio_preserved"]
            and receipt["all_selected_optimizers_preserved"] and receipt["reserved_bytes"] == 460_000_000
            and receipt["intent_sha256"] == sha(storage / "intent.json"), "Continuation storage reservation incomplete")
    paths.extend(storage / name for name in ("intent.json", "receipt.json"))
    files = ("latency58_branch_sdr_blend.py", "check_latency58_branch_continuation_checkpoint.py",
             "train_latency58_branch_sdr_blend.py", "run_latency58_branch_sdr_blend.py")
    paths.extend(ROOT / "research/direct" / name for name in files)
    bindings = {**source["source_bindings"], **{str(path): sha(path) for path in paths}}
    verify_inputs({"source_bindings": bindings})
    out = PHASE / "branch-sdr-blend-001"
    require(not out.exists(), "Preserve earlier continuation plans")
    counted = require_space(budget_source, 455_000_000)
    out.mkdir()
    began = time.monotonic()
    objective = check_objective()
    write(out / "objective-functional.json", objective)
    fixture_source = {**source, "parent_checkpoint": terminal["checkpoint"],
                      "parent_model_state_sha256": parent_sha, "parent_training_updates": 30250,
                      "parent_kind": "saved_trained_branch_memory", "objective_version": VERSION,
                      "direct_sdr_weight": SDR_WEIGHT, "inference_architecture_changed": False}
    checkpoint = check_checkpoint(fixture_source, out)
    write(out / "current-parent-checkpoint-functional.json", checkpoint)
    require(objective["status"] == checkpoint["status"] == "pass"
            and checkpoint["all_40_optimizer_states_checked"]
            and checkpoint["resumed_third_update_and_adam_moments_bit_exact"], "CPU qualification failed")
    require(state_sha256(parent.state_dict()) == parent_sha and not torch.cuda.is_initialized()
            and parent.algorithmic_latency_samples == 256 and len(list(parent.parameters())) == 40,
            "Preparation changed the parent, state inventory or algorithmic delay")
    paths.extend(out / name for name in ("objective-functional.json", "fixture-plan.json",
                                        "current-parent-checkpoint-functional.json"))
    config = {**source["config"], **{k: RECIPE[k] for k in ("steps", "lr", "min_lr", "warmup", "data_start", "seed")},
              "checkpoint_every": RECIPE["steps"]}
    plan = {**budget_source, "name": out.name, "output_directory": str(out), "config": config,
            "source_bindings": {**bindings, **{str(path): sha(path) for path in paths}},
            "parent_checkpoint": terminal["checkpoint"], "parent_kind": "saved_trained_branch_memory",
            "parent_model_state_sha256": parent_sha, "parent_training_updates": 30250,
            "optimizer_initialization": "fresh_adam", "reference_result": str(endpoint / "full14/result.json"),
            "initialized_model_state_sha256": parent_sha,
            "fixed_buffers_sha256": state_sha256(dict(parent.named_buffers())),
            "parameter_names": [name for name, _ in parent.named_parameters()],
            "inference_architecture": parent.architecture_metadata, "inference_architecture_changed": False,
            "objective_version": VERSION, "direct_sdr_weight": SDR_WEIGHT, "quality_endpoints": [1000],
            "continuation_uses_trained_nonzero_branch_memories": True,
            "continuation_kind": "trained_branch_memory_stronger_sdr",
            "native_cost_measured": False, "new_weights_native_cost_measured": False}
    verify_inputs(plan)
    write(out / "preparation.json", {"status": "pass", "parent_full_sdr_db": terminal["full_sdr_db"],
          "parent_audit": audit, "recipe": RECIPE, "objective_version": VERSION, "direct_sdr_weight": SDR_WEIGHT,
          "current_parent_context_and_ram_checkpoint_passed": True, "objective_functional_passed": True,
          "all_40_tensors_trainable_in_production": True, "optimizer_initialization": "fresh_adam",
          "target_full_sdr_db": 5.0, "graph_plus_host_samples": 256, "total_public_stream_states": 8,
          "additional_parameters": 0, "inference_architecture_changed": False,
          "counted_bytes_before": counted, "reserved_training_bytes": 455_000_000,
          "forecast_including_outside_and_run": counted + source["outside_roots_reservation_bytes"] + 455_000_000,
          "artifact_cap_bytes": budget_source["artifact_cap_bytes"],
          "artifact_cap_authorization": budget_source["artifact_cap_authorization"],
          "cuda_initialized": False, "pr13_model_preserved": True, "elapsed_seconds": time.monotonic() - began,
          "rationale": "The 0.1 SDR blend and trained branch memories each improved saved full14 SDR. Continue the exact trained weights for 1000 updates with the fixed SDR blend raised to 0.2, retaining reconstruction supervision, training corpus, augmentations and latency. Include every track/stem/band/absence regression in the endpoint review.",
          "regressions_carried_forward": "The parent regresses in 18/56 SDR cells, 15/56 SIR cells and 17/23 eligible absence cells against its parent. Bass low-band SDR declines; Skelpolu remains substantially worse than C204 for Other SIR and absent vocal output.",
          "limitation": "Single-seed loss-weight and training-schedule experiment with a new crop range and fresh Adam. No matched causal effect of the coefficient alone, quality gain or native timing is established by qualification."})
    plan["source_bindings"][str(out / "preparation.json")] = sha(out / "preparation.json")
    require_space(plan, 450_000_000)
    write(out / "plan.json", plan)
    print(json.dumps({"event": "prepared", "plan": str(out / "plan.json"), "sha256": sha(out / "plan.json"),
                      "initialized_model_state_sha256": parent_sha}), flush=True)


if __name__ == "__main__":
    main()
