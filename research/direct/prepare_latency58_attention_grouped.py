"""Prepare a grouped-rate trial only after the continuation review and storage reservation."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs

RECIPE = {"steps": 1000, "lr": 1e-5, "min_lr": 1e-6, "warmup": 50,
          "attention_lr_multiplier": 100., "data_start": 3_400_000, "seed": 20261019,
          "optimizer_initialization": "fresh_adam"}


def main():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require single-threaded CPU preparation")
    completed = PHASE / "attention-continuation-001"
    required = [completed / name for name in (
        "selection-review.json", "root-execution.json", "result.json", "production-stage/execution.json",
        "full14/execution.json", "full14/result.json", "checkpoint-audit.json")]
    require(all(path.is_file() for path in required),
            "Finish the continuation, saved audit, full14 evaluation and review before preparing a new trial")
    review = read(completed / "selection-review.json")
    root_execution = read(completed / "root-execution.json")
    require(review["status"] in ("selected_for_research", "not_selected")
            and review["actual_root_exit_code"] == root_execution["actual_exit_code"] == 0
            and root_execution["source_bindings_unchanged"]
            and root_execution["result_sha256"] == sha(completed / "result.json")
            and math.isfinite(review["best_full_sdr_db"])
            and 4.288099064999147 <= review["best_full_sdr_db"] < 5.0,
            "Use the reviewed native best below target; a reached target needs the full completion audit")
    verify_inputs(review)
    for name in ("production-stage/execution.json", "full14/execution.json"):
        execution = read(completed / name)
        require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                and not execution.get("timed_out", False), "The prior phase did not close successfully")
    production_execution = read(completed / "production-stage/execution.json")
    monitor_path = Path(production_execution["monitor_result"])
    monitor = read(monitor_path)
    require(monitor["status"] == monitor["supervisor_health"] == "pass" and monitor["child_exit_code"] == 0
            and monitor["post_exit_quiet_completed"], "The previous GPU monitor must close successfully")
    parent_quality_path = Path(review["best_research_reference_result"]).resolve(strict=True)
    require(parent_quality_path in (completed / "full14/result.json",
                                   PHASE / "temporal-attention-001/full14/result.json"),
            "Use the reviewed continuation endpoint or its retained pilot")
    parent_root = parent_quality_path.parent.parent
    source_path = parent_root / "plan.json"
    source, quality = read(source_path), read(parent_quality_path)
    verify_inputs(source)
    verify_inputs(quality)
    require(source["schema"] == "latency58-temporal-attention-training-plan-v1"
            and quality["status"] == "pass" and quality["track_count"] == 14 and quality["excerpt_count"] == 28
            and quality["source_bindings_unchanged"]
            and quality["graph_delay_samples"] == quality["host_queue_samples"] == 128
            and quality["results"][0]["checkpoint"] == review["best_research_checkpoint"]
            and quality["results"][0]["aggregate"]["full_sdr_db"] == review["best_full_sdr_db"],
            "The selected saved parent must match its complete unchanged full14 result")
    import torch
    from research.direct.latency58_temporal_attention_checkpoint import load_model, audit_saved
    from research.direct.latency58_attention_grouped_checkpoint import OPTIMIZER_SCHEMA, SCHEDULE_SCHEMA
    from research.direct.latency58_sdr_checkpoint import require_space
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    audit = audit_saved(review["best_research_checkpoint"], source, sha(source_path))
    model, payload = load_model(review["best_research_checkpoint"])
    fingerprint = state_sha256(model.state_dict())
    require(audit["status"] == "pass" and audit["saved_optimizer_tensor_count"] == 30
            and audit["model_state_sha256"] == fingerprint == quality["results"][0]["model"]["model_state_sha256_after"]
            and model.provenance["training_updates"] in (21250, 25250)
            and bool(torch.count_nonzero(model.temporal_output.weight)), "Saved parent audit or attention identity failed")
    memory_root = PHASE / "attention-grouped-memory-functional-001"
    memory, memory_plan = read(memory_root / "result.json"), read(memory_root / "plan.json")
    memory_execution_path = PHASE / "attention-grouped-memory-functional-stage-001/execution.json"
    memory_execution = read(memory_execution_path)
    require(memory["status"] == "pass" and memory["source_bindings_unchanged"]
            and memory["all_30_optimizer_states_checked"] and memory["optimizer_group_sizes"] == [26, 4]
            and memory["fixture_attention_lr_multiplier"] == RECIPE["attention_lr_multiplier"]
            and memory["resumed_third_update_and_adam_moments_bit_exact"]
            and memory["original_inference_loader_outputs_and_states_bit_exact"]
            and memory["constructor_rates_replaced_by_saved_rates"]
            and memory_execution["actual_exit_code"] == 0 and memory_execution["source_bindings_unchanged"]
            and not memory_execution["timed_out"], "Grouped optimizer checkpoint mechanics must pass first")
    verify_inputs(memory_plan)
    storage_root = PHASE / "attention-grouped-storage-001"
    storage = read(storage_root / "receipt.json")
    require(storage["status"] == "complete" and storage["preserved_bindings_unchanged"]
            and storage["all_inference_models_and_source_audio_preserved"]
            and storage["reserved_bytes"] == 400_000_000, "Reserve storage before preparing the trial")
    out = PHASE / "attention-grouped-001"
    require(not out.exists(), "Preserve prior grouped-rate trials")
    paths = [Path(__file__).resolve(), *required, monitor_path, source_path, parent_quality_path,
             memory_root / "plan.json", memory_root / "result.json", memory_execution_path,
             storage_root / "intent.json", storage_root / "receipt.json",
             PHASE / "attention-parent-ablation-002/analysis.json"]
    paths.extend(parent_root / name for name in (
        "result.json", "production-stage/execution.json", "full14/execution.json", "checkpoint-audit.json",
        "production-run/checkpoint/model.pt", "production-run/checkpoint/optimizer.pt", "production-run/checkpoint/receipt.json"))
    paths.extend(ROOT / "research/direct" / name for name in (
        "latency58_attention_grouped_checkpoint.py", "check_latency58_attention_grouped_memory.py",
        "train_latency58_attention_grouped.py", "run_latency58_attention_grouped.py"))
    config = {**source["config"], **{key: RECIPE[key] for key in ("steps", "lr", "min_lr", "warmup", "data_start", "seed")},
              "checkpoint_every": RECIPE["steps"]}
    require(config["warmup"] < config["steps"] - 1, "The cosine schedule must reach its declared minimum")
    plan = {**source, "schema": "latency58-attention-grouped-training-plan-v1", "name": out.name,
            "output_directory": str(out), "config": config,
            "source_bindings": {**source["source_bindings"], **review["source_bindings"], **memory_plan["source_bindings"],
                                **{str(path): sha(path) for path in paths}},
            "parent_checkpoint": review["best_research_checkpoint"], "parent_kind": "saved_temporal_attention",
            "optimizer_initialization": "fresh_adam", "optimizer_schema": OPTIMIZER_SCHEMA,
            "optimizer_schedule": SCHEDULE_SCHEMA, "attention_lr_multiplier": RECIPE["attention_lr_multiplier"],
            "reference_result": str(parent_quality_path), "parent_model_state_sha256": fingerprint,
            "parent_training_updates": model.provenance["training_updates"], "initialized_model_state_sha256": fingerprint,
            "fixed_buffers_sha256": state_sha256(dict(model.named_buffers())), "quality_endpoints": [RECIPE["steps"]],
            "inference_architecture_changed": False, "continuation_uses_trained_nonzero_attention_head": True,
            "new_weights_native_cost_measured": False, "pr13_model_will_not_be_changed_by_training": True}
    counted = require_space(plan, 400_000_000)
    verify_inputs(plan)
    require(not torch.cuda.is_initialized() and state_sha256(model.state_dict()) == fingerprint,
            "Preparation changed the saved parent or initialized CUDA")
    out.mkdir()
    write(out / "preparation.json", {"status": "pass", "parent_full_sdr_db": review["best_full_sdr_db"],
          "parent_audit": audit, "parent_training_updates": model.provenance["training_updates"],
          "all_30_tensors_trainable_in_production": True, "optimizer_group_sizes": [26, 4],
          "optimizer_initialization": "fresh_adam", "grouped_optimizer_memory_proof_passed": True,
          "target_full_sdr_db": 5.0, "graph_plus_host_samples": 256, "recipe": RECIPE,
          "counted_bytes_before": counted, "reserved_training_bytes": 400_000_000,
          "forecast_including_outside_and_run": counted + 800_000_000 + 400_000_000,
          "cuda_initialized": False, "pr13_model_preserved": True,
          "rationale": "Test faster optimization of the small learned attention correction while reducing the backbone rate. Keep the architecture, reconstruction objective, corpus and validation unchanged; select only after saved-checkpoint audit and full14 regression review.",
          "limitation": "The training-only ablations do not prove this schedule improves quality. This changes optimizer groups, schedule and sample range together, with fresh Adam; it is not an exact optimizer continuation or a matched causal ablation."})
    plan["source_bindings"][str(out / "preparation.json")] = sha(out / "preparation.json")
    write(out / "plan.json", plan)
    print(json.dumps({"event": "prepared", "plan": str(out / "plan.json"), "sha256": sha(out / "plan.json")}), flush=True)


if __name__ == "__main__":
    main()
