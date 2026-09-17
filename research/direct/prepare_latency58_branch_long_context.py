"""Prepare two-second scored contexts from the selected saved 4.465157 dB EMA."""
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import PRODUCTION, state_sha256, verify_inputs

RECIPE = {"steps": 4000, "lr": 6e-5, "min_lr": 6e-6, "warmup": 100,
          "data_start": 4_028_000, "seed": 20261029, "optimizer_initialization": "fresh_adam"}


def main():
    import torch
    from research.direct.latency58_branch_memory_checkpoint import load_model
    from research.direct.latency58_branch_ema_checkpoint import audit_saved, policy as ema_policy
    from research.direct.latency58_branch_sdr_blend import VERSION, SDR_WEIGHT
    from research.direct.latency58_long_context_data import (
        policy as data_policy, dataset as make_dataset, WARMUP_SAMPLES, SCORED_SAMPLES, CROP_SAMPLES)
    from research.direct.latency58_logical_batch_loss import policy as reduction_policy
    from research.direct.check_latency58_branch_long_context_parent import check as check_parent
    from research.direct.check_latency58_branch_long_context_data import load_batches, production
    from research.direct.latency58_recorded301_data import select_tracks
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require single-threaded CPU preparation with CUDA hidden")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    latest = PHASE / "branch-pitch-ema-002"
    source_path, review_path = latest / "plan.json", latest / "selection-review.json"
    source, review = read(source_path), read(review_path)
    verify_inputs(source)
    verify_inputs(review)
    require(review["status"] == "selected_for_research" and review["selected_weight_role"] == "ema"
            and review["actual_root_exit_code"] == 0 and not review["saved_native_quality_target_reached"]
            and review["best_full_sdr_db"] == 4.46515742201644
            and review["total_training_updates"] == 39250
            and source["objective_version"] == VERSION and source["direct_sdr_weight"] == SDR_WEIGHT == .2,
            "Require the selected saved EMA and its completed full14 review")
    checkpoint = review["best_research_checkpoint"]
    quality_path = Path(review["best_research_reference_result"])
    quality = read(quality_path)
    verify_inputs(quality)
    require(quality["status"] == "pass" and quality["track_count"] == 14 and quality["excerpt_count"] == 28
            and quality["results"][0]["checkpoint"] == checkpoint
            and quality["results"][0]["aggregate"]["full_sdr_db"] == review["best_full_sdr_db"],
            "Selected EMA quality result changed")
    paths = [Path(__file__).resolve(), source_path, review_path, quality_path, latest / "checkpoint-audit.json"]
    for path in (latest / "root-execution.json", latest / "production-stage/execution.json",
                 latest / "full14-raw/execution.json", latest / "full14-ema/execution.json",
                 PHASE / "branch-pitch-ema-review-stage-002/execution.json",
                 PHASE / "branch-pitch-ema-review-stage-002/root-execution.json"):
        execution = read(path)
        require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                and not execution.get("timed_out", False), "A required parent execution is incomplete")
        paths.append(path)
    paths.extend(Path(checkpoint["path"]).parent.iterdir())
    audit = audit_saved(checkpoint, source, sha(source_path))
    parent, _ = load_model(checkpoint)
    parent_sha = state_sha256(parent.state_dict())
    require(audit["status"] == "pass" and audit["optimizer_owner"] == "raw-model.pt"
            and audit["saved_optimizer_tensor_count"] == 40
            and parent_sha == audit["model_state_sha256"] == "2b123695bb86e92ef26e8e4fdab06069aecc5a55e8e26b171be25473a33208d5"
            and parent.provenance["training_updates"] == 39250
            and parent.provenance["checkpoint_weight_role"] == "averaged_inference",
            "EMA parent identity or raw Adam ownership changed")
    budget = {k: source[k] for k in ("counted_roots", "stop_counted_bytes", "outside_roots_reservation_bytes")}
    require(budget["stop_counted_bytes"] + budget["outside_roots_reservation_bytes"] == 90_000_000_000,
            "Retain the user-authorized artifact cap")
    data = PHASE / "branch-long-context-data-001"
    data_inputs, data_result = read(data / "inputs.json"), read(data / "result.json")
    verify_inputs(data_inputs)
    require(data_result["status"] == "pass" and data_result["source_bindings_unchanged"]
            and data_result["inputs_sha256"] == sha(data / "inputs.json")
            and data_result["worker_count_replay_and_composed_remix_bit_exact"]
            and data_inputs["first_sample_index"] == RECIPE["data_start"]
            and data_inputs["stop_sample_index"] == RECIPE["data_start"] + 64
            and data_inputs["augmentation_seed"] == RECIPE["seed"]
            and data_inputs["selection"] == source["training_selection"], "Long-context data qualification changed")
    loss = PHASE / "branch-long-context-loss-001"
    loss_plan, loss_result = read(loss / "plan.json"), read(loss / "result.json")
    verify_inputs(loss_plan)
    require(loss_result["status"] == "pass" and loss_result["source_bindings_unchanged"]
            and loss_result["plan_sha256"] == sha(loss / "plan.json")
            and loss_result["policy"] == reduction_policy()
            and loss_result["unchanged_full_batch_objective_and_audio_gradients_match"],
            "Whole-batch loss accumulation is not qualified")
    paths.extend(data / n for n in ("inputs.json", "result.json"))
    paths.extend(loss / n for n in ("plan.json", "result.json"))
    for directory in (PHASE / "branch-long-context-data-stage-001", loss):
        for name in ("execution.json", "root-execution.json"):
            path = directory / name
            execution = read(path)
            require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                    and not execution.get("timed_out", False), "CPU qualification did not close successfully")
            paths.append(path)
    files = ("latency58_long_context_data.py", "latency58_logical_batch_loss.py",
             "check_latency58_branch_long_context_data.py", "check_latency58_logical_batch_loss.py",
             "check_latency58_branch_long_context_parent.py", "check_latency58_branch_long_context_gpu.py",
             "train_latency58_branch_long_context.py", "run_latency58_branch_long_context.py",
             "observe_latency58_branch_long_context_prefix.py", "report_latency58_branch_long_context.py")
    paths.extend(ROOT / "research/direct" / name for name in files)
    bindings = {**source["source_bindings"], **review["source_bindings"], **data_inputs["source_bindings"],
                **loss_plan["source_bindings"], **{str(p): sha(p) for p in paths}}
    verify_inputs({"source_bindings": bindings})
    out = PHASE / "branch-long-context-001"
    require(not out.exists(), "Preserve earlier longer-context preparations")
    counted = require_space(budget, 650_000_000)
    out.mkdir()
    began = time.monotonic()
    config = {**source["config"], **{k: RECIPE[k] for k in ("steps", "lr", "min_lr", "warmup", "data_start", "seed")},
              "checkpoint_every": RECIPE["steps"], "crop_samples": CROP_SAMPLES, "microbatch_size": 8,
              "augmentation": data_policy()}
    plan = {**{k: v for k, v in source.items() if k not in ("retry_of", "retry_reason")},
            **budget, "name": out.name, "output_directory": str(out), "config": config,
            "parent_checkpoint": checkpoint, "parent_kind": "saved_trained_branch_memory", "parent_weight_role": "ema",
            "parent_model_state_sha256": parent_sha, "parent_training_updates": 39250,
            "initialized_model_state_sha256": parent_sha, "optimizer_initialization": "fresh_adam",
            "reference_result": str(quality_path), "parent_selection_review": str(review_path),
            "parent_full_sdr_db": review["best_full_sdr_db"], "fixed_buffers_sha256": state_sha256(dict(parent.named_buffers())),
            "parameter_names": [name for name, _ in parent.named_parameters()],
            "inference_architecture": parent.architecture_metadata, "inference_architecture_changed": False,
            "warmup_samples": WARMUP_SAMPLES, "scored_samples": SCORED_SAMPLES, "accumulation_steps": 2,
            "logical_batch_loss": reduction_policy(), "objective_version": VERSION, "direct_sdr_weight": SDR_WEIGHT,
            "training_context_implementation": "two-second scored suffix after detached two-second final-frame warmup; globally normalized B8 accumulation",
            "continuation_kind": "selected_ema_branch_two_second_context_and_paired_ema",
            "ema": ema_policy(.995), "quality_endpoints": [4000], "quality_weight_roles": ["raw", "ema"],
            "previous_execution_for_resource": str(latest / "production-stage/execution.json"),
            "qualified_data_prefix": data_result["augmented_loaders"][1]["batches"], "source_bindings": bindings}
    functional = check_parent(parent, plan, out)
    write(out / "current-parent-functional.json", functional)
    require(functional["status"] == "pass" and functional["raw_adam_rejected_for_ema_weights"]
            and functional["resumed_third_raw_and_ema_update_and_adam_states_bit_exact"],
            "Selected EMA parent context or fresh-optimizer restart failed")
    print(json.dumps({"event": "current_parent_pass", "scored_samples": SCORED_SAMPLES,
                      "raw_and_ema_restart_exact": True}), flush=True)
    corpus = read(PRODUCTION / "full_config.json")
    _, tracks, _, _ = production.load_corpus_manifest(PRODUCTION / "manifests/combined.manifest.json",
        expected_file_sha256=source["manifest_sha256"], config=corpus)
    tracks = select_tracks(tracks, source["training_selection"])
    stop = config["data_start"] + 64
    integrated = load_batches(make_dataset(production, tracks, config, stop), workers=2,
                              first=config["data_start"], stop=stop, seed=config["seed"])
    require(integrated["batches"] == plan["qualified_data_prefix"] and not torch.cuda.is_initialized()
            and state_sha256(parent.state_dict()) == parent_sha and parent.algorithmic_latency_samples == 256,
            "Production dataset composition or preserved parent differs from qualification")
    write(out / "composed-data-functional.json", integrated)
    for name in ("fixture-plan.json", "current-parent-functional.json", "composed-data-functional.json"):
        plan["source_bindings"][str(out / name)] = sha(out / name)
    verify_inputs(plan)
    write(out / "preparation.json", {"status": "pass", "parent_full_sdr_db": review["best_full_sdr_db"],
          "parent_audit": audit, "parent_weight_role": "ema", "recipe": RECIPE,
          "warmup_samples": WARMUP_SAMPLES, "scored_samples": SCORED_SAMPLES, "logical_batch_size": 16,
          "microbatch_size": 8, "logical_batch_loss": reduction_policy(), "augmentation": data_policy(),
          "ema": plan["ema"], "optimizer_initialization": "fresh_adam", "inference_architecture_changed": False,
          "graph_plus_host_samples": 256, "total_public_stream_states": 8,
          "all_40_tensors_trainable_in_production": True, "current_parent_and_restart_qualified": True,
          "integrated_data_matches_qualified_four_batches": True, "source_bindings_unchanged": True,
          "forecast_including_outside_and_run": counted + budget["outside_roots_reservation_bytes"] + 650_000_000,
          "reserved_training_and_diagnostic_bytes": 650_000_000, "artifact_cap_bytes": 90_000_000_000,
          "elapsed_seconds": time.monotonic() - began, "cuda_initialized": False,
          "rationale": "Extend backpropagation through the existing recurrent model from one scored second to two. Preserve the complete B16 remix group, the .2 SDR blend, pitch/tempo policy and paired EMA, using two B8 contributions with whole-batch activity denominators. Start from selected EMA weights with fresh Adam.",
          "limitation": "Longer scored context, starting weights, seed and crop offsets change together. CPU evidence establishes implementation behavior, not an SDR gain, GPU capacity or M4 timing. The two-update GPU rehearsal, both saved full14 evaluations and all baseline regressions remain required."})
    plan["source_bindings"][str(out / "preparation.json")] = sha(out / "preparation.json")
    require_space(plan, 650_000_000)
    write(out / "plan.json", plan)
    print(json.dumps({"status": "pass", "plan": str(out / "plan.json"), "sha256": sha(out / "plan.json"),
                      "parent_full_sdr_db": review["best_full_sdr_db"], "scored_samples": SCORED_SAMPLES}), flush=True)


if __name__ == "__main__":
    main()
