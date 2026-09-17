"""Prepare pitch/tempo training and paired raw/EMA endpoints from the selected 4.453208 dB parent."""
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import PRODUCTION, state_sha256, verify_inputs

RECIPE = {"steps": 4000, "lr": 6e-5, "min_lr": 6e-6, "warmup": 100,
          "data_start": 3_964_000, "seed": 20261028, "optimizer_initialization": "fresh_adam"}


def main():
    import torch
    from research.direct.latency58_branch_memory_checkpoint import load_model, audit_saved
    from research.direct.latency58_branch_sdr_blend import VERSION, SDR_WEIGHT
    from research.direct.latency58_branch_memory import ADAPTERS
    from research.direct.latency58_remix_augmentation import AUGMENTATION
    from research.direct.latency58_pitch_ema_data import policy as data_policy, dataset as make_dataset
    from research.direct.latency58_branch_ema_checkpoint import policy as ema_policy
    from research.direct.check_latency58_branch_sdr_ema import check as check_ema
    from research.direct.check_latency58_pitch_tempo import load_batches, production
    from research.direct.check_latency58_branch_continuation_checkpoint import check as check_checkpoint
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require single-threaded CPU preparation")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    latest = PHASE / "branch-sdr-continuation-001"
    review_path = latest / "selection-review.json"
    review = read(review_path)
    verify_inputs(review)
    require(review["status"] == "selected_for_research" and review["actual_root_exit_code"] == 0
            and not review["saved_native_quality_target_reached"]
            and review["best_full_sdr_db"] == 4.453207822764789
            and review["candidate_saved_full_sdr_db"] == 4.453207822764789,
            "Require the completed continuation selection and retained best endpoint")
    checkpoint = review["best_research_checkpoint"]
    parent_root = Path(checkpoint["path"]).parents[2]
    source_path = parent_root / "plan.json"
    source = read(source_path)
    quality_path = Path(review["best_research_reference_result"])
    quality = read(quality_path)
    verify_inputs(source)
    verify_inputs(quality)
    require(parent_root == PHASE / "branch-sdr-continuation-001"
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
    for path in (latest / "root-execution.json", PHASE / "branch-sdr-continuation-review-stage-001/execution.json",
                 PHASE / "branch-sdr-continuation-review-stage-001/root-execution.json",
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
            and parent_sha == audit["model_state_sha256"] == "cb6cfda0fb2c901011531c4eb01c3dcdd9f99f6f4683ff14f1b95f609d69e619"
            and parent_updates == source["parent_training_updates"] + source["config"]["steps"] == 35250
            and all(torch.count_nonzero(dict(parent.named_parameters())[name]) > 0 for name in ADAPTERS),
            "The selected parent's audited weights, optimizer or lineage changed")
    files = ("latency58_branch_sdr_blend.py", "check_latency58_branch_continuation_checkpoint.py",
             "latency58_pitch_ema_data.py", "latency58_pitch_tempo_augmentation.py",
             "train_latency58_branch_pitch_ema.py", "run_latency58_branch_pitch_ema.py",
             "train_latency58_branch_sdr_ema.py", "check_latency58_branch_sdr_ema.py",
             "latency58_branch_ema.py", "latency58_branch_ema_checkpoint.py",
             "report_latency58_branch_pitch_ema.py", "observe_latency58_branch_pitch_ema_prefix.py")
    paths.extend(ROOT / "research/direct" / name for name in files)
    pitch_root = PHASE / "pitch-tempo-cpu-001"
    pitch_inputs, pitch_result = read(pitch_root / "inputs.json"), read(pitch_root / "result.json")
    verify_inputs(pitch_inputs)
    require(pitch_result["status"] == "pass" and pitch_result["source_bindings_unchanged"]
            and pitch_result["inputs_sha256"] == sha(pitch_root / "inputs.json")
            and pitch_result["worker_count_replay_and_composed_remix_bit_exact"]
            and pitch_inputs["first_sample_index"] == RECIPE["data_start"]
            and pitch_inputs["augmentation_seed"] == RECIPE["seed"], "Pitch/tempo CPU evidence changed")
    paths.extend(pitch_root / name for name in ("inputs.json", "result.json"))
    pitch_execution_path = PHASE / "pitch-tempo-cpu-stage-001/execution-execution.json"
    pitch_root_path = PHASE / "pitch-tempo-cpu-stage-001/root-execution.json"
    execution, root_execution = read(pitch_execution_path), read(pitch_root_path)
    require(execution["actual_exit_code"] == root_execution["actual_exit_code"] == 0
            and execution["source_bindings_unchanged"] and not execution["timed_out"]
            and root_execution["execution"] == {"path": str(pitch_execution_path), "sha256": sha(pitch_execution_path)}
            and root_execution["result"] == {"path": str(pitch_root / "result.json"), "sha256": sha(pitch_root / "result.json")},
            "Pitch/tempo qualification did not close")
    paths.extend((pitch_execution_path, pitch_root_path))
    for name in ("branch-ema-functional-001", "branch-ema-disk-functional-001", "branch-sdr-ema-control-001"):
        result_path = PHASE / name / "result.json"
        result = read(result_path)
        require(result["status"] == "pass", "An EMA component did not qualify")
        verify_inputs(result)
        paths.extend((result_path, PHASE / name / "plan.json"))
        component = read(PHASE / name / "plan.json")
        verify_inputs(component)
        paths.extend(Path(path) for path in component["source_bindings"])
        for kind in ("execution.json", "root-execution.json"):
            execution = read(PHASE / name / kind)
            require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"],
                    "EMA component qualification did not close")
            paths.append(PHASE / name / kind)
    require(all(source["source_bindings"].get(path, digest) == digest
                for path, digest in review["source_bindings"].items()), "Preceding evidence binds conflicting source bytes")
    bindings = {**source["source_bindings"], **review["source_bindings"], **pitch_inputs["source_bindings"], **{str(path): sha(path) for path in paths}}
    verify_inputs({"source_bindings": bindings})
    out = PHASE / "branch-pitch-ema-001"
    require(not out.exists(), "Preserve earlier continuation plans")
    # Includes raw and EMA models, raw optimizer, recipes, journals and paired full14 reports.
    counted = require_space(budget, 650_000_000)
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
    config["augmentation"] = data_policy()
    ema_control = check_ema({**fixture_source, "ema_decay": .995})
    write(out / "current-parent-ema-control.json", ema_control)
    require(ema_control["status"] == "pass" and ema_control["all_40_raw_gradients_and_adam_states_bit_exact"]
            and ema_control["raw_control_audio_and_eight_states_bit_exact"], "Current-parent EMA hook failed")
    corpus = read(PRODUCTION / "full_config.json")
    _, tracks, _, _ = production.load_corpus_manifest(PRODUCTION / "manifests/combined.manifest.json",
        expected_file_sha256=source["manifest_sha256"], config=corpus)
    from research.direct.latency58_recorded301_data import select_tracks
    tracks = select_tracks(tracks, source["training_selection"])
    stop = config["data_start"] + 64
    combined = make_dataset(production, tracks, config, stop)
    data_result = load_batches(combined, workers=config["workers"], first=config["data_start"], stop=stop, seed=config["seed"])
    require(data_result["batches"] == pitch_result["augmented_loaders"][1]["batches"],
            "Integrated production dataset differs from qualified CPU audio")
    require(not torch.cuda.is_initialized() and state_sha256(parent.state_dict()) == parent_sha,
            "CPU preparation initialized CUDA or changed its parent")
    write(out / "composed-data-functional.json", data_result)
    paths.extend(out / name for name in ("current-parent-ema-control.json", "composed-data-functional.json"))
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
            "continuation_kind": "selected_branch_sdr_pitch_tempo_and_paired_ema",
            "ema": ema_policy(.995), "quality_weight_roles": ["raw", "ema"],
            "qualified_data_prefix": data_result["batches"],
            "augmentation_fractions": {"no_further_remix": .25, "same_crop_remix": .25, "cross_crop_remix": .5},
            "channel_swap": "independent per source in the 12 remixed examples; follows pitch/tempo",
            "pitch_tempo_recipes_logged_every_update": True,
            "augmentation_mapping_semantics": "First four unchanged by remix only; pitch/tempo may precede it",
            "artifact_cap_bytes": 90_000_000_000,
            "native_cost_measured": False, "new_weights_native_cost_measured": False}
    verify_inputs(plan)
    write(out / "preparation.json", {"status": "pass", "parent_full_sdr_db": review["best_full_sdr_db"],
          "parent_audit": audit, "recipe": RECIPE, "objective_version": VERSION, "direct_sdr_weight": SDR_WEIGHT,
          "augmentation": data_policy(), "ema": ema_policy(.995),
          "current_parent_ema_control_passed": True, "integrated_data_matches_qualified_four_batches": True, "current_parent_context_and_ram_checkpoint_passed": True,
          "all_40_tensors_trainable_in_production": True, "optimizer_initialization": "fresh_adam",
          "target_full_sdr_db": 5.0, "graph_plus_host_samples": 256, "total_public_stream_states": 8,
          "additional_parameters": 0, "inference_architecture_changed": False,
          "counted_bytes_before": counted, "reserved_training_and_diagnostic_bytes": 650_000_000,
          "forecast_including_outside_and_run": counted + budget["outside_roots_reservation_bytes"] + 650_000_000,
          "artifact_cap_bytes": 90_000_000_000, "cuda_initialized": False,
          "elapsed_seconds": time.monotonic() - began,
          "rationale": "Continue from the selected 4.453208 dB saved endpoint with the same 4000-update schedule and .2 objective, composing qualified 20% pitch/tempo before ordinary remixes. Retain raw and .995 EMA endpoints for identical full14 comparisons. New seed and crop range prevent causal attribution of augmentation alone.",
          "limitation": "CPU preparation measures no SDR gain, GPU throughput or new native timing. Selected longer crops change offsets. Drum transient and vocal formant perceptual fidelity remain unmeasured. Every endpoint requires full parent/C204/fixed-share regression review; EMA weights do not own Adam moments."})
    plan["source_bindings"][str(out / "preparation.json")] = sha(out / "preparation.json")
    require_space(plan, 650_000_000)
    write(out / "plan.json", plan)
    print(json.dumps({"event": "prepared", "plan": str(out / "plan.json"), "sha256": sha(out / "plan.json"),
                      "initialized_model_state_sha256": parent_sha, "parent_full_sdr_db": review["best_full_sdr_db"]}), flush=True)


if __name__ == "__main__":
    main()
