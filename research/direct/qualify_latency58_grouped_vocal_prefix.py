"""Qualify a prospective continuation's next four B16 input batches on CPU.

The source plan supplies the data policy and planned stream endpoint only.
This does not select model weights, require that endpoint to exist yet, create
a training plan, or start a GPU job. Both ordinary loading and added views are
checked using the previously qualified independent data-checking functions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(key) == "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require CUDA-hidden CPU1")
    import torch
    from research.direct import check_latency58_branch_long_context_data as original
    from research.direct.check_latency58_grouped_vocal_data import replay
    from research.direct.latency58_long_context_data import policy as data_policy, CROP_SAMPLES, EXPANDED_SAMPLES
    from research.direct.latency58_grouped_vocal_auxiliary import policy as loss_policy
    from research.direct.run_latency58_deployed_vocal_views import budget_snapshot
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    source_path = args.source_plan.resolve(strict=True)
    source = read(source_path)
    config = source["config"]
    require(source["schema"] in ("latency58-branch-memory-training-plan-v1", "latency58-grouped-vocal-training-plan-v1")
            and config["augmentation"] == data_policy() and config["batch_size"] == 16
            and config["crop_samples"] == CROP_SAMPLES and config["workers"] == 2
            and source["warmup_samples"] == 88064 and source["scored_samples"] == 88320
            and torch.__version__ == source["torch_version"]
            and type(config["data_start"]) is type(config["steps"]) is type(config["seed"]) is int
            and config["data_start"] >= 0 and config["steps"] > 0,
            "Require the existing addressed long-context data policy")
    first = config["data_start"] + config["steps"] * config["batch_size"]
    stop = first + 64
    out = args.output_directory.resolve()
    require(out.is_relative_to(PHASE) and not out.exists(), "Preserve earlier data qualification")
    previous = PHASE / "grouped-vocal-trainer-integration-cpu-001"
    previous_plan, previous_result, previous_execution = (read(previous / name) for name in ("plan.json", "result.json", "execution.json"))
    require(previous_result["status"] == "pass" and previous_result["source_bindings_unchanged"]
            and previous_result["plan_sha256"] == sha(previous / "plan.json")
            and previous_execution["actual_exit_code"] == previous_execution["actual_enclosing_exit_code"] == 0
            and previous_execution["source_bindings_unchanged"] and not previous_execution["timed_out"],
            "Complete grouped trainer CPU integration before preparing its data")
    bindings = {**source["source_bindings"], **previous_plan["source_bindings"]}
    paths = [Path(__file__).resolve(), source_path, ROOT / "research/direct/check_latency58_grouped_vocal_data.py"]
    paths.extend(previous / name for name in ("plan.json", "result.json", "execution.json"))
    bindings.update({str(path): sha(path) for path in paths})
    verify_inputs({"source_bindings": bindings})
    budget = read(PHASE / "branch-gru-int8-post-ci-storage-001.json")
    before = budget_snapshot(budget)
    out.mkdir()
    data_config = {key: config[key] for key in ("batch_size", "workers", "crop_samples", "seed", "data_seed",
        "root_weights", "source_corpus_root_weights", "vocal_active_probability", "augmentation")}
    data_config.update(data_start=first, qualification_stop_sample_index=stop)
    write(out / "data-config.json", data_config)
    plan = {"schema": "latency58-prospective-grouped-prefix-cpu-v1", "source_bindings": bindings,
        "source_training_plan": {"path": str(source_path), "sha256": sha(source_path)},
        "source_range": {"first": config["data_start"], "planned_stop": first},
        "data_config_sha256": sha(out / "data-config.json"), "first_sample_index": first, "stop_sample_index": stop,
        "augmentation_seed": config["seed"], "training_crop_count": 64, "policy": loss_policy(),
        "warmup_samples": 88064, "scored_samples": 88320, "budget_before": before,
        "source_training_completion_required_before_any_follow_on_gpu": True,
        "model_parent_selected": False, "production_recipe_selected": False, "gpu_used": False}
    write(out / "plan.json", plan)
    local_artifacts = {str(out / name): sha(out / name) for name in ("plan.json", "data-config.json")}
    began = time.monotonic()
    selection = original.selection_contract()
    require(selection == source["training_selection"], "Training/validation selection differs")
    production_config = read(original.PRODUCTION / "full_config.json")
    require(production_config["seed"] == config["data_seed"]
            and production_config["sampling"]["root_weights"] == config["source_corpus_root_weights"]
            and production_config["sampling"]["vocal_active_probability"] == config["vocal_active_probability"]
            and config["root_weights"] == original.ROOT_WEIGHTS, "Addressed corpus sampling differs")
    _, tracks, _, _ = original.production.load_corpus_manifest(original.PRODUCTION / "manifests/combined.manifest.json",
        expected_file_sha256=selection["source_manifest_sha256"], config=production_config)
    tracks = original.select_tracks(tracks, selection)
    kwargs = dict(root_weights=config["root_weights"], seed=config["data_seed"],
                  vocal_active_probability=config["vocal_active_probability"], final_sample_index=stop)
    ordinary = original.TracedCrops(tracks, crop_samples=CROP_SAMPLES, **kwargs)
    expanded = original.TracedCrops(tracks, crop_samples=EXPANDED_SAMPLES, **kwargs)
    dataset = original.PitchTempoCropDataset(ordinary, expanded, seed=config["seed"])
    hashes, choices = {}, []
    for index in range(first, stop):
        hashes[index] = original.audio_sha(*ordinary[index])
        choice = original.recipe(seed=config["seed"], sample_index=index)
        choices.append({"index": index, **choice, "original_sha256": hashes[index],
                        "expanded_sha256": original.audio_sha(*expanded[index]) if choice["selected"] else None})
    files = {**ordinary.read_files, **expanded.read_files}
    require(all(sha(path) == digest for path, digest in files.items()), "Selected training audio differs from its manifest")
    snapshot = {"schema": "latency58-prospective-grouped-prefix-inputs-v1", "selection": selection,
        "source_bindings": {**bindings, **files}, "first_sample_index": first, "stop_sample_index": stop,
        "augmentation_seed": config["seed"], "input_rows": choices, "decoded_training_files": files}
    write(out / "inputs.json", snapshot)
    local_artifacts[str(out / "inputs.json")] = sha(out / "inputs.json")
    print(json.dumps({"event": "prospective_inputs_authenticated", "first_index": first,
                      "stop_index": stop, "training_files": len(files), "pitch_tempo_selected": sum(c["selected"] for c in choices)}), flush=True)
    ordinary_runs = [original.load_batches(dataset, workers=workers, first=first, stop=stop,
        seed=config["seed"], original_hashes=hashes) for workers in (0, 2)]
    require(ordinary_runs[0]["batches"] == ordinary_runs[1]["batches"], "Ordinary worker-count replay differs")
    write(out / "ordinary-reference.json", {"status": "pass", "runs": ordinary_runs})
    local_artifacts[str(out / "ordinary-reference.json")] = sha(out / "ordinary-reference.json")
    print(json.dumps({"event": "prospective_ordinary_reference_pass", "first_index": first}), flush=True)
    grouped_runs = [replay(dataset, workers=workers, snapshot=snapshot, expected=ordinary_runs[0]["batches"])
                    for workers in (0, 2)]
    require(grouped_runs[0] == grouped_runs[1], "Worker count changed source-view samples or activity")
    require({**ordinary.read_files, **expanded.read_files} == files, "Replay read unauthenticated training files")
    verify_inputs(snapshot)
    verify_inputs({"source_bindings": local_artifacts})
    require(sha(source_path) == plan["source_training_plan"]["sha256"] and not torch.cuda.is_initialized(),
            "Source training plan changed or CUDA initialized")
    result = {"status": "pass", "plan_sha256": sha(out / "plan.json"), "inputs_sha256": sha(out / "inputs.json"),
        "data_config_sha256": sha(out / "data-config.json"), "source_bindings_unchanged": True,
        "first_sample_index": first, "stop_sample_index": stop, "recorded_training_crops": 64,
        "selected_pitch_tempo_crops": sum(c["selected"] for c in choices), "decoded_training_files": len(files),
        "independent_ordinary_references_match": True, "zero_and_two_worker_replay_exact": True,
        "ordinary_samples_and_rng_unchanged_by_source_views": True,
        "source_views_exact_through_warmup_and_scored_suffix": True, "batches": grouped_runs[0],
        "model_parent_selected": False, "production_recipe_selected": False, "training_updates": 0,
        "model_weights_loaded": False, "validation_audio_decoded": False, "gpu_used": False, "quality_measured": False,
        "elapsed_seconds": time.monotonic() - began, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "budget_after": budget_snapshot(budget), "completed_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Prospective next addressed prefix, preserving the source plan's sampling and augmentation policy. Usable only if that data policy is selected after the current run's saved-model review."}
    write(out / "result.json", result)
    print(json.dumps({key: result[key] for key in ("status", "first_sample_index", "stop_sample_index", "elapsed_seconds", "gpu_used")}), flush=True)


if __name__ == "__main__":
    main()
