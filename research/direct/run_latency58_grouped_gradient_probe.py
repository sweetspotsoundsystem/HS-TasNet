"""Compare qualified group gradients on two fixed, addressed training batches.

Preparation freezes decoded input hashes before any model gradient is measured.
Execution reconstructs those inputs and measures all four retained checkpoints.
No optimizer, validation decoding, audio export or checkpoint write is involved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import resource
import time

import torch

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import PRODUCTION, state_sha256, verify_inputs, disk_bytes
from research.direct.run_latency58_deployed_vocal_views import budget_snapshot
from research.direct.latency58_branch_memory_checkpoint import load_model
from research.direct.latency58_grouped_gradient_probe import collect, summarize, policy
from research.direct.check_latency58_branch_long_context_data import production, TracedCrops
from research.direct.latency58_recorded301_data import select_tracks
from research.direct.latency58_long_context_data import (
    dataset, recipe as pitch_recipe, audio_sha, CROP_SAMPLES, WARMUP_SAMPLES,
    SCORED_SAMPLES, FFMPEG, policy as data_policy)
from research.direct.latency58_remix_augmentation import augment, recipe as remix_recipe
from research.direct.latency58_grouped_vocal_auxiliary import source_views, prepare_groups


SCHEMA = "latency58-fixed-recorded-training-gradient-probe-v1"
FIRST_INDICES = (4116000, 4116016)
ROLES = ("retained_best", "starting_parent", "raw", "ema")
CPU = PHASE / "grouped-gradient-probe-cpu-001"
SOURCE = PHASE / "branch-grouped-vocal-013"
REVIEW = PHASE / "grouped-continuation-review-013"
QUALIFICATION_SHA = "427af51c5ca13c4c5e8cdef6ed9370bc382470e35086a8a150a690704e922238"
DECISION_SHA = "2a01953bbdb0e012dcbb49202991e17a3507eff9a5e4c7d607190a47846dfcc4"
ALLOWANCE = 5_000_000
MAX_RSS = 18_000_000_000


def progress(event, **fields):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    require(rss < MAX_RSS, "Recorded CPU probe exceeded its memory bound")
    print(json.dumps({"event": event, "utc": datetime.now(timezone.utc).isoformat(),
                      "peak_rss_bytes": rss, **fields}, allow_nan=False), flush=True)


def prerequisites():
    require(sha(CPU / "execution.json") == QUALIFICATION_SHA
            and sha(REVIEW / "continuation-review-decision.json") == DECISION_SHA,
            "Require the completed CPU qualification and continuation decision")
    qualified = read(CPU / "plan.json")
    result, execution = read(CPU / "result.json"), read(CPU / "execution.json")
    require(result["status"] == "pass" and execution["actual_exit_code"] == 0
            and not execution["timed_out"] and execution["source_bindings_unchanged"]
            and result["plan_sha256"] == execution["plan_sha256"] == sha(CPU / "plan.json")
            and execution["result_sha256"] == sha(CPU / "result.json")
            and execution["all_40_combined_gradients_bit_exact"]
            and execution["independent_auxiliary_scalar_references_pass"]
            and result["parent_and_rng_unchanged"] and result["gradient_buffers_clear"]
            and result["warmup_input_gradients_zero"] and not result["gpu_used"]
            and result["policy"] == policy(), "CPU qualification is incomplete")
    verify_inputs(qualified)
    source = read(SOURCE / "plan.json")
    review, review_execution = read(REVIEW / "result.json"), read(REVIEW / "execution.json")
    decision = read(REVIEW / "continuation-review-decision.json")
    training = read(SOURCE / "production-run/result.json")
    require(review["status"] == "pass" and review_execution["actual_exit_code"] == 0
            and review_execution["result_sha256"] == sha(REVIEW / "result.json")
            and not decision["unchanged_recipe_continuation_selected"]
            and training["status"] == "pass" and training["updates"] == source["config"]["steps"] == 1000
            and source["config"]["data_start"] + 16 * training["updates"] == FIRST_INDICES[0]
            and source["config"]["augmentation"] == data_policy()
            and source["warmup_samples"] == WARMUP_SAMPLES
            and source["scored_samples"] == SCORED_SAMPLES,
            "Require completed training and its unchanged full-context data recipe")
    models = {**review["reference_models"], **review["candidate_models"]}
    require(set(models) == set(ROLES), "Require exactly the four reviewed checkpoints")
    bindings = dict(qualified["source_bindings"])
    paths = [Path(__file__).resolve(), SOURCE / "plan.json", SOURCE / "result.json",
             SOURCE / "production-run/result.json", SOURCE / "production-root-execution.json",
             REVIEW / "plan.json", REVIEW / "result.json", REVIEW / "execution.json",
             REVIEW / "continuation-review-decision.json", CPU / "plan.json", CPU / "result.json",
             CPU / "execution.json", PRODUCTION / "full_config.json",
             PRODUCTION / "manifests/combined.manifest.json", ROOT / "research/manifests/valid.json",
             Path(FFMPEG), Path("/lib/x86_64-linux-gnu/librubberband.so.2").resolve(),
             Path("/lib/x86_64-linux-gnu/libavfilter.so.9").resolve()]
    paths.extend(sorted((PRODUCTION / "manifests").glob("*.json")))
    for path in paths:
        digest = sha(path)
        require(str(path) not in bindings or bindings[str(path)] == digest, "Prerequisite binding drift")
        if str(path) in source["source_bindings"]:
            require(source["source_bindings"][str(path)] == digest, "Training dependency drift")
        bindings[str(path)] = digest
    for item in models.values():
        binding = item["checkpoint"]
        bindings[binding["path"]] = binding["sha256"]
    verify_inputs({"source_bindings": bindings})
    return source, models, bindings


def make_dataset(source):
    config = source["config"]
    production_config = read(PRODUCTION / "full_config.json")
    _, tracks, manifest_sha, _ = production.load_corpus_manifest(
        PRODUCTION / "manifests/combined.manifest.json",
        expected_file_sha256=source["manifest_sha256"], config=production_config)
    require(manifest_sha == source["manifest_sha256"]
            and production_config["seed"] == config["data_seed"]
            and production_config["sampling"]["root_weights"] == config["source_corpus_root_weights"]
            and production_config["sampling"]["vocal_active_probability"] == config["vocal_active_probability"],
            "Production sampling or manifest changed")
    tracks = select_tracks(tracks, source["training_selection"])
    return dataset(production, tracks, config, FIRST_INDICES[-1] + 16, dataset_class=TracedCrops)


def fixed_batch(crops, source, first):
    require(first in FIRST_INDICES, "Unplanned training batch address")
    rng = torch.get_rng_state().clone()
    mixtures, references, choices = [], [], []
    for index in range(first, first + 16):
        mixture, truth = crops[index]
        choice = pitch_recipe(seed=source["config"]["seed"], sample_index=index)
        if not choice["selected"]:
            original = crops.original[index]
            require(torch.equal(mixture, original[0]) and torch.equal(truth, original[1]),
                    "Unselected crop changed recorded mixture or targets")
        elif choice["semitones"] != 0 or choice["tempo"] != 1.:
            require(torch.equal(mixture, truth.sum(0)), "Pitch/tempo source closure changed")
        mixtures.append(mixture); references.append(truth)
        choices.append({"index": index, **choice})
    mixture, truth = torch.stack(mixtures), torch.stack(references)
    del mixtures, references
    require(mixture.shape == (16, 2, CROP_SAMPLES) and truth.shape == (16, 4, 2, CROP_SAMPLES)
            and all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in (mixture, truth)),
            "Malformed fixed training batch")
    before = audio_sha(mixture, truth)
    seed = source["config"]["seed"]
    audio, targets, changed, factors = augment(mixture, truth, seed=seed, first_sample_index=first)
    mapping = remix_recipe(seed=seed, first_sample_index=first)
    require(torch.equal(audio[:4], mixture[:4]) and torch.equal(targets[:4], truth[:4])
            and torch.equal(audio[4:], targets[4:].sum(1))
            and torch.equal(changed, mapping["changed"]) and torch.equal(factors, mapping["factors"])
            and changed.sum().item() == 12 and mapping["cross_crop"].sum().item() == 8,
            "Remix composition changed")
    for row in range(16):
        for stem in range(4):
            expected = truth[mapping["source_indices"][row, stem], stem]
            if mapping["swap_channels"][row, stem]:
                expected = expected.flip(0)
            require(torch.equal(targets[row, stem], expected * mapping["factors"][row, stem]),
                    "Full-context remix source/channel/gain mapping changed")
        if row >= 8:
            indices = mapping["source_indices"][row].tolist()
            require(len(set(indices)) == 4 and row not in indices, "Cross-crop source addresses changed")
    require(audio_sha(mixture, truth) == before, "Remix mutated original inputs")
    after = audio_sha(audio, targets)
    auxiliary_mix, auxiliary_truth = source_views(audio, targets)
    expected = torch.zeros_like(auxiliary_truth)
    expected[0, [0, 1, 3]] = targets[14, [0, 1, 3]]
    expected[1, 2] = targets[15, 2]
    require(torch.equal(auxiliary_truth, expected) and torch.equal(auxiliary_mix, expected.sum(1))
            and audio_sha(audio, targets) == after, "Full-context source removal changed")
    reductions = prepare_groups(targets[..., WARMUP_SAMPLES:], auxiliary_truth[..., WARMUP_SAMPLES:])
    counts = {name: {"examples": getattr(reductions, name).examples,
                     "samples": getattr(reductions, name).samples,
                     "active_windows": getattr(reductions, name).active.tolist(),
                     "absent_windows": getattr(reductions, name).absent.tolist()}
              for name in ("ordinary", "auxiliary")}
    require(torch.equal(rng, torch.get_rng_state()), "Addressed training reconstruction changed RNG")
    row = {"first_sample_index": first, "stop_sample_index": first + 16,
           "before_remix_sha256": before, "after_remix_sha256": after,
           "auxiliary_sha256": audio_sha(auxiliary_mix, auxiliary_truth),
           "pitch_tempo_recipes": choices, "remix_mapping": {k: v.tolist() for k, v in mapping.items()},
           "groups": counts, "source_mapping_and_full_context_removal_bit_exact": True,
           "rng_unchanged": True}
    progress("fixed_training_batch_decoded", first_sample_index=first,
             input_sha256=after, selected_pitch_tempo=sum(c["selected"] for c in choices))
    return audio, targets, row


def audio_bindings(crops, source):
    files = {**crops.original.read_files, **crops.expanded.read_files}
    require(files and all(source["source_bindings"].get(p) == digest for p, digest in files.items()),
            "Decoded training files are not bound by the completed training plan")
    verify_inputs({"source_bindings": files})
    return files


def prepare(out):
    require(not out.exists(), "Preserve existing prepared probes")
    source, models, bindings = prerequisites()
    budget = budget_snapshot(source["storage_budget"])
    require(budget["headroom_bytes"] > ALLOWANCE, "Reserve scalar-only probe artifacts")
    crops = make_dataset(source)
    rows = []
    for first in FIRST_INDICES:
        audio, truth, row = fixed_batch(crops, source, first)
        rows.append(row)
        del audio, truth
    files = audio_bindings(crops, source)
    bindings.update(files)
    plan = {"schema": SCHEMA, "source_bindings": bindings, "policy": policy(),
            "models": models, "role_order": list(ROLES), "batches": rows,
            "source_training_plan": {"path": str(SOURCE / "plan.json"), "sha256": sha(SOURCE / "plan.json")},
            "batch_selection": "Two consecutive B16 addresses immediately after the final 013 cursor; fixed before measurement",
            "training_audio_bindings": files, "training_selection": source["training_selection"],
            "data_config": source["config"], "warmup_samples": WARMUP_SAMPLES,
            "scored_samples": SCORED_SAMPLES, "loader_workers": 0, "precision": "fp32", "device": "cpu",
            "preparation_utc": datetime.now(timezone.utc).isoformat(), "storage_budget": source["storage_budget"],
            "budget_before": budget, "output_allowance_bytes": ALLOWANCE,
            "maximum_observed_rss_bytes": MAX_RSS, "timeout_seconds": 5400,
            "optimizer_updates": 0, "checkpoint_files_written": False,
            "validation_audio_decoded": False, "quality_measured": False}
    verify_inputs(plan)
    require(not torch.cuda.is_initialized(), "Preparation initialized CUDA")
    out.mkdir()
    write(out / "plan.json", plan)
    progress("recorded_gradient_plan_prepared", plan_sha256=sha(out / "plan.json"),
             training_files=len(files), batches=len(rows), models=len(models))


def run(plan_path, digest):
    require(sha(plan_path) == digest, "Prepared probe plan changed")
    plan = read(plan_path)
    out = plan_path.parent
    require(plan["schema"] == SCHEMA and plan["policy"] == policy()
            and plan["role_order"] == list(ROLES)
            and [b["first_sample_index"] for b in plan["batches"]] == list(FIRST_INDICES)
            and not (out / "run-start.json").exists() and not (out / "result.json").exists(),
            "Require the fixed, unexecuted prepared plan")
    verify_inputs(plan)
    source, models, _ = prerequisites()
    require(models == plan["models"] and source["config"] == plan["data_config"], "Probe inputs differ")
    budget = budget_snapshot(plan["storage_budget"])
    require(budget["headroom_bytes"] > ALLOWANCE, "Insufficient scalar artifact allowance")
    write(out / "run-start.json", {"plan_sha256": digest, "started_utc": datetime.now(timezone.utc).isoformat(),
                                  "pid": os.getpid(), "gpu_used": False})
    began = time.monotonic()
    rng = torch.get_rng_state().clone()
    crops = make_dataset(source)
    reports = []
    for expected in plan["batches"]:
        first = expected["first_sample_index"]
        audio, truth, row = fixed_batch(crops, source, first)
        require(row == expected, "Decoded batch differs from the plan frozen before gradients")
        for role in ROLES:
            item = models[role]
            progress("checkpoint_probe_started", role=role, first_sample_index=first)
            started = time.monotonic()
            model, payload = load_model(item["checkpoint"])
            del payload
            require(state_sha256(model.state_dict()) == item["model_state_sha256"], "Checkpoint state differs")
            model.train().requires_grad_(True)
            model.training_precision = "fp32"

            def observed(phase, group, offset):
                progress("recorded_gradient_progress", role=role, first_sample_index=first,
                         phase=phase, group=group, offset=offset, elapsed_seconds=time.monotonic() - started)

            gradients, metadata = collect(model, audio, truth, warmup_samples=WARMUP_SAMPLES, progress=observed)
            statistics = summarize(gradients)
            require(metadata["input_sha256"] == row["after_remix_sha256"]
                    and all(p.grad is None for p in model.parameters())
                    and statistics["parameter_tensors"] == 40, "Probe state or parameter inventory differs")
            for name, counts in row["groups"].items():
                require(all(metadata["groups"][name][k] == counts[k]
                            for k in ("examples", "active_windows", "absent_windows")),
                        "Recorded probe activity reduction changed")
            report = {"schema": SCHEMA, "status": "pass", "role": role, "model": item,
                      "first_sample_index": first, "plan_sha256": digest,
                      "metadata": metadata, "statistics": statistics,
                      "elapsed_seconds": time.monotonic() - started,
                      "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
            path = out / (str(first) + "-" + role + ".json")
            write(path, report)
            reports.append({"path": str(path), "sha256": sha(path), "role": role, "first_sample_index": first})
            progress("checkpoint_probe_complete", role=role, first_sample_index=first,
                     elapsed_seconds=report["elapsed_seconds"], pairs=statistics["pairs"])
            del gradients, model, statistics, report, metadata
            gc.collect()
            require(disk_bytes(out) < ALLOWANCE and not torch.cuda.is_initialized(), "Probe resource contract changed")
        require(audio_sha(audio, truth) == row["after_remix_sha256"], "Checkpoint probes changed common inputs")
        del audio, truth
    require(audio_bindings(crops, source) == plan["training_audio_bindings"], "Decoded file inventory changed")
    verify_inputs(plan)
    require(sha(plan_path) == digest and torch.equal(rng, torch.get_rng_state()) and len(reports) == 8,
            "Probe completion inventory, plan or RNG changed")
    result = {"schema": SCHEMA, "status": "pass", "plan_sha256": digest, "reports": reports,
              "source_bindings_unchanged": True, "fixed_input_replay_bit_exact": True,
              "checkpoint_weights_inputs_and_rng_unchanged": True, "completed_probes": len(reports),
              "optimizer_updates": 0, "checkpoint_files_written": False, "gpu_used": False,
              "training_batches": len(FIRST_INDICES), "training_crops": 16 * len(FIRST_INDICES),
              "validation_audio_decoded": False, "quality_measured": False,
              "elapsed_seconds": time.monotonic() - began,
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "completed_utc": datetime.now(timezone.utc).isoformat(),
              "budget_after": budget_snapshot(plan["storage_budget"]),
              "limitations": ["Two fixed training batches do not establish validation SDR or the cause of its changes.",
                              "FP32 CPU gradients precede clipping and Adam preconditioning; they do not replay CUDA BF16 training.",
                              "Source-view vectors are contributions to the joint two-view loss with unchanged denominators and 0.1 weight."]}
    write(out / "result.json", result)
    progress("recorded_gradient_probe_complete", result_sha256=sha(out / "result.json"),
             elapsed_seconds=result["elapsed_seconds"], completed_probes=len(reports))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output-prefix", default="grouped-gradient-probe-training-001")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    args = parser.parse_args()
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require CUDA-hidden CPU1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.prepare:
        require(args.plan is None and args.plan_sha256 is None and args.output_prefix
                and all(c.isalnum() or c in "-_" for c in args.output_prefix), "Invalid preparation arguments")
        prepare(PHASE / args.output_prefix)
    else:
        require(args.plan is not None and args.plan_sha256 is not None, "Execution requires the prepared plan hash")
        path = args.plan.resolve(strict=True)
        require(path.is_relative_to(PHASE) and path.name == "plan.json", "Use a prepared phase plan")
        run(path, args.plan_sha256)


if __name__ == "__main__":
    main()
