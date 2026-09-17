"""Compare fixed warm500 FP32/BF16 arithmetic on recorded training batches."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from research.direct.train_latency58 import (
    PRODUCTION, ROOT, continuity, load_source, read, require, sha, state_sha256, verify_inputs,
)
from research.direct.run_latency58_quality import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Frozen diagnostic plan differs")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-training-precision-diagnostic-v1"
            and plan["batches"] == 8 and plan["training_updates"] == 0,
            "Unexpected diagnostic scope")
    require(all(os.environ.get(k) == v for k, v in plan["environment"].items()), "GPU environment differs")
    parent_plan = read(plan["parent_plan"]["path"])
    config = parent_plan["config"]
    require(config["batch_size"] == 4 and config["workers"] == 2 and config["data_start"] == 922000
            and config["crop_samples"] == 176128 and parent_plan["warmup_samples"] == 88064,
            "Recorded training data extent differs")
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve diagnostic evidence")
    previous = read(plan["previous_monitor"]["path"])
    execution = read(plan["previous_execution"]["path"])
    require(previous["status"] == previous["supervisor_health"] == "pass" and previous["child_exit_code"] == 0
            and previous["post_exit_quiet_completed"] and execution["actual_exit_code"] == 0
            and execution["source_bindings_unchanged"]
            and previous["last_event_record_id"] == plan["previous_event_record_id"], "Previous GPU stage failed")
    monitor = load_source("latency58_precision_watchdog", plan["watchdog_source"])
    _, events = continuity(plan, monitor)
    write(out / "event-continuity.json", events)

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from research import experiment
    from research.direct.latency58_log_relative_checkpoint import load_parent, require_space
    from research.direct.latency58_sdr_context import render_scored_context
    from research.metrics import MetricConfig, SOURCE_ORDER, band_sdr, frame_ranges, mean_or_none, windowed_sdr
    sys.path.insert(0, str(PRODUCTION))
    import train_production as production

    require_space(plan, 2_000_000)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = load_parent(parent_plan).train().requires_grad_(False)
    initial_sha = state_sha256(model.state_dict())
    initial_architecture = copy.deepcopy(model.architecture_metadata)
    initial_provenance = copy.deepcopy(model.provenance)
    require(initial_sha == parent_plan["parent"]["model_state_sha256"]
            and all(not isinstance(m, torch.nn.Dropout) or m.p == 0 for m in model.modules())
            and all(not isinstance(m, (torch.nn.GRU, torch.nn.LSTM, torch.nn.RNN)) or m.dropout == 0
                    for m in model.modules()), "Model has stochastic training layers or changed weights")
    production_config = read(PRODUCTION / "full_config.json")
    _, tracks, manifest_sha, _ = production.load_corpus_manifest(
        PRODUCTION / "manifests/combined.manifest.json", expected_file_sha256=parent_plan["manifest_sha256"],
        config=production_config)
    require(manifest_sha == parent_plan["manifest_sha256"]
            and production_config["sampling"]["root_weights"] == config["root_weights"]
            and production_config["seed"] == config["data_seed"]
            and production_config["sampling"]["vocal_active_probability"] == config["vocal_active_probability"],
            "Production sampling contract differs")
    first, end = config["data_start"], config["data_start"] + plan["batches"] * config["batch_size"]
    dataset = production.CounterAddressedCropDataset(
        tracks, root_weights=config["root_weights"], seed=config["data_seed"], crop_samples=config["crop_samples"],
        vocal_active_probability=config["vocal_active_probability"], final_sample_index=end)
    recorded = [json.loads(line) for line in Path(plan["recorded_journal"]["path"]).read_text().splitlines()]
    require(len(recorded) == 250, "Expected completed recorded 250-update journal")
    metric_config = MetricConfig.from_mapping(read(ROOT / "research/eval_config.json")["metrics"])
    progress_counter = 0
    progress_file = (out / "progress.jsonl").open("x", buffering=1)

    def progress(phase):
        nonlocal progress_counter
        progress_counter += 1
        progress_file.write(json.dumps({"step": progress_counter, "phase": phase, "training_updates": 0}) + "\n")
        print(phase, flush=True)

    progress("cpu_fp32_reference")
    probe = dataset[first][0][None, ..., :5120].contiguous()
    model.training_precision = "fp32"
    with torch.no_grad():
        cpu_reference = render_scored_context(model, probe, warmup_samples=1024, carry_state=True).deployed
    require(torch.__version__ == plan["torch_version"] and torch.cuda.is_available()
            and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(), "GPU runtime differs")
    torch.cuda.set_per_process_memory_fraction(.75)
    production.configure_determinism(config["seed"])
    model.cuda()
    progress("gpu_fp32_parity")
    with torch.no_grad():
        gpu_reference = render_scored_context(model, probe.cuda(), warmup_samples=1024, carry_state=True).deployed.cpu()
    parity_error = float((cpu_reference - gpu_reference).abs().max())
    require(parity_error <= 1e-5, "Short FP32 CPU/GPU parity failed")
    del cpu_reference, gpu_reference, probe
    loader = DataLoader(dataset, batch_size=config["batch_size"], num_workers=config["workers"],
                        sampler=production.AbsoluteIndexSampler(first, end), pin_memory=True,
                        worker_init_fn=production.worker_init, generator=torch.Generator().manual_seed(config["seed"] + 1),
                        multiprocessing_context="spawn", prefetch_factor=2)
    batches, examples = [], []
    began = time.monotonic()
    progress("load_batch_1")
    for index, (mixture, targets) in enumerate(loader):
        progress(f"augment_batch_{index + 1}")
        mixture, targets = mixture.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        mixture, targets, flags = experiment._augment_training_distribution(mixture=mixture, targets=targets)
        digest = hashlib.sha256()
        for name, value in (("mixture", mixture), ("targets", targets), ("deranged", flags)):
            digest.update(name.encode())
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        expected = recorded[index]
        require(expected["step"] == index + 1 and expected["first_sample_index"] == first + index * 4
                and expected["next_sample_index"] == first + (index + 1) * 4
                and expected["augmented_batch_sha256"] == digest.hexdigest()
                and expected["deranged_examples"] == int(flags.sum()), "Actual training augmentation prefix differs")
        rng_cpu, rng_cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        predictions, mode_stats = {}, {}
        for precision in ("fp32", "bf16"):
            progress(f"render_batch_{index + 1}_{precision}")
            model.training_precision = precision
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            with torch.no_grad():
                output = render_scored_context(model, mixture, warmup_samples=88064, carry_state=True)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            closure = float((output.deployed.sum(dim=1) - output.physical_mixture).abs().max())
            require(torch.equal(output.physical_mixture, mixture[..., 88064:]) and output.initial_state_detached
                    and not output.deployed.requires_grad and not output.raw.requires_grad
                    and bool(torch.isfinite(output.raw).all()) and bool(torch.isfinite(output.deployed).all())
                    and closure <= 1e-6 and output.flush_hops == 1 and output.data_hops == 688
                    and torch.equal(rng_cpu, torch.get_rng_state()) and torch.equal(rng_cuda, torch.cuda.get_rng_state())
                    and all(p.grad is None and not p.requires_grad for p in model.parameters()),
                    "Precision render changed alignment, gradients, RNG, closure or finite output")
            mode_stats[precision] = {"render_seconds": elapsed, "closure_max_abs": closure,
                                     "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                     "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            predictions[precision] = output.deployed.cpu().numpy()
            del output
        progress(f"score_batch_{index + 1}")
        references = targets[..., 88064:].cpu().numpy()
        difference = predictions["fp32"].astype(np.float64) - predictions["bf16"]
        for batch in range(4):
            stems = {}
            for stem, name in enumerate(SOURCE_ORDER):
                ref = references[batch, stem]
                diff = difference[batch, stem]
                fp32 = predictions["fp32"][batch, stem].astype(np.float64)
                scores = {}
                for precision in ("fp32", "bf16"):
                    pred = predictions[precision][batch, stem]
                    scores[precision] = {"full": windowed_sdr(ref, pred, metric_config),
                                         "low": band_sdr(ref, pred, metric_config.bands_hz["low_20_250"], metric_config)}
                for key in ("full", "low"):
                    require(scores["fp32"][key]["active_windows"] == scores["bf16"][key]["active_windows"],
                            "Activity eligibility changed with precision")
                stems[name] = {"max_abs_difference": float(np.max(np.abs(diff))),
                               "rms_difference": float(np.sqrt(np.mean(diff ** 2))),
                               "relative_rms_difference": float(np.sqrt(np.sum(diff ** 2) / max(np.sum(fp32 ** 2), 1e-24))),
                               "scores": scores,
                               "fp32_minus_bf16_db": {
                                   key: None if scores["fp32"][key]["db"] is None else
                                   scores["fp32"][key]["db"] - scores["bf16"][key]["db"] for key in ("full", "low")}}
            examples.append({"sample_index": first + index * 4 + batch, "stems": stems})
        batches.append({"batch": index + 1, "first_sample_index": first + index * 4,
                        "augmented_batch_sha256": digest.hexdigest(), "actual_recorded_augmentation_matches": True,
                        "deranged_examples": int(flags.sum()), "modes": mode_stats, "render_rng_unchanged": True})
        del predictions, difference, references, mixture, targets, flags
        progress(f"completed_batch_{index + 1}")
    progress("final_identity_audit")
    require(len(batches) == 8 and len(examples) == 32, "Diagnostic did not finish its planned extent")
    model.cpu()
    require(state_sha256(model.state_dict()) == initial_sha and model.architecture_metadata == initial_architecture
            and model.provenance == initial_provenance and all(p.grad is None for p in model.parameters()),
            "Fixed model state, architecture or provenance changed")
    verify_inputs(plan)
    summaries = {}
    for key in ("full", "low"):
        per_stem = {name: {
            **{precision: mean_or_none(e["stems"][name]["scores"][precision][key]["db"] for e in examples)
               for precision in ("fp32", "bf16")},
            "fp32_minus_bf16_db": mean_or_none(e["stems"][name]["fp32_minus_bf16_db"][key] for e in examples),
            "eligible_examples": sum(e["stems"][name]["scores"]["fp32"][key]["db"] is not None for e in examples)
        } for name in SOURCE_ORDER}
        summaries[key] = {"per_stem": per_stem,
                          "equal_stem_means": {p: mean_or_none(s[p] for s in per_stem.values())
                                               for p in ("fp32", "bf16", "fp32_minus_bf16_db")}}
    result = {"schema": "latency58-training-precision-result-v1", "status": "pass",
              "plan_sha256": args.plan_sha256, "source_bindings_unchanged": True,
              "parent_model_state_sha256": initial_sha, "final_model_state_sha256": initial_sha,
              "model_architecture_and_provenance_unchanged": True, "training_updates": 0,
              "optimizer_instances": 0, "checkpoint_written": False, "audio_written": False,
              "gradients_absent": True, "render_rng_unchanged": True, "cpu_gpu_fp32_max_abs_error": parity_error,
              "examples": 32, "batches": batches, "warmup_samples": 88064, "rendered_scored_samples": 88064,
              "metric_config": metric_config.to_dict(),
              "metric_frame_ranges_in_rendered_suffix": frame_ranges(88064, metric_config.window_samples, metric_config.hop_samples),
              "metric_tail_policy": "Unchanged metric omits the partial final window; low-band FFT uses the whole suffix.",
              "summary": summaries, "per_example": examples, "elapsed_seconds": time.monotonic() - began,
              "qualification_scope": "Training-crop forward arithmetic only; no validation, generalization, adoption or backward-memory claim."}
    write(out / "result.json", result)
    progress("diagnostic_completed")
    progress_file.close()
    print(json.dumps({"status": "pass", "summary": summaries, "cpu_gpu_fp32_max_abs_error": parity_error}), flush=True)


if __name__ == "__main__":
    main()
