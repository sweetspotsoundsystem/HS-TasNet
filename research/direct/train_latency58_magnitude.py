"""Frozen-parent magnitude adaptation using recorded MUSDB training mixtures."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

from research.direct.train_latency58 import ROOT, PRODUCTION, read, require, sha, state_sha256, verify_inputs, load_source, continuity
from research.direct.run_latency58_quality import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--stage-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256 and sha(args.stage) == args.stage_sha256,
            "Direct-SDR plan or stage changed")
    plan, stage = read(args.plan), read(args.stage)
    verify_inputs(plan)
    verify_inputs(stage)
    config = plan["config"]
    require(plan["schema"] == "latency58-direct-sdr-plan-v1" and stage["plan_sha256"] == args.plan_sha256
            and stage["stop_step"] == (2 if stage["resource_only"] else config["steps"])
            and all(os.environ.get(k) == v for k, v in plan["environment"].items()), "Training recipe or environment differs")
    monitor = load_source("direct_sdr_watchdog", plan["watchdog_source"])
    _, events = continuity(stage, monitor)
    run = Path(stage["run_directory"])
    require(run.is_relative_to(ROOT / "research/direct/runs/latency58") and not run.exists(), "Preserve direct-SDR runs")
    from research.direct.latency58_sdr_checkpoint import require_space
    require_space(plan, 180_000_000)
    run.mkdir()
    write(run / "event-continuity.json", events)
    lock = (run / "trainer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import torch
    from torch.utils.data import DataLoader
    from research.direct.latency58_direct_sdr import objective, VERSION
    from research.direct.latency58_magnitude_checkpoint import load_model, audit_live, save_generation
    from research.direct.latency58_magnitude import ADAPTER, VERSION as MODEL_VERSION
    from research.direct.latency58_sdr_context import render_scored_context
    from research.direct.latency58_asymmetric import PRECISION_POLICY
    require(MODEL_VERSION == plan["model_version"] and plan["trainable_parameter_names"] == [ADAPTER]
            and VERSION == plan["objective_version"] and PRECISION_POLICY == plan["precision_policy"], "Objective or precision source differs")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model, _ = load_model(plan["parent_checkpoint"])
    require(state_sha256(model.state_dict()) == plan["parent_model_state_sha256"]
            and not torch.cuda.is_initialized(), "CPU parent initialization changed")
    require(torch.__version__ == plan["torch_version"] and torch.cuda.is_available()
            and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(), "Require the reviewed BF16 GPU")
    torch.cuda.set_per_process_memory_fraction(.75)
    sys.path.insert(0, str(PRODUCTION))
    import train_production as production
    production.configure_determinism(config["seed"])
    from research.direct.latency58_musdb_sdr_data import select_tracks
    production_config = read(PRODUCTION / "full_config.json")
    _, tracks, manifest_sha, _ = production.load_corpus_manifest(
        PRODUCTION / "manifests/combined.manifest.json", expected_file_sha256=plan["manifest_sha256"], config=production_config)
    require(manifest_sha == plan["manifest_sha256"] and production_config["seed"] == config["data_seed"]
            and production_config["sampling"]["root_weights"] == config["source_corpus_root_weights"]
            and production_config["sampling"]["vocal_active_probability"] == config["vocal_active_probability"],
            "Training corpus or sampling distribution changed")
    tracks = select_tracks(tracks, plan["training_selection"])
    require(config["root_weights"] == {"musdb18hq_train": 1.0}
            and config["augmentation"] == "ordinary_original_mixture"
            and not plan["online_teacher_used"] and not plan["teacher_generated_targets_in_current_stage"],
            "Unexpected MUSDB training recipe")
    model.provenance = {**model.provenance, "direct_sdr_current_stage_corpus": "83 frozen MUSDB18-HQ training tracks",
                        "direct_sdr_current_stage_augmentation": "ordinary_original_mixture",
                        "direct_sdr_online_teacher_used": False,
                        "direct_sdr_teacher_generated_targets_in_current_stage": False,
                        "direct_sdr_training_selection": plan["training_selection"]}
    model.cuda().train_adapter_only()
    model.training_precision = "bf16"
    frozen = {name: value.clone() for name, value in model.state_dict().items() if name != ADAPTER}
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=config["lr"], foreach=False)
    audit_live(model, optimizer, 0, frozen)
    final_index = config["data_start"] + stage["stop_step"] * config["batch_size"]
    dataset = production.CounterAddressedCropDataset(tracks, root_weights=config["root_weights"], seed=config["data_seed"],
        crop_samples=config["crop_samples"], vocal_active_probability=config["vocal_active_probability"], final_sample_index=final_index)
    loader = DataLoader(dataset, batch_size=config["batch_size"], num_workers=config["workers"],
        sampler=production.AbsoluteIndexSampler(config["data_start"], final_index), pin_memory=True,
        worker_init_fn=production.worker_init, generator=torch.Generator().manual_seed(config["seed"] + 1),
        multiprocessing_context="spawn", prefetch_factor=2)
    stopped = False
    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    rehearsal = None if stage["resource_only"] else read(stage["resource_result"]["path"])
    began, step, prefix = time.monotonic(), 0, []
    torch.cuda.reset_peak_memory_stats()
    with (run / "metrics.jsonl").open("x", buffering=1) as journal:
        for mixture_cpu, truth_cpu in loader:
            require(not stopped and mixture_cpu.shape[0] == config["batch_size"], "Incomplete or interrupted batch")
            if step < config["warmup"]:
                lr = config["lr"] * (step + 1) / config["warmup"]
            else:
                phase = (step - config["warmup"]) / max(1, config["steps"] - 1 - config["warmup"])
                lr = config["min_lr"] + .5 * (config["lr"] - config["min_lr"]) * (1 + math.cos(math.pi * phase))
            optimizer.param_groups[0]["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            micro_rows, input_hash = [], hashlib.sha256()
            for offset in range(0, config["batch_size"], config["microbatch_size"]):
                require(not stopped, "Interrupted accumulation")
                mixture = mixture_cpu[offset:offset + config["microbatch_size"]].cuda(non_blocking=True)
                targets = truth_cpu[offset:offset + config["microbatch_size"]].cuda(non_blocking=True)
                # Preserve the recorded mixture and aligned recorded stems.
                deranged = torch.zeros(mixture.shape[0], dtype=torch.bool, device=mixture.device)
                for value in (mixture, targets, deranged):
                    input_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
                output = render_scored_context(model, mixture, warmup_samples=plan["warmup_samples"], carry_state=True)
                truth = targets[..., plan["warmup_samples"]:]
                require(torch.equal(output.physical_mixture, mixture[..., plan["warmup_samples"]:])
                        and output.initial_state_detached and output.scored_samples == plan["scored_samples"],
                        "Training output lost physical alignment or detached warmup")
                terms = objective(output.raw, output.deployed, truth, output.physical_mixture)
                (terms.total / plan["accumulation_steps"]).backward()
                require(all(p.grad is not None for p in trainable), "The magnitude projection lacks a gradient")
                micro_rows.append({"loss": float(terms.total.detach()), "negative_sdr_db": float(terms.negative_sdr_db.detach()),
                    "absence_db": float(terms.absence_db.detach()), "anchor": float(terms.relative_l1_anchor.detach()),
                    "per_stem_negative_sdr_db": terms.per_stem_negative_sdr_db.detach().cpu().tolist(),
                    "active_windows": terms.active_window_counts.cpu().tolist(), "absent_windows": terms.absent_window_counts.cpu().tolist()})
                del mixture, targets, deranged, output, truth, terms
            norm = torch.nn.utils.clip_grad_norm_(trainable, 5., error_if_nonfinite=True, foreach=False)
            optimizer.step()
            torch.cuda.synchronize()
            step += 1
            row = {"step": step, "lr": lr, "first_sample_index": config["data_start"] + (step - 1) * config["batch_size"],
                   "next_sample_index": config["data_start"] + step * config["batch_size"],
                   "augmented_inputs_sha256": input_hash.hexdigest(), "grad_norm": float(norm),
                   "loss": sum(r["loss"] for r in micro_rows) / plan["accumulation_steps"],
                   "negative_sdr_db": sum(r["negative_sdr_db"] for r in micro_rows) / plan["accumulation_steps"],
                   "microbatches": micro_rows, "adam_steps_this_update": 1, "gradient_clips_this_update": 1}
            if step <= 2:
                prefix.append(row)
                if rehearsal is not None:
                    require(row == rehearsal["matching_production_updates"][step - 1], "Production prefix differs from resource rehearsal")
                if step == 2:
                    audit_live(model, optimizer, step, frozen)
                    fingerprint = state_sha256(model.state_dict())
                    if rehearsal is not None:
                        require(fingerprint == rehearsal["final_model_state_sha256"], "Rehearsed model update differs")
            row = {**row, "elapsed_seconds": time.monotonic() - began, "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30}
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            print(json.dumps({k: row[k] for k in ("step", "loss", "negative_sdr_db", "grad_norm", "elapsed_seconds", "peak_vram_gib")}), flush=True)
            if step % 25 == 0:
                audit_live(model, optimizer, step, frozen)
                require_space(plan, 140_000_000)
            del mixture_cpu, truth_cpu
    require(not stopped and step == stage["stop_step"], "Training did not reach its requested endpoint")
    audit_live(model, optimizer, step, frozen)
    verify_inputs(plan)
    verify_inputs(stage)
    summary = {"status": "pass", "updates": step, "final_model_state_sha256": state_sha256(model.state_dict()),
               "fixed_buffers_unchanged": True, "inherited_tensors_unchanged": True,
               "trainable_parameter_names": [ADAPTER], "matching_production_updates": prefix,
               "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30, "elapsed_seconds": time.monotonic() - began,
               "source_bindings_unchanged": True, "plan_sha256": args.plan_sha256, "checkpoint_written": not stage["resource_only"]}
    if not stage["resource_only"]:
        summary["checkpoint"] = save_generation(model, optimizer, step, plan, args.plan_sha256, run)
    write(run / "result.json", summary)
    print(json.dumps({"status": "pass", "updates": step, "checkpoint_written": summary["checkpoint_written"]}), flush=True)


if __name__ == "__main__":
    main()
