"""One monitored resource check or bounded soft-capped SDR training stage."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import os
import random
import signal
import sys
import time

from research.direct.train_latency58 import (
    PRODUCTION, ROOT, continuity, load_source, read, require, sha, state_sha256, verify_inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--stage-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256
            and sha(args.stage) == args.stage_sha256, "Frozen plan, stage or cwd differs")
    plan, stage = read(args.plan), read(args.stage)
    verify_inputs(plan)
    verify_inputs(stage)
    config = plan["config"]
    start, stop = stage["start_step"], stage["stop_step"]
    require(plan["schema"] == "latency58-sdr-softcap-training-v1" and stage["schema"] == "latency58-sdr-softcap-stage-v1"
            and stage["plan_sha256"] == args.plan_sha256 and 0 <= start < stop <= config["steps"] == 500
            and stop in ((2,) if plan["resource_only"] else (250, 500)) and config["batch_size"] == 4 and config["workers"] == 2
            and config["crop_samples"] == 176128 and plan["warmup_samples"] == plan["scored_samples"] == 88064
            and plan["carry_state"] is True and type(plan["resource_only"]) is bool and config["precision"] == "bf16"
            and plan["teacher_kind"] == "c91" and plan["teacher_weight"] == 0.5
            and plan["sdr_softcap_weight"] == 0.003,
            "Unsupported bounded training recipe")
    require(all(os.environ.get(k) == v for k, v in plan["environment"].items()), "GPU environment differs")
    previous = read(stage["previous_monitor"]["path"])
    require(previous["status"] == previous["supervisor_health"] == "pass" and previous["child_exit_code"] == 0
            and previous["post_exit_quiet_completed"] and previous["last_event_record_id"] == stage["previous_event_record_id"],
            "Previous monitored stage failed")
    monitor = load_source("latency58_sdr_training_watchdog", plan["watchdog_source"])
    _, events = continuity(stage, monitor)
    helpers = load_source("latency58_sdr_training_helpers", plan["helper_source"])
    from research.direct.latency58_sdr_softcap_checkpoint import (
        load_parent, load_model, read_generation, require_space, save_generation, validate_journal)

    run = Path(plan["run_dir"])
    require(run.parent == ROOT / "research/direct/runs/latency58", "Unexpected training directory")
    require_space(plan, 10_000_000 if plan["resource_only"] else 350_000_000)
    if start == 0:
        require(not run.exists(), "Require a fresh training arm")
        run.mkdir()
        helpers.atomic_json(run / "config.json", plan)
        (run / "metrics.jsonl").touch(exist_ok=False)
    else:
        require(run.is_dir() and read(run / "config.json") == plan, "Resume plan differs")
    lock = (run / "trainer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    helpers.atomic_json(Path(stage["output_directory"]) / "event-continuity.json", events)
    sys.path.insert(0, str(PRODUCTION))
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import train_production as production
    from research import experiment
    from research.direct.latency58_asymmetric import PRECISION_POLICY
    from research.direct.latency58_sdr_teacher import load_teacher
    from research.direct.latency58_sdr_context import render_scored_context, physical_context_teacher
    from research.direct.latency58_teacher import deployed_teacher_l1
    from research.direct.latency58_sdr_softcap import AUXILIARY_WEIGHT, ERROR_RATIO_FLOOR, VERSION, sdr_softcap_error
    require(plan["sdr_softcap_weight"] == AUXILIARY_WEIGHT and plan["sdr_softcap_version"] == VERSION
            and plan["sdr_softcap_error_ratio_floor"] == ERROR_RATIO_FLOOR == 0.01,
            "Auxiliary recipe differs")
    from research.direct.latency_ola512_training import raw4_native_objective

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    stored_optimizer = stored_rng = None
    if start == 0:
        model = load_parent(plan)
        require(state_sha256(model.state_dict()) == plan["parent"]["model_state_sha256"], "Fresh context parent differs")
    else:
        pointer = read(run / "latest.json")
        generation = Path(stage["resume_generation"])
        require(pointer["generation"] == str(generation) and pointer["step"] == start
                and pointer["receipt_sha256"] == sha(generation / "receipt.json")
                and pointer["plan_sha256"] == args.plan_sha256, "Resume pointer differs")
        receipt = read_generation(generation, expected_plan_sha=args.plan_sha256)
        audit, audit_execution = read(stage["resume_audit"]["path"]), read(stage["resume_audit_execution"]["path"])
        require(audit["status"] == "pass" and audit["step"] == start
                and audit["generation_receipt_sha256"] == sha(generation / "receipt.json")
                and audit_execution["actual_exit_code"] == 0 and audit_execution["source_bindings_unchanged"],
                "Resume lacks a successful saved-state audit")
        journal = (run / "metrics.jsonl").read_bytes()
        require(hashlib.sha256(journal).hexdigest() == receipt["metrics_sha256"]
                and len(journal) == receipt["metrics_bytes"]
                and journal == (generation / "metrics.jsonl").read_bytes(), "Journal extends beyond the saved generation")
        validate_journal(journal, start, plan, helpers)
        model, loaded = load_model(generation, plan, expected_plan_sha=args.plan_sha256)
        require(loaded == receipt and receipt["step"] == start, "Loaded resume model differs")
        stored_optimizer = torch.load(generation / "optimizer.pt", map_location="cpu", weights_only=True)
        stored_rng = torch.load(generation / "rng.pt", map_location="cpu", weights_only=False)
    teacher, identity = load_teacher(plan["teacher_kind"], plan["teacher"])
    require(identity["model_state_sha256"] == plan["teacher_model_state_sha256"], "Teacher identity differs")
    require(torch.__version__ == plan["torch_version"] and PRECISION_POLICY == plan["precision_policy"]
            and torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(),
            "Reviewed GPU runtime or precision unavailable")
    torch.cuda.set_per_process_memory_fraction(0.75)
    production.configure_determinism(config["seed"])
    production_config = read(PRODUCTION / "full_config.json")
    _, tracks, manifest_sha, _ = production.load_corpus_manifest(
        PRODUCTION / "manifests/combined.manifest.json", expected_file_sha256=plan["manifest_sha256"], config=production_config)
    require(manifest_sha == plan["manifest_sha256"]
            and production_config["sampling"]["root_weights"] == config["root_weights"]
            and production_config["seed"] == config["data_seed"]
            and production_config["sampling"]["vocal_active_probability"] == config["vocal_active_probability"],
            "Production sampling contract changed")
    model.cuda().train().requires_grad_(True)
    model.training_precision = "bf16"
    teacher.cuda()
    frozen = {name: value.clone() for name, value in model.named_buffers()}
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], foreach=False)
    if start:
        require(receipt["optimizer_parameter_names"] == [name for name, _ in model.named_parameters()]
                and receipt["next_sample_index"] == config["data_start"] + start * config["batch_size"],
                "Resume parameter order or data address differs")
        optimizer.load_state_dict(stored_optimizer)
        helpers.validate_rng(stored_rng, torch)
        random.setstate(stored_rng["python"])
        np.random.set_state(stored_rng["numpy"])
        torch.set_rng_state(stored_rng["torch_cpu"])
        torch.cuda.set_rng_state_all(stored_rng["torch_cuda"])
        del stored_optimizer, stored_rng
    helpers.audit_live(model, optimizer, start, frozen, torch, config)
    first_index = config["data_start"] + start * config["batch_size"]
    final_index = config["data_start"] + stop * config["batch_size"]
    dataset = production.CounterAddressedCropDataset(
        tracks, root_weights=config["root_weights"], seed=config["data_seed"], crop_samples=config["crop_samples"],
        vocal_active_probability=config["vocal_active_probability"], final_sample_index=final_index)
    loader = DataLoader(dataset, batch_size=config["batch_size"], num_workers=config["workers"],
                        sampler=production.AbsoluteIndexSampler(first_index, final_index), pin_memory=True,
                        worker_init_fn=production.worker_init, generator=torch.Generator().manual_seed(config["seed"] + 1),
                        multiprocessing_context="spawn", prefetch_factor=2)
    stopped = False

    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    step, began = start, time.monotonic()
    rehearsal = None if plan["resource_only"] else read(plan["full_resource"]["path"])
    torch.cuda.reset_peak_memory_stats()
    helpers.atomic_json(run / "status.json", {"status": "running", "step": step, "pid": os.getpid(),
                                              "stop_step": stop, "stage_sha256": args.stage_sha256})
    try:
        for mixture, targets in loader:
            require(not stopped, "Stage stop requested; no incomplete generation is published")
            first = config["data_start"] + step * config["batch_size"]
            lr = helpers.learning_rate(step, config)
            optimizer.param_groups[0]["lr"] = lr
            mixture, targets = mixture.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            mixture, targets, flags = experiment._augment_training_distribution(mixture=mixture, targets=targets)
            digest = hashlib.sha256()
            for name, value in (("mixture", mixture), ("targets", targets), ("deranged", flags)):
                digest.update(name.encode())
                digest.update(str((tuple(value.shape), value.dtype)).encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            rng = torch.cuda.get_rng_state().clone()
            teacher_targets = physical_context_teacher(teacher, mixture, kind=plan["teacher_kind"],
                                                       warmup_samples=plan["warmup_samples"])
            teacher_digest = hashlib.sha256(teacher_targets.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            require(torch.equal(rng, torch.cuda.get_rng_state()), "Teacher changed matched augmentation RNG")
            output = render_scored_context(model, mixture, warmup_samples=plan["warmup_samples"],
                                           carry_state=plan["carry_state"])
            require(torch.equal(rng, torch.cuda.get_rng_state()), "Student changed matched augmentation RNG")
            require(torch.equal(output.physical_mixture, mixture[..., plan["warmup_samples"]:])
                    and output.initial_state_detached, "Context training lost alignment or detached-state boundary")
            raw, deployed = output.raw, output.deployed
            terms = raw4_native_objective(raw, targets[..., plan["warmup_samples"]:], flags, projection=True)
            auxiliary = deployed_teacher_l1(deployed, teacher_targets)
            relative = sdr_softcap_error(deployed, targets[..., plan["warmup_samples"]:])
            loss = terms.total + plan["teacher_weight"] * auxiliary + plan["sdr_softcap_weight"] * relative.total
            loss.backward()
            require(all(p.grad is not None for p in model.parameters()), "Missing learned gradient")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True, foreach=False)
            optimizer.step()
            torch.cuda.synchronize()
            step += 1
            row = {"step": step, "lr": lr, "loss": float(loss.detach()), "supervised_loss": float(terms.total.detach()),
                   "waveform_l1": float(terms.waveform_l1.detach()), "projection": float(terms.projection.detach()),
                   "projection_contribution": float(terms.projection_contribution.detach()),
                   "teacher_l1": float(auxiliary.detach()), "teacher_weight": plan["teacher_weight"],
                   "teacher_kind": plan["teacher_kind"], "grad_norm": float(norm),
                   "sdr_softcap_loss": float(relative.total.detach()), "sdr_softcap_weight": plan["sdr_softcap_weight"],
                   "sdr_softcap_per_stem": relative.per_stem.detach().cpu().tolist(),
                   "sdr_softcap_active_windows": relative.active_window_counts.cpu().tolist(),
                   "sdr_softcap_active_examples_per_stem": relative.active_examples_per_stem.cpu().tolist(),
                   "deranged_examples": int(flags.sum()), "augmented_batch_sha256": digest.hexdigest(),
                   "first_sample_index": first, "next_sample_index": first + config["batch_size"],
                   "data_hops": output.data_hops, "flush_hops": output.flush_hops, "stage_start_step": start,
                   "carry_state": plan["carry_state"], "initial_state_detached": output.initial_state_detached,
                   "warmup_samples": plan["warmup_samples"], "scored_samples": plan["scored_samples"],
                   "teacher_targets_sha256": teacher_digest,
                   "elapsed_seconds": time.monotonic() - began,
                   "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30}
            if rehearsal is not None and step <= 2:
                expected = rehearsal["matching_production_updates"][step - 1]
                keys = set(row) - {"elapsed_seconds", "peak_vram_gib", "stage_start_step"}
                require(all(row[k] == expected[k] for k in keys),
                        "Production prefix differs from the actual resource rehearsal")
                if step == 2:
                    require(state_sha256(model.state_dict()) == rehearsal["final_model_state_sha256"],
                            "First two production updates differ from the rehearsed model state")
            with (run / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            print(json.dumps(row, allow_nan=False), flush=True)
            del raw, deployed, output, terms, auxiliary, relative, loss, teacher_targets
        require(not stopped and step == stop, "Stage did not reach its requested endpoint")
        helpers.audit_live(model, optimizer, step, frozen, torch, config)
        if plan["resource_only"]:
            require(start == 0 and step == stop == 2
                    and state_sha256(teacher.state_dict()) == plan["teacher_model_state_sha256"]
                    and all(not p.requires_grad and p.grad is None for p in teacher.parameters()),
                    "Resource fixture or frozen teacher differs")
            journal = (run / "metrics.jsonl").read_bytes()
            rows = validate_journal(journal, step, plan, helpers)
            fingerprint = state_sha256(model.state_dict())
            require(fingerprint != plan["parent"]["model_state_sha256"], "Resource updates did not change learned weights")
            receipt = {"model_state_sha256": fingerprint}
            helpers.atomic_json(run / "resource.json", {
                "schema": "latency58-sdr-softcap-resource-result-v1", "status": "pass",
                "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"],
                "source_bindings_unchanged": True, "initial_model_state_sha256": plan["parent"]["model_state_sha256"],
                "final_model_state_sha256": fingerprint, "teacher_kind": plan["teacher_kind"],
                "teacher_weight": plan["teacher_weight"], "teacher_model_state_sha256": plan["teacher_model_state_sha256"],
                "teacher_unchanged": True, "warmup_samples": plan["warmup_samples"],
                "scored_samples": plan["scored_samples"], "carry_state": True,
                "sdr_softcap_weight": plan["sdr_softcap_weight"], "sdr_softcap_version": VERSION,
                "sdr_softcap_error_ratio_floor": ERROR_RATIO_FLOOR,
                "config": config, "matching_production_updates": rows, "training_updates_executed": step,
                "all_parameter_gradients_present": True, "fixed_buffers_unchanged": True,
                "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
                "checkpoint_written": False, "quality_selected": False})
        else:
            receipt = save_generation(run, model, teacher, optimizer, step, plan, args.plan_sha256, helpers)
        verify_inputs(plan)
        require(sha(args.stage) == args.stage_sha256, "Stage changed during training")
        helpers.atomic_json(run / "status.json", {"status": "resource_complete" if plan["resource_only"] else "complete" if step == config["steps"] else "paused",
                                                  "step": step, "pid": os.getpid(), "stop_step": stop,
                                                  "stage_sha256": args.stage_sha256,
                                                  "model_state_sha256": receipt["model_state_sha256"],
                                                  "elapsed_seconds": time.monotonic() - began})
    except BaseException as error:
        helpers.atomic_json(run / "status.json", {"status": "failed", "step": step, "pid": os.getpid(),
                                                  "stop_step": stop, "error": repr(error),
                                                  "stage_sha256": args.stage_sha256})
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
