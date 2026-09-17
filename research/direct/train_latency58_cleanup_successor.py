"""A fixed two-update rehearsal or 250-update vocal-cleanup trial from the SDR leader."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import os
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
    from research.direct.latency58_cleanup_successor_checkpoint import validate_recipe
    validate_recipe(plan)
    require(stage["schema"] == "latency58-cleanup-successor-stage-v1"
            and stage["plan_sha256"] == args.plan_sha256 and start == 0
            and stop == (2 if plan["resource_only"] else 250), "Require one fresh bounded stage")
    require(all(os.environ.get(k) == v for k, v in plan["environment"].items()), "GPU environment differs")
    previous = read(stage["previous_monitor"]["path"])
    require(previous["status"] == previous["supervisor_health"] == "pass" and previous["child_exit_code"] == 0
            and previous["post_exit_quiet_completed"] and previous["last_event_record_id"] == stage["previous_event_record_id"],
            "Previous monitored stage failed")
    monitor = load_source("latency58_sdr_training_watchdog", plan["watchdog_source"])
    _, events = continuity(stage, monitor)
    helpers = load_source("latency58_sdr_training_helpers", plan["helper_source"])
    from research.direct.latency58_cleanup_successor_checkpoint import (
        load_parent, require_space, save_generation, validate_journal, audit_live, LOSS_KEYS, IDENTITY_KEYS)

    run = Path(plan["run_dir"])
    require(run.parent == ROOT / "research/direct/runs/latency58", "Unexpected training directory")
    require_space(plan, 10_000_000 if plan["resource_only"] else 350_000_000)
    require(not run.exists(), "Require a fresh pilot; no resume or automatic extension")
    run.mkdir()
    helpers.atomic_json(run / "config.json", plan)
    (run / "metrics.jsonl").touch(exist_ok=False)
    lock = (run / "trainer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    helpers.atomic_json(Path(stage["output_directory"]) / "event-continuity.json", events)
    sys.path.insert(0, str(PRODUCTION))
    import torch
    from torch.utils.data import DataLoader
    import train_production as production
    from research.direct.latency58_vocal_focus_augmentation import augment_vocal_focus
    from research.direct.latency58_asymmetric import PRECISION_POLICY
    from research.direct.latency58_sdr_teacher import load_teacher
    from research.direct.latency58_sdr_context import render_scored_context, physical_context_teacher
    from research.direct.latency58_drum_emphasis import VERSION as OBJECTIVE_VERSION
    from research.direct.latency58_controlled_deployed_loss import controlled_deployed_objective
    from research.direct.latency58_counterfactual_journal import rng_state_sha256
    from research.direct.latency58_controlled_deployed_journal import loss_evidence
    from research.direct.latency58_sdr_accum import backward_mean_loss, VERSION
    require(plan["accumulation_version"] == VERSION, "Accumulation implementation differs")
    require(plan["drum_weight"] == 2 and plan["objective_version"] == OBJECTIVE_VERSION,
            "Normalized drum implementation differs")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = load_parent(plan)
    require(state_sha256(model.state_dict()) == plan["initialized_model_state_sha256"],
            "Fresh vocal-focus initialization differs")
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
    audit_live(model, optimizer, start, frozen, torch, plan, helpers)
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
        for mixtures_cpu, targets_cpu in loader:
            require(not stopped, "Stage stop requested; no incomplete generation is published")
            require(mixtures_cpu.shape[0] == targets_cpu.shape[0] == config["batch_size"],
                    "Incomplete accumulated batch")
            first = config["data_start"] + step * config["batch_size"]
            lr = helpers.learning_rate(step, config)
            optimizer.param_groups[0]["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            microbatches = []
            for micro_index in range(plan["accumulation_steps"]):
                require(not stopped, "Stop requested during accumulation; no partial Adam update")
                offset = micro_index * plan["microbatch_size"]
                mixture = mixtures_cpu[offset:offset + plan["microbatch_size"]].cuda(non_blocking=True)
                targets = targets_cpu[offset:offset + plan["microbatch_size"]].cuda(non_blocking=True)
                def tensor_digest(named):
                    digest = hashlib.sha256()
                    for name, value in named:
                        digest.update(name.encode())
                        digest.update(str((tuple(value.shape), value.dtype)).encode())
                        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
                    return digest.hexdigest()

                pristine_digest = tensor_digest((("mixture", mixture), ("targets", targets)))
                rng_before = hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()
                batch = augment_vocal_focus(mixture, targets, first_sample_index=first + offset,
                                            enabled=plan["focused_augmentation"])
                original_digest = tensor_digest(zip(("mixture", "targets", "deranged"), batch.original_augmentation))
                for index, code in enumerate(batch.view_codes):
                    if code in (0, 1):
                        for stem in range(4):
                            wanted = (stem != 2) if code == 0 else (stem == 2)
                            require(torch.equal(batch.targets[index, stem], targets[index, stem]) if wanted
                                    else not bool(torch.count_nonzero(batch.targets[index, stem])),
                                    "Forced source view differs from pristine targets")
                        require(not bool(batch.vocal_derangement[index])
                                and torch.equal(batch.mixture[index], batch.targets[index].sum(dim=0)),
                                "Forced source mixture or flags differ")
                mixture, targets, flags = batch.mixture, batch.targets, batch.vocal_derangement
                codes = list(batch.view_codes)
                digest = tensor_digest((("mixture", mixture), ("targets", targets), ("deranged", flags)))
                del batch
                rng = torch.cuda.get_rng_state().clone()
                rng_after = hashlib.sha256(rng.numpy().tobytes()).hexdigest()
                teacher_targets = physical_context_teacher(teacher, mixture, kind=plan["teacher_kind"],
                                                           warmup_samples=plan["warmup_samples"])
                teacher_digest = hashlib.sha256(teacher_targets.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                require(torch.equal(rng, torch.cuda.get_rng_state()), "Teacher changed augmentation RNG")
                output = render_scored_context(model, mixture, warmup_samples=plan["warmup_samples"],
                                               carry_state=plan["carry_state"])
                require(torch.equal(rng, torch.cuda.get_rng_state()), "Student changed augmentation RNG")
                require(torch.equal(output.physical_mixture, mixture[..., plan["warmup_samples"]:])
                        and output.initial_state_detached, "Context training lost alignment or detached state")
                controlled_terms = controlled_deployed_objective(output.raw, output.deployed,
                    targets[..., plan["warmup_samples"]:], teacher_targets, flags,
                    view_codes=tuple(codes), weight=plan["additional_loss_weight"])
                terms = controlled_terms.base
                loss = controlled_terms.total
                backward_mean_loss(loss)
                require(torch.equal(rng, torch.cuda.get_rng_state()), "Student backward changed augmentation RNG")
                require(all(p.grad is not None for p in model.parameters()), "Missing microbatch gradient")
                microbatches.append({
                    "micro_index": micro_index, "first_sample_index": first + offset,
                    "next_sample_index": first + offset + plan["microbatch_size"],
                    "batch_size": plan["microbatch_size"], "loss": float(loss.detach()),
                    "supervised_loss": float((terms.waveform_l1 + terms.projection_contribution).detach()),
                    "waveform_l1": float(terms.waveform_l1.detach()),
                    "projection": float(terms.projection.detach()),
                    "projection_contribution": float(terms.projection_contribution.detach()),
                    "teacher_l1": float(terms.teacher_l1.detach()), "deranged_examples": int(flags.sum()),
                    **loss_evidence(output.raw, output.deployed, targets[..., plan["warmup_samples"]:],
                                    teacher_targets, controlled_terms, tuple(codes), plan["additional_loss_weight"]),
                    "augmented_batch_sha256": digest, "teacher_targets_sha256": teacher_digest,
                    "pristine_batch_sha256": pristine_digest, "original_augmentation_sha256": original_digest,
                    "augmentation_rng_before_sha256": rng_before, "augmentation_rng_after_sha256": rng_after,
                    "view_codes": codes, "model_and_backward_rng_unchanged": True,
                    "forced_views_use_pristine_sources": True,
                    "data_hops": output.data_hops, "flush_hops": output.flush_hops,
                    "initial_state_detached": output.initial_state_detached,
                })
                del mixture, targets, flags, output, terms, controlled_terms, loss, teacher_targets, rng
            require(len(microbatches) == plan["accumulation_steps"], "Incomplete gradient accumulation")
            mixer_gradients = {name: float(p.grad.detach().abs().max())
                               for name, p in model.named_parameters() if name.startswith("to_spec_masks.mixer.")}
            inherited_gradients = {}
            if step == 0:
                from research.direct.latency58_vocal_focus_model import parent_parameter_name
                inherited_gradients = {parent_parameter_name(name): hashlib.sha256(
                    p.grad.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                    for name, p in model.named_parameters() if not name.startswith("to_spec_masks.mixer.")}
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True, foreach=False)
            optimizer.step()
            torch.cuda.synchronize()
            step += 1
            means = {k: sum(m[k] for m in microbatches) / plan["accumulation_steps"] for k in LOSS_KEYS}
            identities = {k: hashlib.sha256("".join(m[k] for m in microbatches).encode("ascii")).hexdigest()
                          for k in ("augmented_batch_sha256", "teacher_targets_sha256", *IDENTITY_KEYS)}
            row = {"step": step, "lr": lr, **means, **identities,
                   "arm": plan["arm"], "focused_augmentation": plan["focused_augmentation"],
                   "teacher_mode": plan["teacher_mode"], "counterfactual_version": plan["counterfactual_version"],
                   "additional_loss_weight": plan["additional_loss_weight"],
                   "additional_loss_version": plan["additional_loss_version"],
                   "controlled_examples": sum(m["controlled_examples"] for m in microbatches),
                   "ordinary_examples": sum(m["ordinary_examples"] for m in microbatches),
                   "local_mask_mixer": plan["local_mask_mixer"],
                   "augmentation_version": plan["augmentation_version"],
                   "parameter_tensors": plan["parameter_tensors"],
                   "first_update_inherited_gradient_sha256": inherited_gradients,
                   "mixer_gradient_maxima_before_clip": mixer_gradients,
                   "teacher_weight": plan["teacher_weight"], "teacher_kind": plan["teacher_kind"],
                   "drum_weight": plan["drum_weight"], "objective_version": plan["objective_version"],
                   "grad_norm": float(norm), "deranged_examples": sum(m["deranged_examples"] for m in microbatches),
                   "first_sample_index": first, "next_sample_index": first + config["batch_size"],
                   "data_hops": plan["scored_samples"] // 128, "flush_hops": 1, "stage_start_step": start,
                   "carry_state": plan["carry_state"], "initial_state_detached": True,
                   "warmup_samples": plan["warmup_samples"], "scored_samples": plan["scored_samples"],
                   "accumulation_steps": plan["accumulation_steps"], "microbatch_size": plan["microbatch_size"],
                   "microbatches": microbatches, "gradient_clips_this_update": 1, "adam_steps_this_update": 1,
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
            del mixtures_cpu, targets_cpu, microbatches
        require(not stopped and step == stop, "Stage did not reach its requested endpoint")
        audit_live(model, optimizer, step, frozen, torch, plan, helpers)
        if plan["resource_only"]:
            require(start == 0 and step == stop == 2
                    and state_sha256(teacher.state_dict()) == plan["teacher_model_state_sha256"]
                    and all(not p.requires_grad and p.grad is None for p in teacher.parameters()),
                    "Resource fixture or frozen teacher differs")
            journal = (run / "metrics.jsonl").read_bytes()
            rows = validate_journal(journal, step, plan, helpers)
            fingerprint = state_sha256(model.state_dict())
            require(fingerprint != plan["initialized_model_state_sha256"], "Resource updates did not change learned weights")
            receipt = {"model_state_sha256": fingerprint}
            helpers.atomic_json(run / "resource.json", {
                "schema": "latency58-cleanup-successor-resource-result-v1", "status": "pass",
                "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"],
                "source_bindings_unchanged": True, "initial_model_state_sha256": plan["initialized_model_state_sha256"],
                "final_model_state_sha256": fingerprint, "teacher_kind": plan["teacher_kind"],
                "arm": plan["arm"], "parameter_tensors": plan["parameter_tensors"],
                "teacher_mode": plan["teacher_mode"], "counterfactual_version": plan["counterfactual_version"],
                   "additional_loss_weight": plan["additional_loss_weight"],
                   "additional_loss_version": plan["additional_loss_version"],
                "final_rng_state_sha256": rng_state_sha256(),
                "comparison_variable": "fresh_adam_lower_lr_successor",
                "focused_augmentation": plan["focused_augmentation"],
                "local_mask_mixer": plan["local_mask_mixer"],
                "augmentation_version": plan["augmentation_version"],
                "teacher_weight": plan["teacher_weight"], "teacher_model_state_sha256": plan["teacher_model_state_sha256"],
                "drum_weight": plan["drum_weight"], "objective_version": plan["objective_version"],
                "teacher_unchanged": True, "warmup_samples": plan["warmup_samples"],
                "scored_samples": plan["scored_samples"], "carry_state": True,
                "config": config, "matching_production_updates": rows, "training_updates_executed": step,
                "accumulation_steps": plan["accumulation_steps"], "microbatch_size": plan["microbatch_size"],
                "accumulation_version": VERSION, "augmented_examples_executed": step * config["batch_size"],
                "all_parameter_gradients_present": True, "fixed_buffers_unchanged": True,
                "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
                "checkpoint_written": False, "quality_selected": False})
        else:
            receipt = save_generation(run, model, teacher, optimizer, step, plan, args.plan_sha256, helpers)
        verify_inputs(plan)
        require(sha(args.stage) == args.stage_sha256, "Stage changed during training")
        helpers.atomic_json(run / "status.json", {"status": "resource_complete" if plan["resource_only"] else "complete",
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
