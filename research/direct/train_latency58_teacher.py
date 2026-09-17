"""One fresh 250-update matched arm, with a single atomic split checkpoint.

Model, optimizer and RNG are separate files in one published generation. This
avoids duplicating model tensors for scoring. No intermediate checkpoint or
automatic continuation is made; interrupted runs retain logs and restart from
the authenticated parent in a new directory after inspection.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

from research.direct.train_latency58 import (
    ROOT, PRODUCTION, disk_bytes, load_source, read, require, sha, state_sha256, verify_inputs,
)
from research.direct.train_latency58_asymmetric import continuity


def save_endpoint(run, model, teacher, optimizer, step, plan, plan_sha, helpers, torch, np, frozen):
    require(step == plan["config"]["steps"] == 250, "Publish only the complete bounded endpoint")
    verify_inputs(plan)
    helpers.audit_live(model, optimizer, step, frozen, torch, plan["config"])
    require(teacher is None or (state_sha256(teacher.state_dict()) == plan["teacher_model_state_sha256"]
            and all(not p.requires_grad and p.grad is None for p in teacher.parameters())), "Frozen teacher changed")
    require(disk_bytes(run.parent) + 350_000_000 < plan["artifact_allowance_bytes"], "No reserved checkpoint space")
    pending, final = run / "endpoint.pending", run / "endpoint"
    require(not pending.exists() and not final.exists(), "Preserve existing endpoint files")
    pending.mkdir()
    tensors = helpers.tensor_tree_cpu(model.state_dict(), torch)
    fingerprint = state_sha256(tensors)
    provenance = copy.deepcopy(model.provenance)
    provenance.update(training_updates=4750 + step, pilot_updates=2500 + step,
                      asymmetric_training_updates=500 + step, tail_updates=step,
                      teacher_trial_updates=step, parent_checkpoint=plan["parent_checkpoint"],
                      parent_model_state_sha256=plan["initial_model_state_sha256"],
                      training_objective=plan["objective"], teacher_weight=plan["teacher_weight"],
                      teacher_model_state_sha256=plan["teacher_model_state_sha256"],
                      training_plan_sha256=plan_sha, training_precision=plan["precision_policy"])
    payload = {"schema": "latency58-teacher-inference-v1", "step": step, "model": tensors,
               "model_state_sha256": fingerprint, "provenance": provenance,
               "architecture": model.architecture_metadata, "plan_sha256": plan_sha}
    rng = {"python": random.getstate(), "numpy": np.random.get_state(),
           "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}
    helpers.validate_rng(rng, torch)
    journal = (run / "metrics.jsonl").read_bytes()
    helpers.validate_journal(journal, step, plan["config"])
    for name, value in (("model.pt", payload),
                        ("optimizer.pt", helpers.tensor_tree_cpu(optimizer.state_dict(), torch)), ("rng.pt", rng)):
        with (pending / name).open("xb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
    receipt = {"schema": "latency58-teacher-endpoint-v1", "step": step, "plan_sha256": plan_sha,
               "model_state_sha256": fingerprint, "files": {
                   name: {"sha256": sha(pending / name), "bytes": (pending / name).stat().st_size}
                   for name in ("model.pt", "optimizer.pt", "rng.pt")},
               "next_sample_index": plan["config"]["data_start"] + step * plan["config"]["batch_size"],
               "optimizer_parameter_names": [name for name, _ in model.named_parameters()],
               "metrics_sha256": hashlib.sha256(journal).hexdigest(), "metrics_bytes": len(journal),
               "teacher_unchanged_and_no_gradients": True, "teacher_used": teacher is not None}
    helpers.atomic_json(pending / "receipt.json", receipt)
    helpers.fsync_dir(pending)
    os.rename(pending, final)
    helpers.fsync_dir(run)
    helpers.atomic_json(run / "latest.json", {"endpoint": str(final), "receipt_sha256": sha(final / "receipt.json"),
                                              "step": step, "plan_sha256": plan_sha})
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Plan or cwd differs")
    plan = read(args.plan)
    verify_inputs(plan)
    config = plan["config"]
    require(plan["schema"] == "latency58-teacher-training-plan-v1"
            and config["steps"] == config["checkpoint_every"] == 250
            and config["device"] == "cuda" and config["precision"] == "bf16"
            and config["batch_size"] == 4 and config["workers"] == 2 and config["crop_samples"] == 88064
            and plan["teacher_weight"] in (0.0, 0.5)
            and plan["objective"] == ("raw4_l1" if plan["teacher_weight"] == 0 else "raw4_l1_plus_teacher_l1"),
            "Only the declared fresh matched arms are supported")
    require(all(os.environ.get(k) == v for k, v in plan["environment"].items()), "GPU environment differs")
    previous = read(plan["previous_monitor"]["path"])
    require(sha(plan["previous_monitor"]["path"]) == plan["previous_monitor"]["sha256"]
            and previous["status"] == previous["supervisor_health"] == "pass"
            and previous["child_exit_code"] == 0 and previous["post_exit_quiet_completed"]
            and previous["last_event_record_id"] == plan["previous_event_record_id"], "Prior GPU monitor did not pass")
    monitor = load_source("latency58_teacher_monitor", plan["watchdog_source"])
    _, events = continuity(plan, monitor)
    helpers = load_source("latency58_teacher_training_helpers", plan["helper_source"])
    run = Path(plan["run_dir"])
    require(run.parent == ROOT / "research/direct/runs/latency58" and not run.exists(), "Require a fresh phase run")
    require(disk_bytes(run.parent) + 350_000_000 < plan["artifact_allowance_bytes"], "No room for final checkpoint")
    run.mkdir()
    lock = (run / "trainer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    helpers.atomic_json(run / "config.json", plan)
    helpers.atomic_json(run / "continuity.json", events)
    sys.path.insert(0, str(PRODUCTION))
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import train_production as production
    from research import experiment
    from research.direct.latency58 import HOP
    from research.direct.latency58_asymmetric import PRECISION_POLICY
    from research.direct.latency58_asymmetric_checkpoint import make_model, load_model_state
    from research.direct.latency58_teacher import load_frozen_teacher, physical_teacher_targets, deployed_teacher_l1
    from research.direct.latency_ola512_training import raw4_native_objective

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = make_model(plan["hann_parent_checkpoint"])
    require(load_model_state(model, plan["parent_checkpoint"]) == 500
            and state_sha256(model.state_dict()) == plan["initial_model_state_sha256"], "Student parent differs")
    teacher = None
    if plan["teacher_weight"]:
        teacher, identity = load_frozen_teacher(plan["teacher_export_plan"])
        require(identity["model_state_sha256"] == plan["teacher_model_state_sha256"], "Teacher differs")
    require(torch.__version__ == plan["torch_version"] and PRECISION_POLICY == plan["precision_policy"]
            and torch.cuda.is_available() and torch.cuda.device_count() == 1
            and torch.cuda.is_bf16_supported(), "Reviewed GPU precision/runtime unavailable")
    torch.cuda.set_per_process_memory_fraction(0.75)
    production.configure_determinism(config["seed"])
    production_config = read(PRODUCTION / "full_config.json")
    _, tracks, manifest_sha, _ = production.load_corpus_manifest(
        PRODUCTION / "manifests/combined.manifest.json", expected_file_sha256=plan["manifest_sha256"], config=production_config)
    require(manifest_sha == plan["manifest_sha256"]
            and production_config["sampling"]["root_weights"] == config["root_weights"]
            and production_config["seed"] == config["data_seed"]
            and production_config["sampling"]["vocal_active_probability"] == config["vocal_active_probability"],
            "Production sampling differs")
    model.cuda().train().requires_grad_(True)
    model.training_precision = "bf16"
    if teacher is not None:
        teacher.cuda()
    frozen = {name: value.clone() for name, value in model.named_buffers()}
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], foreach=False)
    helpers.audit_live(model, optimizer, 0, frozen, torch, config)
    dataset = production.CounterAddressedCropDataset(
        tracks, root_weights=config["root_weights"], seed=config["data_seed"], crop_samples=config["crop_samples"],
        vocal_active_probability=config["vocal_active_probability"],
        final_sample_index=config["data_start"] + config["steps"] * config["batch_size"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], num_workers=config["workers"],
        sampler=production.AbsoluteIndexSampler(config["data_start"], config["data_start"] + 250 * 4),
        pin_memory=True, worker_init_fn=production.worker_init,
        generator=torch.Generator().manual_seed(config["seed"] + 1),
        multiprocessing_context="spawn", prefetch_factor=2)
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    step, began = 0, time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    helpers.atomic_json(run / "status.json", {"status": "running", "step": 0, "pid": os.getpid()})
    try:
        for mixture, targets in loader:
            require(not stopped, "Stop requested; no partial endpoint is published")
            first = config["data_start"] + step * 4
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
            if teacher is not None:
                rng_before = torch.cuda.get_rng_state().clone()
                teacher_targets = physical_teacher_targets(teacher, mixture)
                require(torch.equal(rng_before, torch.cuda.get_rng_state()), "Teacher perturbed the matched GPU RNG")
            padded = torch.cat((mixture, mixture.new_zeros((4, 2, HOP))), dim=-1)
            output = model.render(padded)
            require(torch.equal(output.delayed_mixture[..., HOP:HOP + config["crop_samples"]], mixture),
                    "Student physical alignment differs")
            raw, deployed = (v[..., HOP:HOP + config["crop_samples"]] for v in (output.raw, output.deployed))
            terms = raw4_native_objective(raw, targets, flags, projection=True)
            auxiliary = deployed_teacher_l1(deployed, teacher_targets) if teacher is not None else terms.total.new_zeros(())
            total = terms.total + plan["teacher_weight"] * auxiliary
            total.backward()
            require(all(p.grad is not None for p in model.parameters()), "Missing learned gradient")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True, foreach=False)
            optimizer.step()
            torch.cuda.synchronize()
            step += 1
            metrics = {"step": step, "lr": lr, "loss": float(total.detach()),
                "supervised_loss": float(terms.total.detach()), "waveform_l1": float(terms.waveform_l1.detach()),
                "projection": float(terms.projection.detach()), "projection_contribution": float(terms.projection_contribution.detach()),
                "teacher_l1": float(auxiliary.detach()), "teacher_weight": plan["teacher_weight"],
                "deployed_waveform_l1": float(torch.nn.functional.l1_loss(deployed.detach(), targets)),
                "augmented_batch_sha256": digest.hexdigest(), "objective": plan["objective"],
                "grad_norm": float(norm), "deranged_examples": int(flags.sum()),
                "first_sample_index": first, "next_sample_index": first + 4, "data_hops": 688, "flush_hops": 1,
                "elapsed_seconds": time.monotonic() - began, "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30}
            with (run / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            print(json.dumps(metrics, allow_nan=False), flush=True)
            del raw, deployed, output, padded, terms, auxiliary, total
            if teacher is not None:
                del teacher_targets
        require(not stopped and step == 250, "Bounded arm did not reach its complete endpoint")
        receipt = save_endpoint(run, model, teacher, optimizer, step, plan, args.plan_sha256, helpers, torch, np, frozen)
        require(sha(args.plan) == args.plan_sha256, "Training plan changed")
        verify_inputs(plan)
        helpers.atomic_json(run / "status.json", {"status": "complete", "step": step,
                            "elapsed_seconds": time.monotonic() - began, "model_state_sha256": receipt["model_state_sha256"]})
    except BaseException as error:
        helpers.atomic_json(run / "status.json", {"status": "failed", "step": step, "error": repr(error)})
        raise
    lock.close()


if __name__ == "__main__":
    main()
