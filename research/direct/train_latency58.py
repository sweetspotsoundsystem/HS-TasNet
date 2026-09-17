"""Bounded GPU-only hop128 adaptation with one atomically replaced resume file.

Only requested stages run. Keep compact receipts for every saved stage and
write an inference snapshot only at an explicit quality endpoint. The accepted
checkpoint is read-only. No computation starts on import.
"""
from __future__ import annotations

import argparse
import base64
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = Path("/home/axel/autoresearch/production/hs-tasnet-c91-full-v1")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def state_sha256(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def verify_inputs(plan):
    require(all(sha(path) == value for path, value in plan["source_bindings"].items()),
            "Training source or prerequisite changed")


def disk_bytes(root):
    total = 0
    for parent, _, files in os.walk(root):
        for name in files:
            path = Path(parent) / name
            if not path.is_symlink():
                total += path.stat().st_size
    return total


def continuity(stage, monitor):
    command = monitor.event_query(stage["previous_event_record_id"], verify_sentinel=True)
    response = subprocess.run([monitor.POWERSHELL, "-NoProfile", "-NonInteractive", "-EncodedCommand",
                               base64.b64encode(command.encode("utf-16le")).decode()],
                              capture_output=True, text=True, timeout=10)
    require(response.returncode == 0, "Pre-training host query failed")
    payload = json.loads(response.stdout)
    newest, rows = monitor.validate_events(payload, stage["previous_event_record_id"])
    require(payload["SentinelVerified"] and not any(monitor.reset_event(row) for row in rows),
            "New host fault since previous monitored GPU stage")
    return newest, payload


def save(run, model, optimizer, step, plan, plan_sha, helpers, torch, np, frozen, save_inference):
    verify_inputs(plan)
    helpers.audit_live(model, optimizer, step, frozen, torch, plan["config"])
    # Bound peak disk use before creating the next resume payload. The old
    # resume stays usable until the new complete file is atomically installed.
    phase_root = ROOT / "research/direct/runs/latency58"
    require(disk_bytes(phase_root) + 350_000_000 < plan["artifact_allowance_bytes"],
            "No reserved room for an atomic resume replacement")
    tensors = helpers.tensor_tree_cpu(model.state_dict(), torch)
    fingerprint = state_sha256(tensors)
    provenance = copy.deepcopy(model.provenance)
    provenance.update(pilot_updates=step, training_updates=2250 + step,
                      training_plan_sha256=plan_sha, training_precision=plan["precision_policy"])
    rng = {"python": random.getstate(), "numpy": np.random.get_state(),
           "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}
    helpers.validate_rng(rng, torch)
    journal = (run / "metrics.jsonl").read_bytes()
    helpers.validate_journal(journal, step, plan["config"])
    payload = {"schema": "latency58-resume-v1", "step": step, "model": tensors,
               "model_state_sha256": fingerprint, "provenance": provenance,
               "architecture": model.architecture_metadata, "plan_sha256": plan_sha,
               "next_sample_index": plan["config"]["data_start"] + step * plan["config"]["batch_size"],
               "optimizer_parameter_names": [name for name, _ in model.named_parameters()],
               "optimizer": helpers.tensor_tree_cpu(optimizer.state_dict(), torch), "rng": rng,
               "metrics_sha256": hashlib.sha256(journal).hexdigest(), "metrics_bytes": len(journal)}
    temporary = run / f"resume-step-{step:06d}.pending.pt"
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    resume_digest = sha(temporary)
    receipt = {"schema": "latency58-checkpoint-receipt-v1", "step": step,
               "resume_sha256": resume_digest, "model_state_sha256": fingerprint,
               "plan_sha256": plan_sha, "metrics_sha256": payload["metrics_sha256"],
               "next_sample_index": payload["next_sample_index"], "inference": None}
    if save_inference:
        require(disk_bytes(phase_root) + 115_000_000 < plan["artifact_allowance_bytes"],
                "No reserved room for an inference snapshot")
        path = run / f"model-step-{step:06d}.pt"
        inference = {key: payload[key] for key in ("step", "model", "model_state_sha256", "provenance",
                                                  "architecture", "plan_sha256")}
        inference["schema"] = "latency58-inference-v1"
        with path.open("xb") as stream:
            torch.save(inference, stream)
            stream.flush()
            os.fsync(stream.fileno())
        receipt["inference"] = {"path": str(path), "sha256": sha(path)}
    receipt_path = run / "receipts" / f"step-{step:06d}.json"
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, run / "resume.pt")
    helpers.fsync_dir(run)
    helpers.atomic_json(run / "latest.json", {**receipt, "receipt_path": str(receipt_path),
                                              "receipt_sha256": sha(receipt_path)})
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--stage-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256 and sha(args.stage) == args.stage_sha256,
            "Plan or stage changed")
    plan, stage = read(args.plan), read(args.stage)
    require(plan["schema"] == "latency58-training-plan-v1"
            and stage["schema"] == "latency58-training-stage-v1"
            and stage["plan_sha256"] == args.plan_sha256, "Wrong training contract")
    verify_inputs(plan)
    require(all(os.environ.get(name) == value for name, value in plan["environment"].items()),
            "Training environment differs")
    config = plan["config"]
    require(config["device"] == "cuda" and config["precision"] == "bf16"
            and config["batch_size"] == 4 and config["workers"] == 2
            and config["crop_samples"] == 88064
            and 0 <= stage["start_step"] < stage["stop_step"] <= config["steps"]
            and stage["stop_step"] - stage["start_step"] <= 250,
            "Only bounded GPU B4 stages are allowed")
    if stage["start_step"]:
        audit_path = Path(stage["previous_checkpoint_audit"]["path"])
        require(sha(audit_path) == stage["previous_checkpoint_audit"]["sha256"], "Prior saved-state audit changed")
        audit = read(audit_path)
        require(audit["status"] == "pass" and audit["step"] == stage["start_step"]
                and audit["resume_sha256"] == stage["resume_sha256"], "Prior saved state did not pass")
    previous_monitor = Path(stage["previous_monitor"]["path"])
    require(sha(previous_monitor) == stage["previous_monitor"]["sha256"], "Previous monitor result changed")
    previous = read(previous_monitor)
    require(previous["status"] == "pass" and previous["child_exit_code"] == 0
            and previous["post_exit_quiet_completed"] and previous["supervisor_health"] == "pass"
            and previous["last_event_record_id"] == stage["previous_event_record_id"],
            "Previous monitor did not finish successfully")
    monitor = load_source("latency58_training_monitor", plan["watchdog_source"])
    _, events = continuity(stage, monitor)
    with (args.stage.parent / "continuity.json").open("x") as stream:
        json.dump(events, stream, indent=2)
    helpers = load_source("latency58_frozen_training_helpers", plan["helper_source"])
    run = Path(plan["run_dir"])
    require(run.parent == ROOT / "research/direct/runs/latency58", "Unexpected run directory")
    resuming = stage["start_step"] > 0
    require(run.is_dir() if resuming else not run.exists(), "Explicit fresh/resume state differs")
    run.mkdir(exist_ok=resuming)
    (run / "receipts").mkdir(exist_ok=resuming)
    lock = (run / "trainer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if resuming:
        require(read(run / "config.json") == plan, "Saved run configuration changed")
    else:
        helpers.atomic_json(run / "config.json", plan)
    sys.path.insert(0, str(PRODUCTION))
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import train_production as production
    from research import experiment
    from research.direct.latency58_gpu import Latency58GPUModel, POLICY
    from research.direct.latency58 import HOP
    from research.direct.latency_ola512_training import raw4_native_objective

    require(torch.__version__ == plan["torch_version"] and POLICY == plan["precision_policy"]
            and torch.cuda.is_available() and torch.cuda.device_count() == 1
            and torch.cuda.is_bf16_supported(), "Reviewed GPU precision/runtime unavailable")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(0.75)
    production.configure_determinism(config["seed"])
    production_config = read(PRODUCTION / "full_config.json")
    _, tracks, manifest_sha, _ = production.load_corpus_manifest(
        PRODUCTION / "manifests/combined.manifest.json", expected_file_sha256=plan["manifest_sha256"],
        config=production_config)
    require(manifest_sha == plan["manifest_sha256"]
            and production_config["sampling"]["root_weights"] == config["root_weights"]
            and production_config["seed"] == config["data_seed"]
            and production_config["sampling"]["vocal_active_probability"] == config["vocal_active_probability"],
            "Production sampling contract changed")
    model = Latency58GPUModel.from_accepted().cuda().train().requires_grad_(True)
    model.training_precision = "bf16"
    frozen = {name: value.clone() for name, value in model.named_buffers()}
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], foreach=False)
    step = stage["start_step"]
    if resuming:
        require(sha(run / "resume.pt") == stage["resume_sha256"], "Resume file changed")
        stored = torch.load(run / "resume.pt", map_location="cpu", weights_only=False)
        require(stored["schema"] == "latency58-resume-v1" and stored["step"] == step
                and stored["plan_sha256"] == args.plan_sha256
                and stored["architecture"] == model.architecture_metadata
                and stored["model_state_sha256"] == state_sha256(stored["model"])
                and stored["next_sample_index"] == config["data_start"] + step * config["batch_size"]
                and stored["optimizer_parameter_names"] == [name for name, _ in model.named_parameters()],
                "Saved resume identity, geometry or data position differs")
        journal = (run / "metrics.jsonl").read_bytes()
        require(hashlib.sha256(journal).hexdigest() == stored["metrics_sha256"]
                and len(journal) == stored["metrics_bytes"], "Journal extends beyond the saved completed update")
        helpers.validate_journal(journal, step, config)
        model.load_state_dict(stored["model"], strict=True)
        model.provenance = stored["provenance"]
        optimizer.load_state_dict(stored["optimizer"])
        random.setstate(stored["rng"]["python"])
        np.random.set_state(stored["rng"]["numpy"])
        torch.set_rng_state(stored["rng"]["torch_cpu"])
        torch.cuda.set_rng_state_all(stored["rng"]["torch_cuda"])
        del stored
    helpers.audit_live(model, optimizer, step, frozen, torch, config)
    dataset = production.CounterAddressedCropDataset(
        tracks, root_weights=config["root_weights"], seed=config["data_seed"],
        crop_samples=config["crop_samples"], vocal_active_probability=config["vocal_active_probability"],
        final_sample_index=config["data_start"] + config["steps"] * config["batch_size"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], num_workers=config["workers"],
        sampler=production.AbsoluteIndexSampler(config["data_start"] + step * config["batch_size"],
            config["data_start"] + stage["stop_step"] * config["batch_size"]),
        pin_memory=True, worker_init_fn=production.worker_init,
        generator=torch.Generator().manual_seed(config["seed"] + 1),
        multiprocessing_context="spawn", prefetch_factor=2)
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    helpers.atomic_json(run / "status.json", {"status": "running", "step": step, "pid": os.getpid()})
    try:
        for mixture, targets in loader:
            if stopped:
                break
            first_index = config["data_start"] + step * config["batch_size"]
            lr = helpers.learning_rate(step, config)
            optimizer.param_groups[0]["lr"] = lr
            mixture, targets = mixture.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            mixture, targets, flags = experiment._augment_training_distribution(mixture=mixture, targets=targets)
            padded = torch.cat((mixture, mixture.new_zeros((config["batch_size"], 2, HOP))), dim=-1)
            output = model.render(padded)
            require(torch.equal(output.delayed_mixture[..., HOP:HOP + config["crop_samples"]], mixture),
                    "Training output lost physical crop alignment")
            raw = output.raw[..., HOP:HOP + config["crop_samples"]]
            loss = raw4_native_objective(raw, targets, flags, projection=True)
            loss.total.backward()
            require(all(p.grad is not None for p in model.parameters()), "Missing trainable parameter gradient")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True, foreach=False)
            optimizer.step()
            torch.cuda.synchronize()
            step += 1
            metrics = {"step": step, "lr": lr, "loss": float(loss.total.detach()),
                "waveform_l1": float(loss.waveform_l1.detach()), "projection": float(loss.projection.detach()),
                "projection_contribution": float(loss.projection_contribution.detach()), "grad_norm": float(norm),
                "deranged_examples": int(flags.sum()), "first_sample_index": first_index,
                "next_sample_index": first_index + config["batch_size"], "data_hops": 688, "flush_hops": 1,
                "elapsed_seconds": time.monotonic() - start,
                "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30}
            with (run / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            print(json.dumps(metrics, allow_nan=False), flush=True)
            del output, raw, loss, padded
            if step % config["checkpoint_every"] == 0 or step == stage["stop_step"] or stopped:
                save(run, model, optimizer, step, plan, args.plan_sha256, helpers, torch, np, frozen,
                     save_inference=stage["save_inference"] and step == stage["stop_step"])
            if stopped:
                break
        require(step > stage["start_step"], "No completed update to preserve")
        if not (run / "latest.json").exists() or read(run / "latest.json")["step"] != step:
            save(run, model, optimizer, step, plan, args.plan_sha256, helpers, torch, np, frozen, False)
    except BaseException as error:
        helpers.atomic_json(run / "status.json", {"status": "failed", "step": step, "error": repr(error)})
        raise
    verify_inputs(plan)
    require(sha(args.plan) == args.plan_sha256 and sha(args.stage) == args.stage_sha256, "Run contracts changed")
    status = {"status": "paused", "step": step, "requested_stop_step": stage["stop_step"],
              "stop_requested": stopped, "elapsed_seconds": time.monotonic() - start,
              "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
              "resume_sha256": sha(run / "resume.pt"), "plan_sha256": args.plan_sha256}
    helpers.atomic_json(run / "status.json", status)
    print(json.dumps(status, allow_nan=False), flush=True)
    require(step == stage["stop_step"], "Stopped before requested endpoint; audit before continuation")
    lock.close()


if __name__ == "__main__":
    main()
