"""Prepared standalone GPU-only OLA pilot. Not released or executed.

Use a separately reviewed, hash-pinned plan and explicit bounded stop step.
No existing latency trainer, runtime loader or evaluation file is imported.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
PRODUCTION = Path("/home/axel/autoresearch/production/hs-tasnet-c91-full-v1")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def fsync_dir(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)
    fsync_dir(path.parent)


def verify_bindings(plan):
    for path, expected in plan["source_bindings"].items():
        require(sha(path) == expected, f"Reviewed source/input changed: {path}")


def learning_rate(step, config):
    peak, floor, total, warmup = (config[key] for key in ("lr", "min_lr", "steps", "warmup"))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    return floor + (peak - floor) * .5 * (1 + math.cos(math.pi * progress))


def tensor_tree_cpu(value, torch):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: tensor_tree_cpu(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [tensor_tree_cpu(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(tensor_tree_cpu(item, torch) for item in value)
    return copy.deepcopy(value)


def validate_journal(raw, step, config):
    require(type(step) is int and step >= 0, "Invalid journal endpoint")
    require(not raw or raw.endswith(b"\n"), "Incomplete metrics append")
    rows = [json.loads(line) for line in raw.splitlines()]
    require(len(rows) == step, "Metrics row count differs from completed updates")
    for index, row in enumerate(rows):
        first = config["data_start"] + index * config["batch_size"]
        require(type(row.get("step")) is int and row["step"] == index + 1
                and row.get("first_sample_index") == first
                and row.get("next_sample_index") == first + config["batch_size"]
                and row.get("lr") == learning_rate(index, config),
                f"Metrics step/LR/sample prefix differs at update {index + 1}")


def validate_adam_group(group, parameter_ids, step, config):
    recipe = {"betas": (.9, .999), "eps": 1e-8, "weight_decay": 0,
              "amsgrad": False, "maximize": False, "foreach": False,
              "capturable": False, "differentiable": False, "fused": None,
              "decoupled_weight_decay": False}
    expected_lr = config["lr"] if step == 0 else learning_rate(step - 1, config)
    require(set(group) == set(recipe) | {"params", "lr"}
            and all(group[key] == value for key, value in recipe.items())
            and group["lr"] == expected_lr, "Adam recipe or saved LR differs")
    require(len(group["params"]) == len(parameter_ids)
            and all(a == b for a, b in zip(group["params"], parameter_ids)),
            "Adam parameter order/mapping differs")


def validate_rng(rng, torch):
    require(set(rng) == {"python", "numpy", "torch_cpu", "torch_cuda"}, "RNG inventory differs")
    require(isinstance(rng["torch_cuda"], list) and len(rng["torch_cuda"]) == 1,
            "Require exactly one saved CUDA RNG state")
    for value, expected in ((rng["torch_cpu"], torch.get_rng_state()),
                            (rng["torch_cuda"][0], torch.cuda.get_rng_state(0))):
        require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                and value.dtype == torch.uint8 and value.ndim == 1 and value.shape == expected.shape,
                "Malformed CPU/CUDA RNG byte state")


def audit_live(model, optimizer, step, frozen_buffers, torch, config):
    parameters = list(model.parameters())
    require(len(parameters) == 21 and all(p.requires_grad and p.dtype == torch.float32 for p in parameters),
            "Expected all 21 FP32 OLA parameters trainable")
    require(all(bool(torch.isfinite(p).all()) for p in parameters), "Non-finite model parameter")
    for name, value in model.named_buffers():
        require(torch.equal(value, frozen_buffers[name]), f"Fixed buffer/gain changed: {name}")
    require(len(optimizer.param_groups) == 1, "Expected one Adam parameter group")
    live_group = optimizer.param_groups[0]
    require(len(live_group["params"]) == len(parameters)
            and all(a is b for a, b in zip(live_group["params"], parameters)),
            "Live Adam parameter identity/order differs")
    validate_adam_group({**live_group, "params": list(range(21))}, list(range(21)), step, config)
    require(set(optimizer.state) == (set(parameters) if step else set()), "Unexpected Adam state inventory")
    for parameter, state in optimizer.state.items():
        require(set(state) == {"step", "exp_avg", "exp_avg_sq"}, "Unexpected Adam state schema")
        require(state["step"].shape == () and state["step"].dtype == torch.float32
                and state["step"].device.type == "cpu" and float(state["step"]) == step,
                "Adam scalar step differs from completed-update count")
        for name in ("exp_avg", "exp_avg_sq"):
            value = state[name]
            require(value.dtype == torch.float32 and value.shape == parameter.shape
                    and bool(torch.isfinite(value).all()), f"Invalid Adam {name}")


def save_generation(run, model, optimizer, step, plan, plan_sha, frozen_buffers, torch, np, state_hash):
    """Publish a complete immutable generation, then atomically advance pointer."""
    verify_bindings(plan)
    audit_live(model, optimizer, step, frozen_buffers, torch, plan["config"])
    generations = run / "checkpoints"
    generations.mkdir(exist_ok=True)
    destination = generations / f"step-{step:06d}"
    require(not destination.exists(), "Refuse to overwrite an immutable checkpoint generation")
    temporary = Path(tempfile.mkdtemp(prefix=f"pending-{step:06d}-", dir=generations))
    tensors = tensor_tree_cpu(model.state_dict(), torch)
    provenance = copy.deepcopy(model.provenance)
    provenance.update(training_updates=step, training_plan_sha256=plan_sha,
                      training_precision_policy=plan["precision_policy"])
    identity = state_hash(tensors)
    inference = {"schema": "ola512-inference-v1", "step": step, "model": tensors,
                 "model_state_sha256": identity, "provenance": provenance,
                 "architecture": model.architecture_metadata, "plan_sha256": plan_sha}
    rng = {"python": random.getstate(), "numpy": np.random.get_state(),
           "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}
    validate_rng(rng, torch)
    payload = {"schema": "ola512-resume-v1", "step": step, "model": tensors,
               "model_state_sha256": identity, "provenance": provenance, "plan_sha256": plan_sha,
               "next_sample_index": plan["config"]["data_start"] + step * plan["config"]["batch_size"],
               "optimizer_parameter_names": [name for name, _ in model.named_parameters()],
               "optimizer": tensor_tree_cpu(optimizer.state_dict(), torch), "rng": rng}
    for name, value in (("resume.pt", payload), ("model.pt", inference)):
        with (temporary / name).open("xb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
    metrics = (run / "metrics.jsonl").read_bytes() if (run / "metrics.jsonl").exists() else b""
    validate_journal(metrics, step, plan["config"])
    with (temporary / "metrics.jsonl").open("xb") as stream:
        stream.write(metrics)
        stream.flush()
        os.fsync(stream.fileno())
    inventory = {name: sha(temporary / name) for name in ("resume.pt", "model.pt", "metrics.jsonl")}
    atomic_json(temporary / "receipt.json", {"status": "pass", "step": step, "files": inventory,
                "model_state_sha256": identity, "plan_sha256": plan_sha})
    fsync_dir(temporary)
    os.rename(temporary, destination)
    fsync_dir(generations)
    pointer = {"schema": "ola512-checkpoint-pointer-v1", "step": step, "generation": str(destination),
               "files": inventory, "receipt_sha256": sha(destination / "receipt.json"), "plan_sha256": plan_sha}
    atomic_json(run / "latest.json", pointer)
    return pointer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--stop-after", type=int, required=True)
    parser.add_argument("--resume-sha256", help="Required existing immutable resume file hash when continuing")
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Training plan differs from reviewed hash")
    plan = read(args.plan)
    require(plan["schema"] == "ola512-gpu-training-plan-v1"
            and plan["release"] == "root_reviewed_gpu_stage", "Prepared plan is not released")
    verify_bindings(plan)
    config = plan["config"]
    require(config["device"] == "cuda" and config["precision"] == "bf16", "GPU BF16 only")
    require(config["batch_size"] in (2, 4, 8) and type(config["workers"]) is int and 0 <= config["workers"] <= 4,
            "Use a reviewed B2/B4/B8 resource setting")
    require(0 < args.stop_after <= config["steps"] and 0 <= config["warmup"] < config["steps"]
            and 0 < config["min_lr"] <= config["lr"] and config["checkpoint_every"] > 0,
            "Invalid bounded stage or learning-rate schedule")
    require(config["crop_samples"] == 88064 and config["data_start"] >= 0, "Unexpected crop/sample schedule")
    for name, wanted in plan["environment"].items():
        require(os.environ.get(name) == wanted, f"Launch environment differs: {name}")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "Explicit GPU0 required")
    run = Path(plan["run_dir"])
    require(run.is_absolute() and run.parent == ROOT / "research/direct/runs/latency11", "Use a new isolated run")
    resume = args.resume_sha256 is not None
    require(run.is_dir() if resume else not run.exists(), "Use explicit resume only for an existing run")
    run.mkdir(exist_ok=True)
    lock = (run / "trainer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if resume:
        require(read(run / "config.json") == plan, "Existing run configuration differs")
    else:
        atomic_json(run / "config.json", plan)
    # Imports are below stdlib preflight; no imported module starts a workload.
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(PRODUCTION))
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import train_production as production
    from research import experiment
    from research.direct.latency_ola512_training import aligned_ola512_crop, raw4_native_objective
    from ola_gpu_model import OLAGPUModel, POLICY, RIGHT_STATE, build_right_baked, state_sha256

    require(plan["precision_policy"] == POLICY, "Precision implementation differs")
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported() and torch.cuda.device_count() == 1,
            "Reviewed single CUDA BF16 device unavailable; no CPU fallback")
    require(torch.__version__ == plan["torch_version"], "PyTorch version differs")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    production.configure_determinism(config["seed"])
    production_config = read(PRODUCTION / "full_config.json")
    _, tracks, manifest_hash, _ = production.load_corpus_manifest(
        PRODUCTION / "manifests/combined.manifest.json",
        expected_file_sha256=plan["manifest_sha256"], config=production_config)
    require(manifest_hash == plan["manifest_sha256"] and production_config["sampling"]["root_weights"] == config["root_weights"],
            "Corpus or native 0.5/0.25/0.25 sampling differs")
    require(production_config["seed"] == config["data_seed"]
            and production_config["sampling"]["vocal_active_probability"] == config["vocal_active_probability"],
            "Data seed or vocal-active sampling differs")
    model = build_right_baked()
    require(state_sha256(model.state_dict()) == RIGHT_STATE, "Wrong initial OLA state")
    model = model.to("cuda", dtype=torch.float32).train().requires_grad_(True)
    model.training_precision = "bf16"
    frozen_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], foreach=False)
    step = 0
    if resume:
        pointer = read(run / "latest.json")
        generation = Path(pointer["generation"])
        require(generation.parent == run / "checkpoints" and generation.name == f"step-{pointer['step']:06d}", "Bad generation path")
        require(pointer["plan_sha256"] == args.plan_sha256, "Checkpoint plan changed")
        require(sha(generation / "receipt.json") == pointer["receipt_sha256"], "Checkpoint receipt changed")
        receipt = read(generation / "receipt.json")
        require(set(pointer["files"]) == {"resume.pt", "model.pt", "metrics.jsonl"}
                and receipt["status"] == "pass" and receipt["step"] == pointer["step"]
                and receipt["plan_sha256"] == pointer["plan_sha256"]
                and receipt["files"] == pointer["files"], "Pointer/receipt inventory or identity differs")
        for name, expected in pointer["files"].items():
            require(sha(generation / name) == expected, f"Checkpoint file changed: {name}")
        require(sha(generation / "resume.pt") == args.resume_sha256, "Resume differs from explicit reviewed identity")
        payload = torch.load(generation / "resume.pt", map_location="cpu", weights_only=False)
        require(payload["schema"] == "ola512-resume-v1" and payload["plan_sha256"] == args.plan_sha256,
                "Resume schema/configuration differs")
        step = payload["step"]
        require(step == pointer["step"] and 0 <= step < args.stop_after, "No updates remain in requested continuation")
        require(payload["next_sample_index"] == config["data_start"] + step * config["batch_size"], "Sample counter differs")
        require(state_sha256(payload["model"]) == payload["model_state_sha256"] == receipt["model_state_sha256"],
                "Stored tensor/receipt identity differs")
        snapshot = torch.load(generation / "model.pt", map_location="cpu", weights_only=False)
        require(snapshot["schema"] == "ola512-inference-v1" and snapshot["step"] == step
                and snapshot["plan_sha256"] == args.plan_sha256
                and state_sha256(snapshot["model"]) == snapshot["model_state_sha256"] == payload["model_state_sha256"],
                "Inference snapshot differs from resumable model identity")
        del snapshot
        names = [name for name, _ in model.named_parameters()]
        require(payload["optimizer_parameter_names"] == names, "Stored Adam parameter names/order differ")
        stored_optimizer = payload["optimizer"]
        require(set(stored_optimizer) == {"state", "param_groups"}
                and len(stored_optimizer["param_groups"]) == 1
                and set(stored_optimizer["state"]) == (set(range(21)) if step else set()),
                "Stored Adam state/group inventory differs")
        validate_adam_group(stored_optimizer["param_groups"][0], list(range(21)), step, config)
        journal = (run / "metrics.jsonl").read_bytes() if (run / "metrics.jsonl").exists() else b""
        require(journal == (generation / "metrics.jsonl").read_bytes(),
                "Journal extends beyond or differs from checkpoint; preserve and inspect before deliberate recovery")
        validate_journal(journal, step, config)
        model.load_state_dict(payload["model"], strict=True)
        model.provenance = payload["provenance"]
        optimizer.load_state_dict(payload["optimizer"])
        rng = payload["rng"]
        validate_rng(rng, torch)
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch_cpu"])
        torch.cuda.set_rng_state_all(rng["torch_cuda"])
        del payload
    audit_live(model, optimizer, step, frozen_buffers, torch, plan["config"])
    if not resume:
        save_generation(run, model, optimizer, 0, plan, args.plan_sha256, frozen_buffers, torch, np, state_sha256)
    dataset = production.CounterAddressedCropDataset(
        tracks, root_weights=config["root_weights"], seed=config["data_seed"], crop_samples=config["crop_samples"],
        vocal_active_probability=config["vocal_active_probability"],
        final_sample_index=config["data_start"] + config["steps"] * config["batch_size"])
    options = {"multiprocessing_context": "spawn", "prefetch_factor": 2} if config["workers"] else {}
    loader = DataLoader(dataset, batch_size=config["batch_size"],
        sampler=production.AbsoluteIndexSampler(config["data_start"] + step * config["batch_size"],
            config["data_start"] + args.stop_after * config["batch_size"]),
        num_workers=config["workers"], pin_memory=True, worker_init_fn=production.worker_init,
        generator=torch.Generator().manual_seed(config["seed"] + 1), **options)
    stopped = False
    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    atomic_json(run / "status.json", {"status": "running", "step": step, "pid": os.getpid()})
    try:
        for mixture, targets in loader:
            if stopped:
                break
            first_index = config["data_start"] + step * config["batch_size"]
            lr = learning_rate(step, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            mixture, targets = mixture.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            mixture, targets, deranged = experiment._augment_training_distribution(mixture=mixture, targets=targets)
            aligned = aligned_ola512_crop(model, mixture, training=True, group_hops=None)
            require(torch.equal(aligned.delayed_mixture, mixture), "Crop physical alignment changed")
            loss = raw4_native_objective(aligned.raw, targets, deranged, projection=True)
            loss.total.backward()
            require(all(p.grad is not None for p in model.parameters()), "A trainable OLA parameter has no gradient")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True, foreach=False)
            optimizer.step()
            torch.cuda.synchronize()
            step += 1
            metrics = {"step": step, "lr": lr, "loss": float(loss.total.detach()),
                "waveform_l1": float(loss.waveform_l1.detach()), "projection": float(loss.projection.detach()),
                "projection_contribution": float(loss.projection_contribution.detach()), "grad_norm": float(grad_norm),
                "deranged_examples": int(deranged.sum()), "first_sample_index": first_index,
                "next_sample_index": first_index + config["batch_size"], "elapsed_seconds": time.monotonic() - started,
                "data_hops": 344, "flush_hops": 1, "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30}
            with (run / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            print(json.dumps(metrics, allow_nan=False), flush=True)
            del aligned, loss
            if step % config["checkpoint_every"] == 0 or step == args.stop_after or stopped:
                save_generation(run, model, optimizer, step, plan, args.plan_sha256, frozen_buffers, torch, np, state_sha256)
            if stopped:
                break
        # A signal may arrive while fetching the next batch after an update.
        if read(run / "latest.json")["step"] != step:
            save_generation(run, model, optimizer, step, plan, args.plan_sha256, frozen_buffers, torch, np, state_sha256)
    except BaseException as error:
        atomic_json(run / "status.json", {"status": "failed", "step": step, "error": repr(error)})
        raise
    require(stopped or step == args.stop_after, "DataLoader ended before requested stage")
    verify_bindings(plan)
    require(sha(args.plan) == args.plan_sha256, "Plan changed during execution")
    status = {"status": "complete" if step == config["steps"] else "paused", "step": step,
              "stop_requested": stopped, "pid": os.getpid(), "elapsed_seconds": time.monotonic() - started,
              "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
              "latest_checkpoint": read(run / "latest.json"), "host_stability_proven": False}
    atomic_json(run / "status.json", status)
    print(json.dumps(status, allow_nan=False), flush=True)
    lock.close()


if __name__ == "__main__":
    main()
