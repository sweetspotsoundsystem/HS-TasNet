"""Direct C91 continuation: one recipe, one run directory, ordinary checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hs-tasnet-matplotlib")

import numpy as np
import torch
from torch.utils.data import DataLoader

from research import experiment
from research.direct.checkpoints import load_model, save_deployment

PRODUCTION_ROOT = Path("/home/axel/autoresearch/production/hs-tasnet-c91-full-v1")
sys.path.insert(0, str(PRODUCTION_ROOT))
import train_production as production

DEPLOYED_DBV_GAINS = (1.0, 1.0, 0.9)
TEACHER_ROOT = "recordpool_best200_v1"


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def learning_rate(step: int, *, total: int, peak: float, floor: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def train_batch(
    model, optimizer, mixture, targets, *, projection: bool, deployed_l1_weight: float = 0.0,
) -> dict:
    optimizer.zero_grad(set_to_none=True)
    mixture, targets, deranged = experiment._augment_training_distribution(
        mixture=mixture, targets=targets,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        waveform_loss, estimates = model(mixture, targets=targets, return_targets_with_loss=True)
    waveform_loss = waveform_loss.float()
    projection_loss = waveform_loss.new_zeros(())
    contribution = waveform_loss.new_zeros(())
    if projection:
        projection_loss = experiment._deranged_vocal_projection_loss(
            estimates=estimates,
            targets=targets[..., :estimates.shape[-1]],
            vocal_derangement=deranged,
        )
        weight = torch.minimum(
            projection_loss.new_tensor(experiment.DERANGED_PROJECTION_LOSS_WEIGHT),
            experiment.DERANGED_PROJECTION_MAX_L1_FRACTION
            * waveform_loss.detach() / projection_loss.detach().clamp_min(1e-8),
        )
        contribution = weight * projection_loss
    if deployed_l1_weight:
        # The vectorized forward removes causal padding: no sample shift is needed.
        length = estimates.shape[-1]
        gains = estimates.new_tensor(DEPLOYED_DBV_GAINS, dtype=torch.float32).view(1, 3, 1, 1)
        dbv = estimates[:, :3].float() * gains
        other = mixture[..., :length].float().unsqueeze(1) - dbv.sum(dim=1, keepdim=True)
        deployed_l1 = torch.nn.functional.l1_loss(
            torch.cat((dbv, other), dim=1), targets[..., :length].float(),
        )
        loss = (1.0 - deployed_l1_weight) * waveform_loss + deployed_l1_weight * deployed_l1 + contribution
    else:
        loss = waveform_loss + contribution
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite training loss")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
    optimizer.step()
    metrics = {
        "loss": float(loss.detach()),
        "waveform_l1": float(waveform_loss.detach()),
        "projection": float(projection_loss.detach()),
        "projection_contribution": float(contribution.detach()),
        "grad_norm": float(grad_norm),
        "deranged_examples": int(deranged.sum()),
    }
    if deployed_l1_weight:
        metrics["deployed_l1"] = float(deployed_l1.detach())
    return metrics


def save_checkpoint(run_dir, model, optimizer, step, config) -> None:
    # Keep the model used for optimization raw; finalize only a separate copy.
    payload = {
        "model": model.state_dict(), "config": model._config,
        "optimizer": optimizer.state_dict(), "step": step, "run_config": config,
        "rng": {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
        },
    }
    temporary = run_dir / "resume.pt.tmp"
    torch.save(payload, temporary)
    temporary.replace(run_dir / "resume.pt")
    save_deployment(model, run_dir / f"step-{step:06d}-deploy.pt")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--min-lr", type=float, default=3e-6)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--projection", choices=("off", "on"), default="off")
    parser.add_argument("--deployed-l1-weight", type=float, default=0.0,
                        help="L1 fraction on calibrated DBV and residual Other; 0 keeps raw L1 only.")
    parser.add_argument("--teacher-weight", type=float, default=0.25,
                        help="Teacher-root sampling probability; human roots retain their 2:1 ratio.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--data-start", type=int, default=50_000 * 16)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--stop-after", type=int, help="Pause at this run-local update; resume with the same command.")
    args = parser.parse_args(argv)
    if not (args.steps > 0 and args.batch_size > 1 and args.workers >= 0 and args.data_start >= 0):
        parser.error("Require positive steps, batch size > 1, and nonnegative workers/data start")
    if not (0 < args.min_lr <= args.lr and 0 <= args.warmup < args.steps and args.checkpoint_every > 0):
        parser.error("Require 0 < min-lr <= lr, 0 <= warmup < steps, checkpoint-every > 0")
    if args.stop_after is not None and not 0 < args.stop_after <= args.steps:
        parser.error("stop-after must be between 1 and steps")
    if not 0.0 <= args.deployed_l1_weight <= 1.0:
        parser.error("deployed-l1-weight must be between 0 and 1")
    if not 0.0 <= args.teacher_weight < 1.0:
        parser.error("teacher-weight must be at least 0 and less than 1")
    if not torch.cuda.is_available():
        raise RuntimeError("This continuation trainer requires CUDA with BF16 support")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    production.configure_determinism(args.seed)
    production_config = json.loads((PRODUCTION_ROOT / "full_config.json").read_text())
    manifest_path = PRODUCTION_ROOT / "manifests/combined.manifest.json"
    _, tracks, manifest_hash, _ = production.load_corpus_manifest(
        manifest_path, expected_file_sha256=None, config=production_config,
    )
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items()
              if key not in ("run_dir", "stop_after", "deployed_l1_weight", "teacher_weight")}
    config.update({
        "parent_sha256": file_hash(args.parent), "manifest_sha256": manifest_hash,
        "crop_samples": production_config["model"]["crop_samples"],
        "root_weights": production_config["sampling"]["root_weights"],
        "vocal_active_probability": production_config["sampling"]["vocal_active_probability"],
        "data_seed": production_config["seed"], "optimizer": "fresh Adam, default betas, no weight decay",
        "stem_subset_probability": experiment.STEM_SUBSET_AUGMENT_PROBABILITY,
        "vocal_derangement_probability": experiment.VOCAL_DERANGEMENT_PROBABILITY,
        "projection_weight": experiment.DERANGED_PROJECTION_LOSS_WEIGHT,
        "projection_max_l1_fraction": experiment.DERANGED_PROJECTION_MAX_L1_FRACTION,
        "torch_version": torch.__version__,
        "output_convention": "train raw four heads; evaluate calibrated Drums/Bass/Vocals and residual Other",
    })
    if args.teacher_weight != 0.25:
        original_weights = config["root_weights"]
        human_weight = sum(original_weights.values()) - original_weights[TEACHER_ROOT]
        config["root_weights"] = {
            root: args.teacher_weight if root == TEACHER_ROOT
            else weight * (1.0 - args.teacher_weight) / human_weight
            for root, weight in original_weights.items()
            if root != TEACHER_ROOT or args.teacher_weight > 0.0
        }
        tracks = [track for track in tracks if track.root_id in config["root_weights"]]
        config.update({"teacher_weight": args.teacher_weight, "track_count": len(tracks)})
        print(json.dumps({"sampling": {
            "root_weights": config["root_weights"], "track_count": len(tracks),
            "teacher_weight": args.teacher_weight,
        }}), flush=True)
    if args.deployed_l1_weight:
        config.update({
            "deployed_l1_weight": args.deployed_l1_weight,
            "deployed_l1_dbv_gains": list(DEPLOYED_DBV_GAINS),
            "output_convention": "train blended raw/deployed L1; deployed Other = augmented mixture - calibrated DBV sum",
        })
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.run_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Run configuration differs; use a new run directory")
    write_json(config_path, config)
    model = load_model(args.parent, device="cuda", raw=True).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    step = 0
    resume_path = args.run_dir / "resume.pt"
    if resume_path.exists():
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if payload["run_config"] != config:
            raise ValueError("Resume configuration differs")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        step = payload["step"]
        rng = payload["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state_all(rng["cuda"])
        del payload

    stop_step = args.stop_after or args.steps
    if step >= stop_step:
        print(json.dumps({"status": "already_at_requested_step", "step": step}), flush=True)
        return
    dataset = production.CounterAddressedCropDataset(
        tracks, root_weights=config["root_weights"], seed=config["data_seed"],
        crop_samples=config["crop_samples"],
        vocal_active_probability=config["vocal_active_probability"],
        final_sample_index=args.data_start + args.steps * args.batch_size,
    )
    loader_options = {}
    if args.workers:
        loader_options = {"multiprocessing_context": "spawn", "prefetch_factor": 2}
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=production.AbsoluteIndexSampler(
            args.data_start + step * args.batch_size,
            args.data_start + stop_step * args.batch_size,
        ),
        num_workers=args.workers, pin_memory=True, worker_init_fn=production.worker_init,
        generator=torch.Generator().manual_seed(args.seed + 1), **loader_options,
    )
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.monotonic()
    start_step = step
    torch.cuda.reset_peak_memory_stats()
    log_path = args.run_dir / "metrics.jsonl"
    write_json(args.run_dir / "status.json", {"status": "running", "step": step, "pid": os.getpid()})
    try:
        for mixture, targets in loader:
            lr = learning_rate(step, total=args.steps, peak=args.lr, floor=args.min_lr, warmup=args.warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr
            metrics = train_batch(
                model, optimizer, mixture.to("cuda", non_blocking=True), targets.to("cuda", non_blocking=True),
                projection=args.projection == "on",
                deployed_l1_weight=args.deployed_l1_weight,
            )
            step += 1
            metrics.update({"step": step, "lr": lr, "elapsed_seconds": time.monotonic() - started})
            with log_path.open("a") as handle:
                handle.write(json.dumps(metrics, allow_nan=False) + "\n")
            if step % 25 == 0 or step == start_step + 1:
                metrics["steps_per_second"] = (step - start_step) / (time.monotonic() - started)
                print(json.dumps(metrics), flush=True)
                write_json(args.run_dir / "status.json", {"status": "running", "pid": os.getpid(), **metrics})
            if step % args.checkpoint_every == 0 or step == stop_step or stop_requested:
                save_checkpoint(args.run_dir, model, optimizer, step, config)
            if stop_requested:
                break
    except Exception as error:
        write_json(args.run_dir / "status.json", {"status": "failed", "step": step, "error": repr(error)})
        raise
    status = {
        "status": "complete" if step == args.steps else "paused", "step": step,
        "elapsed_seconds": time.monotonic() - started,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    write_json(args.run_dir / "status.json", status)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
