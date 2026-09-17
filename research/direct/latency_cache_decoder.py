"""Train C191 decoder/post or post alone on paired cached crops in FP32.

Only mixture, targets and frozen teacher estimates are read from a complete
head cache. Every model output is recomputed from a fresh zero crop state.
By default ordinary Adam updates both mask estimators, the baked decoder weight
and the existing post stack on CPU. Post-only scope freezes the complete core.
The router, all buffers and native vocal calibration remain fixed in both scopes.
Version 2 adds explicit device/scope identity; existing v1 run configs cannot resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time

import torch
from torch.nn import functional as F

from research.direct import latency_cache_head as head_cache
from research.direct import latency_cache_post as post_cache
from research.direct import latency_train as training
from research.direct.latency11 import file_sha256, load_model, save_model
from research.direct.latency11_c191 import HEAD_PHASE_FEATURES_VERSION


TRAIN_SCOPES = {
    "decoder-post": "c191-mask-estimators-decoder-weight-and-post-except-router",
    "post": "c191-post-except-router",
}
POST_MODULES = ("c140", "temporal", "refiner", "correction", "head")


def source_identities():
    identities = post_cache.source_identities()
    identities[str(Path(__file__).resolve())] = file_sha256(Path(__file__))
    return identities


def configure_scope(model, scope="decoder-post"):
    if scope not in TRAIN_SCOPES:
        raise ValueError(f"Unknown training scope: {scope}")
    model.train().requires_grad_(False)
    model.core.eval()
    model.engine.router.eval()
    for name in POST_MODULES:
        getattr(model.engine, name).train().requires_grad_(True)
    if scope == "decoder-post":
        model.core.to_spec_masks.train().requires_grad_(True)
        model.core.to_waveform_masks.train().requires_grad_(True)
        model.core.conv_decode.weight.requires_grad_(True)
        # Both functional decoder paths omit the inherited ConvTranspose1d bias.
        model.core.conv_decode.bias.requires_grad_(False)
        if not model.core.conv_decode.hann_window_baked:
            raise ValueError("Decoder training requires the retained baked filters")
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def frozen_state_sha256(model):
    digest = hashlib.sha256()
    frozen = [(name, value) for name, value in model.named_parameters() if not value.requires_grad]
    for name, value in (*frozen, *model.named_buffers()):
        digest.update(name.encode("utf-8"))
        digest.update(post_cache.tensor_sha256(value).encode("ascii"))
    return digest.hexdigest()


def check_frozen(model, config):
    active = [name for name, value in model.named_parameters() if value.requires_grad]
    if active != config["trainable_names"]:
        raise ValueError("Decoder training parameter scope changed")
    if any(value.grad is not None for value in model.parameters() if not value.requires_grad):
        raise ValueError("A frozen parameter received gradients")
    if any(value.requires_grad or value.grad is not None for value in model.buffers()):
        raise ValueError("A frozen buffer received gradients")
    if frozen_state_sha256(model) != config["frozen_state_sha256"]:
        raise ValueError("A frozen parameter or buffer changed")
    if (model.vocal_gain != config["vocal_gain"]
            or model._vocal_output_ratio != config["vocal_output_ratio"]):
        raise ValueError("Native vocal calibration changed")


def paired_batch(path, config, device="cpu"):
    source = torch.load(path, map_location="cpu", weights_only=True)
    batch = {name: source[name] for name in post_cache.PAIRED_FIELDS}
    post_cache.validate_batch(batch, config, include_core=False)
    return {name: value.to(device) for name, value in batch.items()}


def batch_loss(model, batch, *, seam_weight, lowband_weight, distillation_weight):
    raw, _ = model.forward_raw(batch["mixture"])
    waveform = F.l1_loss(raw, batch["targets"])
    loss, metrics = waveform, {"waveform_l1": float(waveform.detach())}
    if seam_weight or lowband_weight:
        deployed = torch.cat((raw[:, :3], batch["mixture"][:, None]
                              - raw[:, :3].sum(dim=1, keepdim=True)), dim=1)
    if seam_weight:
        seam = training.seam_error_l1(deployed, batch["targets"], radius=0)
        loss = loss + seam_weight * seam
        metrics["seam_error_l1"] = float(seam.detach())
    if lowband_weight:
        lowband = training.lowband_error_l1(deployed, batch["targets"])
        loss = loss + lowband_weight * lowband
        metrics["lowband_l1"] = float(lowband.detach())
    if distillation_weight:
        teacher = F.l1_loss(raw[:, :3], batch["teacher_dbv"])
        loss = loss + distillation_weight * teacher
        metrics["distillation_dbv_l1"] = float(teacher.detach())
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite decoder training loss")
    metrics["loss"] = float(loss.detach())
    return loss, metrics


def save_checkpoint(model, optimizer, step, config, run_dir):
    check_frozen(model, config)
    model.provenance["cached_decoder_training"] = {
        "step": step, "train_scope": config["train_scope"],
        "device": config["device"], "precision": config["precision"],
        "cache_config_sha256": config["cache_config_sha256"],
        "cache_index_sha256": config["cache_index_sha256"],
        "source_identities": config["source_identities"],
        "run_config_sha256": file_sha256(run_dir / "config.json"),
        "frozen_state_sha256": config["frozen_state_sha256"],
    }
    payload = {"latency11_kind": "c191", "model": model.engine.state_dict(),
               "config": model.engine.core_config, "vocal_gain": model.vocal_gain,
               "provenance": model.provenance, "optimizer": optimizer.state_dict(),
               "step": step, "run_config": config, "torch_rng": torch.get_rng_state(),
               "cuda_rng": torch.cuda.get_rng_state() if config["device"] == "cuda" else None}
    if model.engine.head.phase_features:
        payload["c191_head_phase_features"] = HEAD_PHASE_FEATURES_VERSION
    save_model(model, run_dir / f"step-{step:06d}.pt")
    head_cache.atomic_save(payload, run_dir / "resume.pt")


def train(args):
    stopped = post_cache.stop_flag()
    source, rows, config_sha, index_sha = post_cache.read_source_cache(args.cache_dir)
    # Authenticate the complete input set before the first optimizer update.
    for row in rows:
        if file_sha256(args.cache_dir / row["file"]) != row["sha256"]:
            raise ValueError(f"Changed source cache batch: {row['file']}")
    model = load_model("c191", Path(source["parent"]), device="cpu").float().to(args.device)
    parameters = configure_scope(model, args.scope)
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items() if key not in ("run_dir", "stop_after", "command")}
    config.update({"schema_version": 2, "cache_config_sha256": config_sha,
                   "cache_index_sha256": index_sha, "parent_sha256": source["parent_sha256"],
                   "teacher_sha256": source["teacher_sha256"], "source_identities": source_identities(),
                   "train_scope": TRAIN_SCOPES[args.scope], "precision": "float32",
                   "torch_version": str(torch.__version__), "vocal_gain": model.vocal_gain,
                   "vocal_output_ratio": model._vocal_output_ratio,
                   "trainable_names": [name for name, value in model.named_parameters() if value.requires_grad],
                   "trainable_parameters": sum(value.numel() for value in parameters),
                   "frozen_state_sha256": frozen_state_sha256(model),
                   "optimizer": "fresh Adam, default betas, no weight decay; clip norm 5",
                   "schedule": "randperm(batch_count, CPU generator seed + epoch); step determines epoch and offset",
                   "seam_radius": 0, "zero_initial_state_each_crop": True,
                   "loss_denominators": "Full mean over raw four stems; deployed seam/lowband four stems; teacher DBV three stems",
                   "cache_fields_used": list(post_cache.PAIRED_FIELDS), "online_augmentation": False,
                   "model_outputs": "Recomputed every update from paired cached inputs"})
    if args.device == "cuda":
        config["cuda"] = {"torch_cuda_version": torch.version.cuda,
                          "device_name": torch.cuda.get_device_name(),
                          "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                          "cudnn_deterministic": torch.backends.cudnn.deterministic,
                          "cudnn_benchmark": torch.backends.cudnn.benchmark,
                          "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                          "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.run_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Decoder training configuration differs; use a new directory")
    training.write_json(config_path, config)
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    resume_path, step = args.run_dir / "resume.pt", 0
    if resume_path.exists():
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        expected_phase = HEAD_PHASE_FEATURES_VERSION if model.engine.head.phase_features else None
        if (payload["run_config"] != config or not 0 <= payload["step"] <= args.steps
                or payload["config"] != model.engine.core_config or payload["vocal_gain"] != model.vocal_gain
                or payload.get("c191_head_phase_features") != expected_phase
                or (isinstance(payload.get("cuda_rng"), torch.Tensor) != (args.device == "cuda"))):
            raise ValueError("Resume configuration, calibration, phase features or step differs")
        model.engine.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        model.provenance = payload["provenance"]
        torch.set_rng_state(payload["torch_rng"])
        if args.device == "cuda":
            torch.cuda.set_rng_state(payload["cuda_rng"])
        step = payload["step"]
    check_frozen(model, config)
    post_cache.reconcile_metrics(args.run_dir, step)
    stop_step = args.stop_after or args.steps
    if step >= stop_step:
        print(json.dumps({"status": "already_at_requested_step", "step": step}), flush=True)
        return
    started, start_step, epoch, order = time.monotonic(), step, -1, None
    training.write_json(args.run_dir / "status.json", {"status": "running", "step": step, "pid": os.getpid()})
    print(json.dumps({"trainable_parameters": config["trainable_parameters"],
                      "cache_examples": source["examples"], "start_step": step}), flush=True)
    try:
        while step < stop_step and not stopped[0]:
            this_epoch = step // len(rows)
            if this_epoch != epoch:
                epoch = this_epoch
                order = torch.randperm(len(rows), generator=torch.Generator(device="cpu").manual_seed(args.seed + epoch)).tolist()
            row = rows[order[step % len(rows)]]
            batch = paired_batch(args.cache_dir / row["file"], source, args.device)
            lr = training.learning_rate(step, total=args.steps, peak=args.lr, floor=args.min_lr, warmup=args.warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = batch_loss(model, batch, seam_weight=args.seam_weight,
                                       lowband_weight=args.lowband_weight, distillation_weight=args.distillation_weight)
            loss.backward()
            if any(value.grad is None for value in parameters):
                raise ValueError("An active decoder/post parameter received no gradient")
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            metrics.update({"step": step, "epoch": epoch, "cache_batch": row["batch_index"],
                            "source_batch_sha256": row["sha256"], "lr": lr,
                            "grad_norm": float(grad_norm), "elapsed_seconds": time.monotonic() - started})
            with (args.run_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(metrics, allow_nan=False) + "\n")
            training.write_json(args.run_dir / "status.json", {"status": "running", "pid": os.getpid(), **metrics})
            if step == start_step + 1 or step % 25 == 0:
                print(json.dumps(metrics), flush=True)
            if step % args.checkpoint_every == 0 or step == stop_step or stopped[0]:
                save_checkpoint(model, optimizer, step, config, args.run_dir)
        if stopped[0]:
            save_checkpoint(model, optimizer, step, config, args.run_dir)
        check_frozen(model, config)
    except Exception as error:
        training.write_json(args.run_dir / "status.json", {"status": "failed", "step": step, "error": repr(error)})
        raise
    status = {"status": "complete" if step == args.steps else "paused", "step": step,
              "elapsed_seconds": time.monotonic() - started,
              "peak_process_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
              "peak_process_rss_scope": "Linux current process lifetime",
              "frozen_parameters_unchanged": True, "all_buffers_unchanged": True}
    if args.device == "cuda":
        status["peak_cuda_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        status["peak_cuda_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    training.write_json(args.run_dir / "status.json", status)
    print(json.dumps(status), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("train", help="Train masks, decoder weight and corrections using paired cached inputs")
    run.add_argument("--cache-dir", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--scope", choices=tuple(TRAIN_SCOPES), default="decoder-post")
    run.add_argument("--steps", type=int, default=2000)
    run.add_argument("--lr", type=float, default=3e-5)
    run.add_argument("--min-lr", type=float, default=3e-6)
    run.add_argument("--warmup", type=int, default=100)
    run.add_argument("--seam-weight", type=float, default=.1)
    run.add_argument("--lowband-weight", type=float, default=.1)
    run.add_argument("--distillation-weight", type=float, default=.25)
    run.add_argument("--checkpoint-every", type=int, default=500)
    run.add_argument("--stop-after", type=int)
    run.add_argument("--seed", type=int, default=20260905)
    run.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if not (args.threads > 0 and args.seed >= 0 and math.isfinite(args.lr)
            and 0 < args.min_lr <= args.lr and 0 <= args.warmup < args.steps and args.checkpoint_every > 0):
        parser.error("Require positive threads/checkpoint interval, nonnegative seed, finite 0 < min-lr <= lr and 0 <= warmup < steps")
    if any(not math.isfinite(value) or value < 0 for value in (args.seam_weight, args.lowband_weight, args.distillation_weight)):
        parser.error("Loss weights must be finite and nonnegative")
    if args.stop_after is not None and not 0 < args.stop_after <= args.steps:
        parser.error("stop-after must be between 1 and steps")
    if args.device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    head_cache.set_cpu_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    train(args)


if __name__ == "__main__":
    main()
