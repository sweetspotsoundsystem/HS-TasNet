"""Cache the frozen C191 core, then train its existing correction stack on CPU.

Preparation reuses a complete head cache's augmented mixture, targets and
teacher estimates exactly. It does not read production audio, augment again,
or run the teacher. Training freezes the core and C184 router weights while
ordinary Adam updates C140, temporal, refiner, correction and head parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import time

import torch
from torch.nn import functional as F

from research.direct import latency_cache_head as head_cache
from research.direct import latency_train as training
from research.direct.latency11 import file_sha256, load_model, save_model
from research.direct.latency11_c191 import FUSION_SCALE, HEAD_PHASE_FEATURES_VERSION


CACHE_KIND = "c191_frozen_core_from_head_cache_v1"
PAIRED_FIELDS = ("mixture", "targets", "teacher_dbv")


def source_identities():
    identities = head_cache.source_identities()
    identities[str(Path(__file__).resolve())] = file_sha256(Path(__file__))
    return identities


def tensor_sha256(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def module_sha256(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor_sha256(value).encode("ascii"))
    return digest.hexdigest()


def read_source_cache(directory):
    config_path, index_path = directory / "config.json", directory / "index.json"
    config, index = json.loads(config_path.read_text()), json.loads(index_path.read_text())
    rows = head_cache.validate_index(index, config, config_path, require_complete=True)
    if config.get("source_identities") != head_cache.source_identities():
        raise ValueError("Source head-cache implementations have changed")
    if not config.get("teacher") or not config.get("teacher_sha256"):
        raise ValueError("Post preparation requires a head cache with paired teacher estimates")
    if config.get("parent_precision") != "float32" or config.get("parent_device") != "cpu":
        raise ValueError("Post preparation requires a CPU FP32 source cache")
    if file_sha256(Path(config["parent"])) != config["parent_sha256"]:
        raise ValueError("Source head-cache parent has changed")
    if not rows or config["batch_size"] != 2 or config["crop_samples"] < 4096 or config["crop_samples"] % 512:
        raise ValueError("Require nonempty paired B2 crops with hop-aligned length >= 4096")
    return config, rows, file_sha256(config_path), file_sha256(index_path)


def validate_batch(batch, config, *, include_core):
    batch_size, samples = config["batch_size"], config["crop_samples"]
    shapes = {"mixture": (batch_size, 2, samples),
              "targets": (batch_size, 4, 2, samples),
              "teacher_dbv": (batch_size, 3, 2, samples)}
    if include_core:
        shapes.update({"raw": (batch_size, 4, 2, samples),
                       "spec_dbv": (batch_size, 3, 2, samples),
                       "waveform_dbv": (batch_size, 3, 2, samples),
                       "final_fusion_hidden": (2, batch_size, 1000)})
    for name, shape in shapes.items():
        value = batch.get(name)
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.dtype != torch.float32 or value.device.type != "cpu"):
            raise ValueError(f"Incorrect cached {name}; expected CPU FP32 {shape}")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite cached {name}")


@torch.no_grad()
def capture_core(model, source_batch):
    audio = source_batch["mixture"]
    state = model.initial_state(audio.shape[0], device="cpu")
    raw, hiddens, components = model.core(
        torch.cat((state[0], audio), dim=-1),
        hiddens=(None, None, state[1] / FUSION_SCALE, None, None),
        auto_causal_pad=False, auto_curtail_length_to_multiple=False,
        is_streaming=True, return_streaming_components=True,
    )
    if len(components) != 2:
        raise ValueError("Expected spectral and waveform components in native order")
    batch = {name: source_batch[name].detach().clone() for name in PAIRED_FIELDS}
    batch.update({"raw": raw.detach().clone(),
                  "spec_dbv": components[0][:, :3].detach().clone(),
                  "waveform_dbv": components[1][:, :3].detach().clone(),
                  "final_fusion_hidden": hiddens[2].detach().clone()})
    return batch


def validate_post_index(index, config, config_path, source_rows, *, require_complete):
    rows = head_cache.validate_index(index, config, config_path, require_complete=require_complete)
    for row, source_row in zip(rows, source_rows):
        if (row.get("source_batch_index") != source_row["batch_index"]
                or row.get("source_file") != source_row["file"]
                or row.get("source_sha256") != source_row["sha256"]
                or set(row.get("paired_tensor_sha256", {})) != set(PAIRED_FIELDS)):
            raise ValueError("Post-cache index has lost its paired source-batch identity")
    return rows


def stop_flag():
    stopped = [False]

    def request_stop(signum, frame):
        stopped[0] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    return stopped


def prepare(args):
    if args.source_cache_dir.resolve() == args.cache_dir.resolve():
        raise ValueError("Post-cache directory must differ from its source head cache")
    source, source_rows, source_config_sha, source_index_sha = read_source_cache(args.source_cache_dir)
    model = load_model("c191", Path(source["parent"]), device="cpu").float().eval().requires_grad_(False)
    config = {"kind": CACHE_KIND, "schema_version": 1,
              "source_cache_dir": str(args.source_cache_dir.resolve()),
              "source_cache_config_sha256": source_config_sha,
              "source_cache_index_sha256": source_index_sha,
              "source_cache_config": source,
              "parent": source["parent"], "parent_sha256": source["parent_sha256"],
              "teacher": source["teacher"], "teacher_sha256": source["teacher_sha256"],
              "examples": source["examples"], "batch_size": source["batch_size"],
              "crop_samples": source["crop_samples"], "data_start": source["data_start"],
              "manifest_sha256": source["manifest_sha256"],
              "core_state_sha256": module_sha256(model.core),
              "router_state_sha256": module_sha256(model.engine.router),
              "source_identities": source_identities(), "threads": args.threads,
              "torch_version": str(torch.__version__), "precision": "float32", "device": "cpu",
              "component_order": ["spec_dbv", "waveform_dbv"],
              "component_gains": "Already applied by the frozen core; raw stored independently",
              "initial_state": "Fresh zero state independently for every crop",
              "pairing": "Exact source mixture/targets/teacher; no augmentation or teacher inference"}
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    config_path, index_path = args.cache_dir / "config.json", args.cache_dir / "index.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Post-cache configuration differs; use a new directory")
    training.write_json(config_path, config)
    rows = validate_post_index(json.loads(index_path.read_text()), config, config_path,
                               source_rows, require_complete=False) if index_path.exists() else []
    for row in rows:
        if file_sha256(args.cache_dir / row["file"]) != row["sha256"]:
            raise ValueError("Existing post-cache prefix has changed")
    if len(rows) == len(source_rows) or args.stop_after_batches and len(rows) >= args.stop_after_batches:
        print(json.dumps({"status": "already_at_requested_batch", "batches": len(rows)}), flush=True)
        return
    stopped, started = stop_flag(), time.monotonic()
    training.write_json(args.cache_dir / "status.json", {"status": "preparing", "batches": len(rows), "pid": os.getpid()})
    try:
        for source_row in source_rows[len(rows):]:
            source_path = args.source_cache_dir / source_row["file"]
            if file_sha256(source_path) != source_row["sha256"]:
                raise ValueError(f"Changed source head-cache batch: {source_row['file']}")
            source_batch = torch.load(source_path, map_location="cpu", weights_only=True)
            validate_batch(source_batch, config, include_core=False)
            batch = capture_core(model, source_batch)
            validate_batch(batch, config, include_core=True)
            paired_hashes = {name: tensor_sha256(batch[name]) for name in PAIRED_FIELDS}
            if any(not torch.equal(batch[name], source_batch[name]) for name in PAIRED_FIELDS):
                raise ValueError("A paired input changed during frozen-core capture")
            filename = source_row["file"]
            head_cache.atomic_save(batch, args.cache_dir / filename)
            row = {key: source_row[key] for key in ("batch_index", "first_absolute_data_index", "examples",
                   "deranged_examples", "silenced_examples", "toned_examples")}
            row.update({"file": filename, "sha256": file_sha256(args.cache_dir / filename),
                        "bytes": (args.cache_dir / filename).stat().st_size,
                        "source_batch_index": source_row["batch_index"], "source_file": filename,
                        "source_sha256": source_row["sha256"], "paired_tensor_sha256": paired_hashes})
            rows.append(row)
            training.write_json(index_path, {"config_sha256": file_sha256(config_path), "batches": rows,
                                "status": "complete" if len(rows) == len(source_rows) else "preparing"})
            status = {"status": "complete" if len(rows) == len(source_rows) else "preparing",
                      "batches": len(rows), "examples": len(rows) * config["batch_size"],
                      "elapsed_seconds": time.monotonic() - started, "pid": os.getpid()}
            training.write_json(args.cache_dir / "status.json", status)
            if len(rows) == 1 or len(rows) % 16 == 0:
                print(json.dumps(status), flush=True)
            if stopped[0] or args.stop_after_batches and len(rows) >= args.stop_after_batches:
                break
    except Exception as error:
        training.write_json(args.cache_dir / "status.json", {"status": "failed", "batches": len(rows), "error": repr(error)})
        raise
    status.update({"status": "complete" if len(rows) == len(source_rows) else "paused"})
    training.write_json(args.cache_dir / "status.json", status)
    print(json.dumps(status), flush=True)


def configure_scope(model):
    model.train().requires_grad_(True)
    model.core.eval().requires_grad_(False)
    model.engine.router.eval().requires_grad_(False)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def cached_raw(model, batch):
    state = model.initial_state(batch["mixture"].shape[0], device="cpu")
    raw, _ = model.engine._forward_corrections(
        batch["mixture"], state, batch["raw"],
        (None, None, batch["final_fusion_hidden"], None, None),
        (batch["spec_dbv"], batch["waveform_dbv"]), return_raw=True)
    return model._calibrate(batch["mixture"], raw, return_raw=True)


def cached_loss(model, batch, *, seam_weight, lowband_weight, distillation_weight):
    raw = cached_raw(model, batch)
    waveform = F.l1_loss(raw, batch["targets"])
    loss, metrics = waveform, {"waveform_l1": float(waveform.detach())}
    if seam_weight or lowband_weight:
        deployed = torch.cat((raw[:, :3], batch["mixture"][:, None] - raw[:, :3].sum(dim=1, keepdim=True)), dim=1)
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
        raise FloatingPointError("Non-finite post-cache training loss")
    metrics["loss"] = float(loss.detach())
    return loss, metrics


def check_frozen(model, config):
    for module, key in ((model.core, "core_state_sha256"), (model.engine.router, "router_state_sha256")):
        if any(parameter.grad is not None or parameter.requires_grad for parameter in module.parameters()):
            raise ValueError("Frozen core or router unexpectedly received gradients")
        if module_sha256(module) != config[key]:
            raise ValueError("Frozen core or router state changed")


def save_checkpoint(model, optimizer, step, config, run_dir):
    check_frozen(model, config)
    model.provenance["cached_post_training"] = {
        "step": step, "train_scope": config["train_scope"],
        "cache_config_sha256": config["cache_config_sha256"],
        "cache_index_sha256": config["cache_index_sha256"],
        "source_identities": config["source_identities"],
        "run_config_sha256": file_sha256(run_dir / "config.json"),
        "frozen_core_state_sha256": config["core_state_sha256"],
        "frozen_router_state_sha256": config["router_state_sha256"],
    }
    payload = {"latency11_kind": "c191", "model": model.engine.state_dict(),
               "config": model.engine.core_config, "vocal_gain": model.vocal_gain,
               "provenance": model.provenance, "optimizer": optimizer.state_dict(),
               "step": step, "run_config": config, "torch_rng": torch.get_rng_state()}
    if model.engine.head.phase_features:
        payload["c191_head_phase_features"] = HEAD_PHASE_FEATURES_VERSION
    save_model(model, run_dir / f"step-{step:06d}.pt")
    head_cache.atomic_save(payload, run_dir / "resume.pt")


def reconcile_metrics(run_dir, step):
    """Keep the committed metric prefix and preserve any interrupted replay tail."""
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        if step:
            raise ValueError("Resume checkpoint has no matching metric history")
        return
    lines = path.read_text().splitlines(keepends=True)
    if len(lines) < step:
        raise ValueError("Metric history is shorter than the resume checkpoint")
    for expected, line in enumerate(lines[:step], start=1):
        if json.loads(line).get("step") != expected:
            raise ValueError("Committed metric history is not a unique ordered step sequence")
    if len(lines) > step:
        with (run_dir / "metrics-uncommitted.jsonl").open("a") as handle:
            handle.write("".join(lines[step:]))
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(lines[:step]))
        temporary.replace(path)


def train(args):
    cache_config_path, cache_index_path = args.cache_dir / "config.json", args.cache_dir / "index.json"
    cache_config, cache_index = json.loads(cache_config_path.read_text()), json.loads(cache_index_path.read_text())
    if cache_config.get("kind") != CACHE_KIND or cache_config.get("source_identities") != source_identities():
        raise ValueError("Post-cache schema or source implementations have changed")
    source, source_rows, source_config_sha, source_index_sha = read_source_cache(Path(cache_config["source_cache_dir"]))
    if (source_config_sha != cache_config["source_cache_config_sha256"]
            or source_index_sha != cache_config["source_cache_index_sha256"]
            or source != cache_config["source_cache_config"]):
        raise ValueError("Paired source head-cache metadata changed")
    rows = validate_post_index(cache_index, cache_config, cache_config_path, source_rows, require_complete=True)
    for row in rows:
        if file_sha256(args.cache_dir / row["file"]) != row["sha256"]:
            raise ValueError(f"Changed post-cache batch: {row['file']}")
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items() if key not in ("run_dir", "stop_after", "command")}
    config.update({"schema_version": 1, "cache_config_sha256": file_sha256(cache_config_path),
                   "cache_index_sha256": file_sha256(cache_index_path),
                   "parent_sha256": cache_config["parent_sha256"],
                   "core_state_sha256": cache_config["core_state_sha256"],
                   "router_state_sha256": cache_config["router_state_sha256"],
                   "teacher_sha256": cache_config["teacher_sha256"],
                   "source_identities": source_identities(), "torch_version": str(torch.__version__),
                   "train_scope": "c191-post-except-core-and-router", "precision": "float32", "device": "cpu",
                   "optimizer": "fresh Adam, default betas, no weight decay; clip norm 5",
                   "schedule": "randperm(batch_count, CPU generator seed + epoch); step determines epoch and offset",
                   "seam_radius": 0, "zero_initial_state_each_crop": True,
                   "loss_denominators": "Full mean over raw four stems; deployed seam/lowband four stems; teacher DBV three stems",
                   "online_augmentation": False})
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.run_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Post training configuration differs; use a new directory")
    training.write_json(config_path, config)
    model = load_model("c191", Path(cache_config["parent"]), device="cpu").float()
    parameters = configure_scope(model)
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    resume_path, step = args.run_dir / "resume.pt", 0
    if resume_path.exists():
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if payload["run_config"] != config or not 0 <= payload["step"] <= args.steps:
            raise ValueError("Resume training configuration or step differs")
        model.engine.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        model.provenance = payload["provenance"]
        torch.set_rng_state(payload["torch_rng"])
        step = payload["step"]
    check_frozen(model, config)
    reconcile_metrics(args.run_dir, step)
    stop_step = args.stop_after or args.steps
    if step >= stop_step:
        print(json.dumps({"status": "already_at_requested_step", "step": step}), flush=True)
        return
    stopped, started, start_step, epoch, order = stop_flag(), time.monotonic(), step, -1, None
    training.write_json(args.run_dir / "status.json", {"status": "running", "step": step, "pid": os.getpid()})
    print(json.dumps({"trainable_parameters": sum(parameter.numel() for parameter in parameters),
                      "cache_examples": cache_config["examples"], "start_step": step}), flush=True)
    try:
        while step < stop_step:
            this_epoch = step // len(rows)
            if this_epoch != epoch:
                epoch = this_epoch
                order = torch.randperm(len(rows), generator=torch.Generator(device="cpu").manual_seed(args.seed + epoch)).tolist()
            row = rows[order[step % len(rows)]]
            batch = torch.load(args.cache_dir / row["file"], map_location="cpu", weights_only=True)
            validate_batch(batch, cache_config, include_core=True)
            if any(tensor_sha256(batch[name]) != row["paired_tensor_sha256"][name] for name in PAIRED_FIELDS):
                raise ValueError("Post training inputs differ from their paired source hashes")
            lr = training.learning_rate(step, total=args.steps, peak=args.lr, floor=args.min_lr, warmup=args.warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = cached_loss(model, batch, seam_weight=args.seam_weight,
                                       lowband_weight=args.lowband_weight, distillation_weight=args.distillation_weight)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            metrics.update({"step": step, "epoch": epoch, "cache_batch": row["batch_index"],
                            "source_batch_sha256": row["source_sha256"],
                            "lr": lr, "grad_norm": float(grad_norm), "elapsed_seconds": time.monotonic() - started})
            with (args.run_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(metrics, allow_nan=False) + "\n")
            if step == start_step + 1 or step % 25 == 0:
                print(json.dumps(metrics), flush=True)
                training.write_json(args.run_dir / "status.json", {"status": "running", "pid": os.getpid(), **metrics})
            if step % args.checkpoint_every == 0 or step == stop_step or stopped[0]:
                save_checkpoint(model, optimizer, step, config, args.run_dir)
            if stopped[0]:
                break
    except Exception as error:
        training.write_json(args.run_dir / "status.json", {"status": "failed", "step": step, "error": repr(error)})
        raise
    check_frozen(model, config)
    status = {"status": "complete" if step == args.steps else "paused", "step": step,
              "elapsed_seconds": time.monotonic() - started,
              "peak_process_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
              "peak_process_rss_scope": "Linux current process lifetime", "core_unchanged": True, "router_unchanged": True}
    training.write_json(args.run_dir / "status.json", status)
    print(json.dumps(status), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Reuse paired head-cache inputs and capture native frozen-core outputs")
    prep.add_argument("--source-cache-dir", type=Path, required=True)
    prep.add_argument("--stop-after-batches", type=int)
    run = commands.add_parser("train", help="Train C191 post modules with core and router weights frozen")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--steps", type=int, default=2000)
    run.add_argument("--lr", type=float, default=1e-4)
    run.add_argument("--min-lr", type=float, default=1e-5)
    run.add_argument("--warmup", type=int, default=100)
    run.add_argument("--seam-weight", type=float, default=.1)
    run.add_argument("--lowband-weight", type=float, default=.1)
    run.add_argument("--distillation-weight", type=float, default=.25)
    run.add_argument("--checkpoint-every", type=int, default=500)
    run.add_argument("--stop-after", type=int)
    run.add_argument("--seed", type=int, default=20260905)
    for child in (prep, run):
        child.add_argument("--cache-dir", type=Path, required=True)
        child.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.command == "prepare":
        if args.stop_after_batches is not None and args.stop_after_batches < 1:
            parser.error("stop-after-batches must be positive")
    else:
        if not (args.seed >= 0 and math.isfinite(args.lr) and 0 < args.min_lr <= args.lr
                and 0 <= args.warmup < args.steps and args.checkpoint_every > 0):
            parser.error("Require nonnegative seed, finite 0 < min-lr <= lr, 0 <= warmup < steps, positive checkpoint interval")
        if any(not math.isfinite(value) or value < 0 for value in (args.seam_weight, args.lowband_weight, args.distillation_weight)):
            parser.error("Loss weights must be finite and nonnegative")
        if args.stop_after is not None and not 0 < args.stop_after <= args.steps:
            parser.error("stop-after must be between 1 and steps")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    head_cache.set_cpu_seed(getattr(args, "seed", 20260905))
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    (prepare if args.command == "prepare" else train)(args)


if __name__ == "__main__":
    main()
