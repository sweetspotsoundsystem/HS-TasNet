"""Prepare frozen C191 features, then train its small final head on CPU.

This is ordinary Adam training on a finite, explicitly recorded training cache.
Cache examples are newly sampled production crops; validation audio and the
diagnostic tone probes are never cache inputs. Both compared heads consume the
same augmented waveforms, parent features, targets and frozen C91 teacher.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import time

import torch
from torch.utils.data import DataLoader

from research import experiment
from research.direct import latency_train as training
from research.direct.checkpoints import load_model as load_teacher
from research.direct.latency11 import file_sha256, load_model, save_model
from research.direct.latency11_c191 import _ExactAdd


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def set_cpu_seed(seed):
    torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())


def capture_batch(model, mixture, targets, teacher=None):
    """Save the original pre-head history and independent frozen output tail."""
    captured = []

    def capture(module, inputs):
        captured.append(inputs[0].detach().clone())

    handle = model.engine.head.register_forward_pre_hook(capture)
    try:
        raw, _ = model.forward_raw(mixture)
    finally:
        handle.remove()
    if len(captured) != 1 or captured[0].shape != (
            mixture.shape[0], 4, mixture.shape[-1] + 2048):
        raise RuntimeError("Expected one unmodulated pre-head sequence with 2048 samples of history")
    batch = {"dense_db": captured[0], "tail": raw[:, 2:].detach().clone(),
             "mixture": mixture.detach().clone(), "targets": targets.detach().clone()}
    if teacher is not None:
        batch["teacher_dbv"] = training.aligned_teacher_dbv(teacher, mixture).clone()
    if any(not torch.isfinite(value).all() for value in batch.values()):
        raise FloatingPointError("Non-finite frozen training cache")
    return batch


def source_identities():
    """Bind cached parent features and objectives to their actual implementations."""
    directory = Path(__file__).parent
    paths = [directory / name for name in (
        "latency11.py", "latency11_c191.py", "causal_core.py",
        "latency_train.py", "latency_augment.py", "checkpoints.py", "latency_cache_head.py")]
    paths.extend((Path(experiment.__file__), Path(training.production.__file__),
                  directory.parents[1] / "hs_tasnet" / "hs_tasnet.py"))
    return {str(path.resolve()): file_sha256(path) for path in paths}


def validate_index(index, config, config_path, *, require_complete):
    if index.get("config_sha256") != file_sha256(config_path):
        raise ValueError("Cache index belongs to a different configuration")
    rows = index.get("batches", [])
    expected = config["examples"] // config["batch_size"]
    if len(rows) > expected or require_complete and (
            index.get("status") != "complete" or len(rows) != expected):
        raise ValueError("Cache index has an incorrect batch count or completion state")
    for number, row in enumerate(rows):
        if (row.get("batch_index") != number or row.get("file") != f"batch-{number:06d}.pt"
                or row.get("examples") != config["batch_size"]
                or row.get("first_absolute_data_index") != config["data_start"] + number * config["batch_size"]):
            raise ValueError("Cache index is not the expected unique, ordered example sequence")
    return rows


def cached_raw(head, batch):
    samples = batch["mixture"].shape[-1]
    delta = head(batch["dense_db"], samples)
    db = _ExactAdd.apply(batch["dense_db"][..., -samples:], delta)
    return torch.cat((db.reshape(-1, 2, 2, samples), batch["tail"]), dim=1)


def cached_loss(head, batch, *, seam_weight, lowband_weight, distillation_weight):
    raw = cached_raw(head, batch)
    targets = batch["targets"]
    waveform = torch.nn.functional.l1_loss(raw, targets)
    loss = waveform
    metrics = {"waveform_l1": float(waveform.detach())}
    if seam_weight or lowband_weight:
        other = batch["mixture"][:, None] - raw[:, :3].sum(dim=1, keepdim=True)
        deployed = torch.cat((raw[:, :3], other), dim=1)
    if seam_weight:
        seam = training.seam_error_l1(deployed, targets, radius=0)
        loss = loss + seam_weight * seam
        metrics["seam_error_l1"] = float(seam.detach())
    if lowband_weight:
        lowband = training.lowband_error_l1(deployed, targets)
        loss = loss + lowband_weight * lowband
        metrics["lowband_l1"] = float(lowband.detach())
    if distillation_weight:
        teacher = torch.nn.functional.l1_loss(raw[:, :3], batch["teacher_dbv"])
        loss = loss + distillation_weight * teacher
        metrics["distillation_dbv_l1"] = float(teacher.detach())
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite cached training loss")
    metrics["loss"] = float(loss.detach())
    return loss, metrics


def prepare(args):
    production = training.production
    production_config = json.loads((training.PRODUCTION_ROOT / "full_config.json").read_text())
    manifest = training.PRODUCTION_ROOT / "manifests/combined.manifest.json"
    _, tracks, manifest_hash, _ = production.load_corpus_manifest(
        manifest, expected_file_sha256=None, config=production_config)
    config = {
        "schema_version": 1, "parent": str(args.parent.resolve()),
        "parent_sha256": file_sha256(args.parent),
        "teacher": str(args.teacher.resolve()) if args.teacher else None,
        "teacher_sha256": file_sha256(args.teacher) if args.teacher else None,
        "examples": args.examples, "batch_size": args.batch_size,
        "seed": args.seed, "data_start": args.data_start,
        "data_seed": production_config["seed"], "manifest_sha256": manifest_hash,
        "crop_samples": production_config["model"]["crop_samples"],
        "root_weights": production_config["sampling"]["root_weights"],
        "vocal_active_probability": production_config["sampling"]["vocal_active_probability"],
        "silence_probability": args.silence_probability,
        "bass_tone_probability": args.bass_tone_probability,
        "remix": "production subset and vocal derangement before added tones and silence",
        "augmentation_rng": "CPU seed + 104729 * cache batch index, independent per batch",
        "tone_rng": "dedicated CPU generator: seed + first absolute data index",
        "parent_precision": "float32", "parent_device": "cpu",
        "source_identities": source_identities(),
        "head_input": "unmodulated [B,4,2048+T] C188 D/B history and current output",
        "initial_state": "zero independently for each production crop",
        "teacher_alignment": "explicit C91 causal padding; current-sample indices; frozen baked gains",
        "torch_version": torch.__version__, "threads": args.threads,
        "source_sha256": file_sha256(Path(__file__)),
        "augmentation_source_sha256": file_sha256(Path(training.__file__).with_name("latency_augment.py")),
    }
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.cache_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Cache configuration differs; use a new directory")
    training.write_json(config_path, config)
    index_path = args.cache_dir / "index.json"
    rows = validate_index(json.loads(index_path.read_text()), config, config_path,
                          require_complete=False) if index_path.exists() else []
    for index, row in enumerate(rows):
        path = args.cache_dir / row["file"]
        if row["batch_index"] != index or file_sha256(path) != row["sha256"]:
            raise ValueError("Existing cache prefix is incomplete or changed")
    total_batches = args.examples // args.batch_size
    if len(rows) == total_batches:
        print(json.dumps({"status": "already_complete", "examples": args.examples}), flush=True)
        return
    if args.stop_after_batches is not None and len(rows) >= args.stop_after_batches:
        print(json.dumps({"status": "already_at_requested_batch", "batches": len(rows)}), flush=True)
        return
    model = load_model("c191", args.parent, device="cpu").eval().requires_grad_(False)
    teacher = load_teacher(args.teacher, device="cpu", raw=False).float().eval().requires_grad_(False) \
        if args.teacher else None
    dataset = production.CounterAddressedCropDataset(
        tracks, root_weights=config["root_weights"], seed=config["data_seed"],
        crop_samples=config["crop_samples"],
        vocal_active_probability=config["vocal_active_probability"],
        final_sample_index=args.data_start + args.examples)
    options = {"multiprocessing_context": "spawn", "prefetch_factor": 2} if args.workers else {}
    loader = DataLoader(dataset, batch_size=args.batch_size,
        sampler=production.AbsoluteIndexSampler(args.data_start + len(rows) * args.batch_size,
                                               args.data_start + args.examples),
        num_workers=args.workers, worker_init_fn=production.worker_init,
        generator=torch.Generator().manual_seed(args.seed + 1), **options)
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started, first = time.monotonic(), len(rows)
    for mixture, targets in loader:
        index = len(rows)
        set_cpu_seed(args.seed + 104729 * index)
        mixture, targets, deranged = experiment._augment_training_distribution(
            mixture=mixture, targets=targets)
        toned = torch.zeros(args.batch_size, dtype=torch.bool)
        if args.bass_tone_probability:
            mixture, targets, toned = training.add_bass_tones(mixture, targets,
                args.bass_tone_probability, args.seed + args.data_start + index * args.batch_size)
        silenced = torch.zeros_like(toned)
        if args.silence_probability:
            mixture, targets, deranged, silenced = training.apply_silence(
                mixture, targets, deranged, args.silence_probability)
        batch = capture_batch(model, mixture, targets, teacher)
        filename = f"batch-{index:06d}.pt"
        atomic_save(batch, args.cache_dir / filename)
        row = {"batch_index": index, "file": filename,
               "sha256": file_sha256(args.cache_dir / filename),
               "first_absolute_data_index": args.data_start + index * args.batch_size,
               "examples": args.batch_size, "deranged_examples": int(deranged.sum()),
               "silenced_examples": int(silenced.sum()),
               "toned_examples": int((toned & ~silenced).sum()),
               "bytes": (args.cache_dir / filename).stat().st_size}
        rows.append(row)
        report = {"config_sha256": file_sha256(config_path), "batches": rows,
                  "status": "complete" if len(rows) == total_batches else "preparing"}
        training.write_json(index_path, report)
        if len(rows) % 16 == 0 or len(rows) == first + 1:
            print(json.dumps({"event": "cache_batch", "batches": len(rows),
                "examples": len(rows) * args.batch_size,
                "elapsed_seconds": time.monotonic() - started,
                "batches_per_second": (len(rows) - first) / (time.monotonic() - started)}), flush=True)
        if stop or args.stop_after_batches and len(rows) >= args.stop_after_batches:
            break
    print(json.dumps({"status": "complete" if len(rows) == total_batches else "paused",
                      "batches": len(rows), "elapsed_seconds": time.monotonic() - started}), flush=True)


def train(args):
    cache_config_path, cache_index_path = args.cache_dir / "config.json", args.cache_dir / "index.json"
    cache_config = json.loads(cache_config_path.read_text())
    cache_index = json.loads(cache_index_path.read_text())
    validate_index(cache_index, cache_config, cache_config_path, require_complete=True)
    if cache_config["source_identities"] != source_identities():
        raise ValueError("Frozen parent or supervision implementation differs from cached features")
    if args.distillation_weight and not cache_config["teacher"]:
        raise ValueError("Distillation requires cached teacher estimates")
    parent = Path(cache_config["parent"])
    if file_sha256(parent) != cache_config["parent_sha256"]:
        raise ValueError("Frozen parent differs from cached features")
    # Verify each cached file once, before any optimizer update.
    for row in cache_index["batches"]:
        if file_sha256(args.cache_dir / row["file"]) != row["sha256"]:
            raise ValueError(f"Changed cache batch: {row['file']}")
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items() if key not in ("run_dir", "stop_after", "command")}
    config.update({"schema_version": 1, "cache_config_sha256": file_sha256(cache_config_path),
        "cache_index_sha256": file_sha256(cache_index_path),
        "parent_sha256": cache_config["parent_sha256"],
        "optimizer": "fresh Adam, default betas, no weight decay, clip norm 5",
        "precision": "float32", "device": "cpu", "train_scope": "c191-head",
        "schedule": "seeded complete-cache permutation per epoch; no online augmentation",
        "seam_radius": 0, "source_sha256": file_sha256(Path(__file__)),
        "torch_version": torch.__version__})
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.run_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Training configuration differs; use a new directory")
    training.write_json(config_path, config)
    model = load_model("c191", parent, device="cpu").eval().requires_grad_(False)
    if model.engine.head.phase_features and not args.phase_features:
        raise ValueError("A phase-enabled parent requires explicit --phase-features")
    if args.phase_features:
        model.engine.head.enable_phase_features()
    head = model.engine.head.train().requires_grad_(True)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    resume_path, step = args.run_dir / "resume.pt", 0
    if resume_path.exists():
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if payload["run_config"] != config:
            raise ValueError("Resume configuration differs")
        head.load_state_dict(payload["head"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        step = payload["step"]
    stop_step = args.stop_after or args.steps
    if step >= stop_step:
        print(json.dumps({"status": "already_at_requested_step", "step": step}), flush=True)
        return
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started, start_step, epoch, order = time.monotonic(), step, -1, None
    batch_count = len(cache_index["batches"])
    training.write_json(args.run_dir / "status.json", {"status": "running", "step": step, "pid": os.getpid()})
    print(json.dumps({"trainable_parameters": sum(p.numel() for p in head.parameters()),
                      "cache_examples": cache_config["examples"], "start_step": step}), flush=True)
    try:
        while step < stop_step:
            this_epoch = step // batch_count
            if this_epoch != epoch:
                epoch = this_epoch
                order = torch.randperm(batch_count, generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
            row = cache_index["batches"][order[step % batch_count]]
            batch = torch.load(args.cache_dir / row["file"], map_location="cpu", weights_only=True)
            lr = training.learning_rate(step, total=args.steps, peak=args.lr, floor=args.min_lr, warmup=args.warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = cached_loss(head, batch, seam_weight=args.seam_weight,
                lowband_weight=args.lowband_weight, distillation_weight=args.distillation_weight)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            metrics.update({"step": step, "epoch": epoch, "cache_batch": row["batch_index"],
                "lr": lr, "grad_norm": float(grad_norm), "elapsed_seconds": time.monotonic() - started})
            with (args.run_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(metrics, allow_nan=False) + "\n")
            if step % 25 == 0 or step == start_step + 1:
                print(json.dumps(metrics), flush=True)
                training.write_json(args.run_dir / "status.json", {"status": "running", "pid": os.getpid(), **metrics})
            if step % args.checkpoint_every == 0 or step == stop_step or stop:
                atomic_save({"head": head.state_dict(), "optimizer": optimizer.state_dict(),
                             "step": step, "run_config": config}, resume_path)
                save_model(model, args.run_dir / f"step-{step:06d}.pt")
            if stop:
                break
    except Exception as error:
        training.write_json(args.run_dir / "status.json", {"status": "failed", "step": step, "error": repr(error)})
        raise
    import resource
    status = {"status": "complete" if step == args.steps else "paused", "step": step,
              "elapsed_seconds": time.monotonic() - started,
              "peak_process_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
              "peak_process_rss_scope": "Linux current process lifetime"}
    training.write_json(args.run_dir / "status.json", status)
    print(json.dumps(status), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Compute parent and teacher features once")
    prep.add_argument("--parent", type=Path, required=True)
    prep.add_argument("--teacher", type=Path)
    prep.add_argument("--examples", type=int, default=2048)
    prep.add_argument("--batch-size", type=int, default=2)
    prep.add_argument("--workers", type=int, default=2)
    prep.add_argument("--data-start", type=int, default=896000)
    prep.add_argument("--silence-probability", type=float, default=.0625)
    prep.add_argument("--bass-tone-probability", type=float, default=.25)
    prep.add_argument("--stop-after-batches", type=int)
    run = commands.add_parser("train", help="Train a head against the complete fixed cache")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--phase-features", action="store_true")
    run.add_argument("--steps", type=int, default=4000)
    run.add_argument("--lr", type=float, default=3e-4)
    run.add_argument("--min-lr", type=float, default=3e-5)
    run.add_argument("--warmup", type=int, default=100)
    run.add_argument("--seam-weight", type=float, default=.1)
    run.add_argument("--lowband-weight", type=float, default=.1)
    run.add_argument("--distillation-weight", type=float, default=.25)
    run.add_argument("--checkpoint-every", type=int, default=500)
    run.add_argument("--stop-after", type=int)
    for child in (prep, run):
        child.add_argument("--cache-dir", type=Path, required=True)
        child.add_argument("--threads", type=int, default=2)
        child.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    if args.threads < 1 or args.seed < 0:
        parser.error("Require positive threads and nonnegative seed")
    if args.command == "prepare":
        if args.examples < 2 or args.batch_size < 2 or args.examples % args.batch_size or args.workers < 0 or args.data_start < 0:
            parser.error("Require examples divisible by batch size >= 2 and nonnegative workers/data start")
        if any(not 0 <= value <= 1 for value in (args.silence_probability, args.bass_tone_probability)):
            parser.error("Augmentation probabilities must be finite and between 0 and 1")
        if args.stop_after_batches is not None and args.stop_after_batches < 1:
            parser.error("stop-after-batches must be positive")
    else:
        if not (0 < args.min_lr <= args.lr and math.isfinite(args.lr) and 0 <= args.warmup < args.steps and args.checkpoint_every > 0):
            parser.error("Require finite 0 < min-lr <= lr, 0 <= warmup < steps, checkpoint-every > 0")
        if any(not math.isfinite(value) or value < 0 for value in (args.seam_weight, args.lowband_weight, args.distillation_weight)):
            parser.error("Loss weights must be finite and nonnegative")
        if args.stop_after is not None and not 0 < args.stop_after <= args.steps:
            parser.error("stop-after must be between 1 and steps")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    set_cpu_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    (prepare if args.command == "prepare" else train)(args)


if __name__ == "__main__":
    main()
