"""Direct full-model C126/C191 continuation at 512-sample deployment latency."""

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
from research.direct.checkpoints import load_model as load_c91_model
from research.direct.latency11 import C126_CHECKPOINT, load_model, long_analysis_metadata, save_model
from research.direct.latency_augment import add_bass_tones

PRODUCTION_ROOT = Path("/home/axel/autoresearch/production/hs-tasnet-c91-full-v1")
sys.path.insert(0, str(PRODUCTION_ROOT))
import train_production as production

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


def seam_error_l1(estimates, targets, radius=32):
    """Target-error first differences around each callback edge.

    Compare derivatives of the error, so faithful transients receive no penalty.
    Ignore the first four callbacks, whose recurrent state starts at zero.
    """
    if estimates.shape[-1] <= 2048:
        return estimates.float().sum() * 0.0
    if not 0 <= radius < 256:
        raise ValueError("Seam radius must be between 0 and 255 samples")
    positions = torch.arange(1, estimates.shape[-1], device=estimates.device)
    phase = positions.remainder(512)
    selected = ((phase <= radius) | (phase >= 512 - radius)) & (positions >= 2048)
    differences = torch.diff(estimates.float() - targets.float(), dim=-1)
    return differences[..., selected].abs().mean()


def lowband_error_l1(estimates, targets):
    """Supervised 20–250 Hz error for all four deployed stems, in FP32.

    Hann tapering suppresses the artificial periodic join of a finite crop.
    Divide only by the fixed window mean to retain waveform amplitude units.
    The existing untapered waveform loss still supervises the crop edges.
    """
    if estimates.shape != targets.shape or estimates.ndim != 4 or estimates.shape[1:3] != (4, 2):
        raise ValueError("Low-band loss requires matching [batch,4,2,samples] audio")
    if estimates.shape[-1] < 4096:
        raise ValueError("Low-band loss requires at least 4096 samples")
    with torch.autocast(estimates.device.type, enabled=False):
        error = estimates.float() - targets.float()
        length = error.shape[-1]
        window = torch.hann_window(length, periodic=False, device=error.device)
        frequencies = torch.fft.rfftfreq(length, d=1.0 / 44_100, device=error.device)
        selected = (frequencies >= 20.0) & (frequencies <= 250.0)
        spectrum = torch.fft.rfft(error * window, dim=-1)
        low_error = torch.fft.irfft(spectrum * selected, n=length, dim=-1)
        return low_error.abs().mean() / window.mean()


@torch.no_grad()
def aligned_teacher_dbv(teacher, mixture):
    """Frozen FP32 C91 deployment-gain DBV on the student's current timeline."""
    with torch.autocast(mixture.device.type, enabled=False):
        estimates, _ = teacher(
            mixture.float(), auto_causal_pad=True,
            auto_curtail_length_to_multiple=False, is_streaming=False,
        )
    expected = (mixture.shape[0], 4, mixture.shape[1], mixture.shape[-1])
    if estimates.shape != expected:
        raise ValueError(f"Teacher alignment differs: {estimates.shape} != {expected}")
    return estimates[:, :3].float()


def apply_silence(mixture, targets, deranged, probability):
    """Replace selected complete examples with silence after ordinary remixing."""
    if not probability:
        return mixture, targets, deranged, None
    selected = torch.rand(mixture.shape[0], device=mixture.device) < probability
    mixture, targets, deranged = mixture.clone(), targets.clone(), deranged.clone()
    mixture[selected] = 0
    targets[selected] = 0
    deranged[selected] = False
    return mixture, targets, deranged, selected


def configure_training_scope(model, scope):
    """Select ordinary trainable parameters without changing model state."""
    if scope == "all":
        model.train().requires_grad_(True)
    elif scope in ("c191-head", "c191-core", "c191-core-analysis"):
        if model.kind != "c191":
            raise ValueError(f"{scope} training requires a C191 model")
        if scope == "c191-core-analysis" and long_analysis_metadata(model) is None:
            raise ValueError("c191-core-analysis requires an explicit long-analysis checkpoint")
        model.eval().requires_grad_(False)
        trainable_module = model.engine.head if scope == "c191-head" else model.engine.core
        # Frozen post layers retain ordinary autograd back to a trainable core.
        trainable_module.train().requires_grad_(True)
        if scope == "c191-core-analysis":
            model.engine.long_analysis.train().requires_grad_(True)
    else:
        raise ValueError(f"Unknown training scope: {scope}")
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def train_batch(
    model, optimizer, mixture, targets, *, projection: bool, deployed_l1_weight: float = 0.0,
    seam_weight: float = 0.0,
    seam_radius: int = 32,
    lowband_weight: float = 0.0,
    distillation_teacher=None,
    distillation_weight: float = 0.0,
    silence_probability: float = 0.0,
    precision: str = "bf16",
    bass_tone_probability: float = 0.0,
    bass_tone_seed: int | None = None,
    pure_bass_probability: float = 0.0,
    pure_bass_seed: int | None = None,
    uncalibrated_raw_loss: bool = False,
) -> dict:
    if precision not in ("bf16", "fp32"):
        raise ValueError(f"Unknown forward precision: {precision}")
    if mixture.device.type == "cpu" and precision != "fp32":
        raise ValueError("CPU training requires FP32 precision")
    optimizer.zero_grad(set_to_none=True)
    mixture, targets, deranged = experiment._augment_training_distribution(
        mixture=mixture, targets=targets,
    )
    if bass_tone_probability:
        if bass_tone_seed is None:
            raise ValueError("Bass-tone augmentation requires an explicit data-index seed")
        mixture, targets, toned = add_bass_tones(
            mixture, targets, bass_tone_probability, bass_tone_seed,
        )
    pure_bass_drawn = pure_bass_active = None
    if pure_bass_probability:
        from research.direct.latency_pure_bass import replace_with_pure_bass
        mixture, targets, deranged, pure_bass_drawn = replace_with_pure_bass(
            mixture, targets, deranged, pure_bass_probability, seed=pure_bass_seed,
        )
        pure_bass_active = pure_bass_drawn
    if silence_probability:
        mixture, targets, deranged, silenced = apply_silence(
            mixture, targets, deranged, silence_probability,
        )
        if pure_bass_probability:
            pure_bass_active = pure_bass_drawn & ~silenced
    if distillation_weight:
        if distillation_teacher is None:
            raise ValueError("A positive distillation weight requires a teacher")
        teacher_dbv = aligned_teacher_dbv(distillation_teacher, mixture)
    with torch.autocast(mixture.device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
        estimates, _ = model.forward_raw(mixture)
    if estimates.shape != targets.shape:
        raise ValueError(f"Current-chunk estimates and targets differ: {estimates.shape} != {targets.shape}")
    # Native gains and the effective decoder are already applied by the model.
    # Every output sample corresponds to this crop's same input sample.
    if uncalibrated_raw_loss:
        from research.direct.latency_raw_loss import uncalibrated_raw_l1
        waveform_loss = uncalibrated_raw_l1(estimates, targets, model.output_source_scales)
    else:
        waveform_loss = torch.nn.functional.l1_loss(estimates.float(), targets.float())
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
    if deployed_l1_weight or seam_weight or lowband_weight:
        length = estimates.shape[-1]
        dbv = estimates[:, :3].float()
        other = mixture[..., :length].float().unsqueeze(1) - dbv.sum(dim=1, keepdim=True)
        deployed = torch.cat((dbv, other), dim=1)
    if deployed_l1_weight:
        deployed_l1 = torch.nn.functional.l1_loss(
            deployed, targets[..., :length].float(),
        )
        loss = (1.0 - deployed_l1_weight) * waveform_loss + deployed_l1_weight * deployed_l1 + contribution
    else:
        loss = waveform_loss + contribution
    if seam_weight:
        seam_l1 = seam_error_l1(deployed, targets, radius=seam_radius)
        seam_contribution = seam_weight * seam_l1
        loss = loss + seam_contribution
    if lowband_weight:
        lowband_l1 = lowband_error_l1(deployed, targets)
        lowband_contribution = lowband_weight * lowband_l1
        loss = loss + lowband_contribution
    if distillation_weight:
        if pure_bass_probability:
            teacher_error = torch.nn.functional.l1_loss(
                estimates[:, :3].float(), teacher_dbv, reduction="none",
            )
            # Zero final pure-Bass examples; retain the original B*3*2*T denominator.
            distillation_l1 = teacher_error.masked_fill(
                pure_bass_active[:, None, None, None], 0.0,
            ).mean()
        else:
            distillation_l1 = torch.nn.functional.l1_loss(estimates[:, :3].float(), teacher_dbv)
        distillation_contribution = distillation_weight * distillation_l1
        loss = loss + distillation_contribution
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
    if uncalibrated_raw_loss:
        metrics["native_raw_waveform_l1"] = float(torch.nn.functional.l1_loss(
            estimates.detach().float(), targets.float(),
        ))
    if seam_weight:
        metrics["seam_error_l1"] = float(seam_l1.detach())
        metrics["seam_contribution"] = float(seam_contribution.detach())
        metrics["seam_fraction_of_waveform_l1"] = float(
            seam_contribution.detach() / waveform_loss.detach().clamp_min(1e-8),
        )
    if distillation_weight:
        metrics["distillation_dbv_l1"] = float(distillation_l1.detach())
        metrics["distillation_contribution"] = float(distillation_contribution.detach())
    if lowband_weight:
        metrics["lowband_l1"] = float(lowband_l1.detach())
        metrics["lowband_contribution"] = float(lowband_contribution.detach())
        metrics["lowband_fraction_of_waveform_l1"] = (
            metrics["lowband_contribution"] / metrics["waveform_l1"]
            if metrics["waveform_l1"] > 0.0 else None
        )
    if silence_probability:
        metrics["silenced_examples"] = int(silenced.sum())
    if bass_tone_probability:
        metrics["bass_tone_drawn_examples"] = int(toned.sum())
        surviving_toned = toned
        if pure_bass_probability:
            surviving_toned = surviving_toned & ~pure_bass_drawn
        if silence_probability:
            surviving_toned = surviving_toned & ~silenced
        metrics["bass_tone_examples"] = int(surviving_toned.sum())
    if pure_bass_probability:
        metrics["pure_bass_drawn_examples"] = int(pure_bass_drawn.sum())
        metrics["pure_bass_examples"] = int(pure_bass_active.sum())
        if distillation_weight:
            metrics["distillation_excluded_pure_bass_examples"] = int(pure_bass_active.sum())
    return metrics


def save_checkpoint(run_dir, model, optimizer, step, config) -> None:
    if config.get("pure_bass_probability", 0.0):
        model.provenance["pure_bass_training"] = {
            "step": step,
            "probability": config["pure_bass_probability"],
            "augmentation": config["pure_bass_augmentation"],
            "parent_path": config["parent_path"],
            "parent_sha256": config["parent_sha256"],
            "run_config_sha256": file_hash(run_dir / "config.json"),
        }
    payload = {
        "latency11_kind": model.kind,
        "model": model.engine.state_dict(), "config": model.engine.core_config,
        "vocal_gain": model.vocal_gain, "provenance": model.provenance,
        "optimizer": optimizer.state_dict(), "step": step, "run_config": config,
        "rng": {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if config.get("device", "cuda") == "cuda" else [],
        },
    }
    if model.kind == "c191" and getattr(model.engine.head, "phase_features", False):
        payload["c191_head_phase_features"] = "phase12-v1"
    analysis_metadata = long_analysis_metadata(model)
    if config.get("c191_long_analysis") != analysis_metadata:
        raise ValueError("Checkpoint architecture differs from long-analysis run configuration")
    if analysis_metadata is not None:
        payload["c191_long_analysis"] = analysis_metadata
        payload["provenance"] = dict(payload["provenance"],
                                     c191_long_analysis=analysis_metadata,
                                     n_params=model.num_parameters)
    temporary = run_dir / "resume.pt.tmp"
    torch.save(payload, temporary)
    temporary.replace(run_dir / "resume.pt")
    save_model(model, run_dir / f"step-{step:06d}.pt")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("c126", "c191"), required=True)
    parser.add_argument("--parent", type=Path,
                        help="Direct snapshot; omitted loads the retained native baseline.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--min-lr", type=float, default=3e-6)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--projection", choices=("off", "on"), default="off")
    parser.add_argument("--vocal-gain", type=float,
                        help="Optional final-output vocal calibration; omitted preserves native gain.")
    parser.add_argument("--deployed-l1-weight", type=float, default=0.0,
                        help="L1 fraction with residual Other; 0 supervises all native raw heads.")
    parser.add_argument("--uncalibrated-raw-loss", action="store_true",
                        help="Use inverse-gain raw waveform L1 without changing forward gains.")
    parser.add_argument("--seam-weight", type=float, default=0.0,
                        help="Weight on target-error differences near callback seams; try 0.1.")
    parser.add_argument("--seam-radius", type=int, default=32,
                        help="Samples on each side of each seam; 0 selects exact seams only.")
    parser.add_argument("--lowband-weight", type=float, default=0.0,
                        help="Added 20–250 Hz deployed error L1 with a Hann taper; try 0.1.")
    parser.add_argument("--teacher-weight", type=float, default=0.25,
                        help="Teacher-root sampling probability; human roots retain their 2:1 ratio.")
    parser.add_argument("--distillation-teacher", type=Path,
                        help="Frozen C91 deployment checkpoint for optional aligned DBV distillation.")
    parser.add_argument("--distillation-weight", type=float, default=0.0,
                        help="Added weight on mean DBV teacher L1; independent of corpus sampling.")
    parser.add_argument("--silence-probability", type=float, default=0.0,
                        help="Probability of replacing a remixed crop and all four targets with silence.")
    parser.add_argument("--bass-tone-probability", type=float, default=0.0,
                        help="Add randomized clean bass notes to Bass and mixture after remixing.")
    parser.add_argument("--pure-bass-probability", type=float, default=0.0,
                        help="Replace selected crops with fresh pure-Bass notes before silence; omit their teacher error.")
    parser.add_argument("--corrections-fp32", action="store_true",
                        help="Keep C191 correction modules in FP32 while its core uses BF16.")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16",
                        help="Forward precision; fp32 also avoids rounding in a frozen core.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda",
                        help="Training device; CPU requires --precision fp32.")
    parser.add_argument("--cpu-threads", type=int, default=1,
                        help="Positive intra-op thread count for CPU training; ignored on CUDA.")
    parser.add_argument("--train-scope", choices=("all", "c191-head", "c191-core", "c191-core-analysis"), default="all",
                        help="Train all weights, the C191 D/B head, its core, or core plus opt-in analysis features.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--data-start", type=int, default=50_000 * 16)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--stop-after", type=int, help="Pause at this run-local update; resume with the same command.")
    args = parser.parse_args(argv)
    if args.cpu_threads <= 0:
        parser.error("cpu-threads must be a positive integer")
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
    if not math.isfinite(args.distillation_weight) or args.distillation_weight < 0:
        parser.error("distillation-weight must be finite and nonnegative")
    if args.distillation_weight and args.distillation_teacher is None:
        parser.error("positive distillation-weight requires distillation-teacher")
    if not 0.0 <= args.silence_probability <= 1.0:
        parser.error("silence-probability must be between 0 and 1")
    if not 0.0 <= args.bass_tone_probability <= 1.0:
        parser.error("bass-tone-probability must be finite and between 0 and 1")
    if not math.isfinite(args.pure_bass_probability) or not 0.0 <= args.pure_bass_probability <= 1.0:
        parser.error("pure-bass-probability must be finite and between 0 and 1")
    if args.corrections_fp32 and args.kind != "c191":
        parser.error("corrections-fp32 applies only to C191")
    if args.train_scope in ("c191-head", "c191-core", "c191-core-analysis") and args.kind != "c191":
        parser.error(f"{args.train_scope} training requires kind c191")
    if not math.isfinite(args.seam_weight) or args.seam_weight < 0:
        parser.error("seam-weight must be finite and nonnegative")
    if not 0 <= args.seam_radius < 256:
        parser.error("seam-radius must be between 0 and 255 samples")
    if not math.isfinite(args.lowband_weight) or args.lowband_weight < 0:
        parser.error("lowband-weight must be finite and nonnegative")
    if args.vocal_gain is not None and not (math.isfinite(args.vocal_gain) and args.vocal_gain > 0):
        parser.error("vocal-gain must be finite and positive")
    if args.device == "cpu" and args.precision != "fp32":
        parser.error("CPU training requires --precision fp32")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; CPU requires --device cpu --precision fp32")
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 training requires CUDA with BF16 support")

    torch.set_num_threads(args.cpu_threads if args.device == "cpu" else 1)
    torch.set_num_interop_threads(1)
    if args.device == "cuda":
        production.configure_determinism(args.seed)
    else:
        random.seed(args.seed)
        np.random.seed(args.seed % (2**32))
        # torch.manual_seed also seeds CUDA; CPU runs must not touch that RNG.
        torch.random.default_generator.manual_seed(args.seed)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
    production_config = json.loads((PRODUCTION_ROOT / "full_config.json").read_text())
    manifest_path = PRODUCTION_ROOT / "manifests/combined.manifest.json"
    _, tracks, manifest_hash, _ = production.load_corpus_manifest(
        manifest_path, expected_file_sha256=None, config=production_config,
    )
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items()
              if key not in ("run_dir", "stop_after", "distillation_teacher",
                             "distillation_weight", "silence_probability", "corrections_fp32",
                             "lowband_weight", "train_scope", "bass_tone_probability", "device", "cpu_threads",
                             "pure_bass_probability", "uncalibrated_raw_loss")}
    if args.device != "cuda":
        config["device"] = args.device
        config["cpu_threads"] = torch.get_num_threads()
    if args.parent is not None:
        parent_path = args.parent.resolve()
    elif args.kind == "c126":
        parent_path = C126_CHECKPOINT
    else:
        from research.direct.latency11_c191 import FULL_CHECKPOINT
        parent_path = FULL_CHECKPOINT
    config.update({
        "parent_path": str(parent_path),
        "parent_sha256": file_hash(parent_path), "manifest_sha256": manifest_hash,
        "crop_samples": production_config["model"]["crop_samples"],
        "root_weights": production_config["sampling"]["root_weights"],
        "vocal_active_probability": production_config["sampling"]["vocal_active_probability"],
        "data_seed": production_config["seed"], "optimizer": "fresh Adam, default betas, no weight decay",
        "stem_subset_probability": experiment.STEM_SUBSET_AUGMENT_PROBABILITY,
        "vocal_derangement_probability": experiment.VOCAL_DERANGEMENT_PROBABILITY,
        "projection_weight": experiment.DERANGED_PROJECTION_LOSS_WEIGHT,
        "projection_max_l1_fraction": experiment.DERANGED_PROJECTION_MAX_L1_FRACTION,
        "torch_version": torch.__version__,
        "output_convention": "native gains already applied; raw four-head L1 at current-sample alignment; deployed Other = mixture - DBV sum",
        "algorithmic_latency_samples": 512,
        "alignment_samples": 0,
        "precision": args.precision,
        "zero_initial_state_each_crop": True,
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
            "output_convention": "blend native raw-four-head L1 with residual-Other L1; final Other = augmented mixture - DBV sum",
        })
    if args.uncalibrated_raw_loss:
        from research.direct import latency_raw_loss
        config.update({
            "uncalibrated_raw_loss": True,
            "raw_waveform_loss": {
                "kind": "mean L1 of raw predictions divided by fixed effective output gains against references",
                "effective_gains": "2 * model.output_source_scales, including final Vocal output ratio",
                "precision": "float32",
                "denominator": "all batch * 4 sources * channels * samples",
                "forwarding": "native model gains, internal correction inputs and decoder representation unchanged",
                "auxiliary_losses": "teacher, deployed, seam and lowband remain in their existing output domains",
                "projection_cap": "if enabled, its existing waveform-relative cap uses the selected inverse-gain waveform loss",
                "implementation_sha256": file_hash(Path(latency_raw_loss.__file__)),
            },
            "output_convention": "native forward gains preserved; (1-deployed_l1_weight)*inverse-gain raw-head L1 + deployed_l1_weight*deployed-head L1; deployed Other = mixture - DBV sum",
        })
    if args.seam_weight:
        config["seam_loss"] = {
            "kind": "absolute first difference of deployed estimate-minus-target error",
            "radius_samples": args.seam_radius, "skip_samples": 2048,
            "phase": "local crop and callback origin", "sources": "all four",
        }
    if args.lowband_weight:
        config["lowband_weight"] = args.lowband_weight
        config["lowband_loss"] = {
            "kind": "waveform L1 of FFT-bandpassed deployed estimate-minus-target error",
            "sample_rate": 44_100, "band_hz": [20.0, 250.0],
            "window": "full-crop Hann, periodic=False",
            "normalization": "divide by mean Hann amplitude; no signal-dependent normalization",
            "sources": "all four deployed, including mixture-minus-DBV Other",
            "precision": "float32", "alignment": "same current-sample target indices",
            "crop_edges": "auxiliary taper only; existing untapered waveform L1 retained",
        }
    if args.distillation_weight:
        config.update({
            "distillation_teacher": str(args.distillation_teacher.resolve()),
            "distillation_teacher_sha256": file_hash(args.distillation_teacher),
            "distillation_weight": args.distillation_weight,
            "distillation_sources": ["drums", "bass", "vocals"],
            "distillation_precision": "float32",
            "distillation_alignment": "explicit causal padding; no curtailment; same current-sample indices",
            "distillation_gains": "checkpoint gains already applied; no additional scaling",
        })
    if args.silence_probability:
        config["silence_probability"] = args.silence_probability
    if args.bass_tone_probability:
        config["bass_tone_probability"] = args.bass_tone_probability
        config["bass_tone_augmentation"] = {
            "kind": "same new stereo waveform added to Bass target and mixture",
            "position": "after ordinary remixing, before silence replacement and teacher inference",
            "fundamental_hz": [32.0, 220.0], "fundamental_distribution": "log-uniform",
            "harmonics": [1, 4], "harmonic_decay_exponent": [1.0, 3.0],
            "peak_amplitude": [0.01, 0.15], "peak_distribution": "log-uniform",
            "envelope": "random onset/offset with raised-cosine attack and release",
            "rng": "dedicated device generator, seed = training seed + first absolute data index of batch",
            "global_augmentation_rng_unchanged": True,
            "implementation_sha256": file_hash(Path(__file__).with_name("latency_augment.py")),
        }
    if args.pure_bass_probability:
        from research.direct.latency_pure_bass import SEED_DOMAIN
        config["pure_bass_probability"] = args.pure_bass_probability
        config["pure_bass_augmentation"] = {
            "kind": "independent Bernoulli replacement: mixture=note, D/B/V/O targets=0/note/0/0",
            "position": "after remixing and additive tones, before silence and teacher inference",
            "distribution": "existing continuous-frequency add_bass_tones distribution on zeros at probability1",
            "base_seed": "training seed + first absolute data index of batch",
            "seed_domain": SEED_DOMAIN,
            "seed_derivation": "low63 bits of first8 SHA256 bytes, little-endian; domain/purpose/base_seed",
            "seed_purposes": ["selection", "notes"],
            "global_rng_unchanged": True,
            "derangement": "cleared on replacements; silence subsequently clears its selections",
            "teacher_policy": "zero absolute DBV teacher error on final pure examples; unchanged full B*3*2*T denominator",
            "supervision": "existing waveform, deployed, seam and lowband losses retained",
            "implementation_sha256": file_hash(Path(__file__).with_name("latency_pure_bass.py")),
            "note_generator_sha256": file_hash(Path(__file__).with_name("latency_augment.py")),
        }
    if args.corrections_fp32:
        config["corrections_fp32"] = True
    if args.train_scope != "all":
        config["train_scope"] = args.train_scope
    resume_path = args.run_dir / "resume.pt"
    model = load_model(
        args.kind, resume_path if resume_path.exists() else args.parent,
        device=args.device, vocal_gain=args.vocal_gain,
    ).train()
    if args.kind == "c191":
        model.engine.corrections_fp32 = config.get("corrections_fp32", False)
    trainable_parameters = configure_training_scope(model, args.train_scope)
    analysis_metadata = long_analysis_metadata(model)
    if analysis_metadata is not None:
        config["c191_long_analysis"] = analysis_metadata
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.run_dir / "config.json"
    existing_config = json.loads(config_path.read_text()) if config_path.exists() else None
    legacy_cpu_resume = (
        args.device == "cpu" and resume_path.is_file()
        and isinstance(existing_config, dict) and existing_config.get("device") == "cpu"
        and "cpu_threads" not in existing_config
    )
    if legacy_cpu_resume:
        if config["cpu_threads"] != 1:
            raise ValueError("Legacy CPU runs used one thread; resume with --cpu-threads 1 or use a new run directory")
        config.pop("cpu_threads")
    if config_path.exists() and existing_config != config:
        raise ValueError("Run configuration differs; use a new run directory")
    if not legacy_cpu_resume:
        write_json(config_path, config)
    distillation_teacher = None
    if args.distillation_weight:
        distillation_teacher = load_c91_model(
            args.distillation_teacher, device=args.device, raw=False,
        ).float().eval().requires_grad_(False)
        print(json.dumps({"distillation_teacher": {
            "path": str(args.distillation_teacher.resolve()),
            "sha256": config["distillation_teacher_sha256"],
            "output_source_scales": distillation_teacher.output_source_scales.detach().cpu().tolist(),
            "precision": "float32", "weight": args.distillation_weight,
        }}), flush=True)
    print(json.dumps({"model": {
        "kind": model.kind, "n_params": model.num_parameters,
        "n_trainable_params": sum(parameter.numel() for parameter in trainable_parameters),
        "effective_output_source_scales": model.output_source_scales.detach().cpu().tolist(),
        "native_output_source_scales": model.core.output_source_scales.detach().cpu().tolist(),
        "provenance": model.provenance,
    }}), flush=True)
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr)
    step = 0
    if resume_path.exists():
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if payload["run_config"] != config:
            raise ValueError("Resume configuration differs")
        model.engine.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        step = payload["step"]
        rng = payload["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if args.device == "cuda":
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
        num_workers=args.workers, pin_memory=args.device == "cuda", worker_init_fn=production.worker_init,
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
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    log_path = args.run_dir / "metrics.jsonl"
    write_json(args.run_dir / "status.json", {"status": "running", "step": step, "pid": os.getpid()})
    try:
        for mixture, targets in loader:
            lr = learning_rate(step, total=args.steps, peak=args.lr, floor=args.min_lr, warmup=args.warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr
            metrics = train_batch(
                model, optimizer, mixture.to(args.device, non_blocking=args.device == "cuda"),
                targets.to(args.device, non_blocking=args.device == "cuda"),
                projection=args.projection == "on",
                deployed_l1_weight=args.deployed_l1_weight,
                seam_weight=args.seam_weight,
                seam_radius=args.seam_radius,
                lowband_weight=args.lowband_weight,
                distillation_teacher=distillation_teacher,
                distillation_weight=args.distillation_weight,
                silence_probability=args.silence_probability,
                precision=args.precision,
                bass_tone_probability=args.bass_tone_probability,
                bass_tone_seed=args.seed + args.data_start + step * args.batch_size
                               if args.bass_tone_probability else None,
                pure_bass_probability=args.pure_bass_probability,
                pure_bass_seed=args.seed + args.data_start + step * args.batch_size
                               if args.pure_bass_probability else None,
                uncalibrated_raw_loss=args.uncalibrated_raw_loss,
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
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30 if args.device == "cuda" else None,
    }
    if args.device == "cpu":
        import resource
        status["peak_process_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
            2**30 if sys.platform == "darwin" else 2**20)
        status["peak_process_rss_scope"] = "current process lifetime; excludes data-loader workers"
    write_json(args.run_dir / "status.json", status)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
