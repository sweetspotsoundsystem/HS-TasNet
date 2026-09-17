"""Fit a small branch balance on training audio and measure separate training crops."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import PRODUCTION, state_sha256, verify_inputs

VERSION = "latency58-training-only-spectral-waveform-balance-v1"
WINDOW = 44100


def identity(torch):
    return torch.cat((torch.eye(4, dtype=torch.float64)[:3], torch.zeros(3, 4, dtype=torch.float64)), 1)


def matrices(coefficients):
    import torch
    physical = coefficients.new_tensor([1, 1, 1, 1, 0, 0, 0, 0])
    return torch.cat((coefficients, physical[None] - coefficients.sum(0, keepdim=True)), 0)


def moments(features, targets, mixture):
    require(features.shape[1:3] == (8, 2) and targets.shape[1:3] == (4, 2)
            and features.shape[0] == targets.shape[0] == mixture.shape[0]
            and features.shape[-1] == targets.shape[-1] == mixture.shape[-1], "Unaligned branch features")
    windows = features.shape[-1] // WINDOW
    def frame(value):
        return value[..., :windows * WINDOW].reshape(value.shape[0], value.shape[1], 2, windows, WINDOW).permute(
            0, 3, 1, 2, 4).reshape(-1, value.shape[1], 2 * WINDOW).double()
    x, target = frame(features), frame(targets)
    physical = frame(mixture[:, None])[:, 0]
    return {"gram": x @ x.transpose(1, 2), "cross": x @ target.transpose(1, 2),
            "truth_energy": target.square().sum(-1), "mixture_energy": physical.square().sum(-1)}


def objective(coefficients, statistics):
    import torch
    matrix = matrices(coefficients)
    prediction = torch.einsum("oi,nij,oj->no", matrix, statistics["gram"], matrix)
    correlation = torch.einsum("oi,nio->no", matrix, statistics["cross"])
    signal = statistics["truth_energy"]
    error = (prediction - 2 * correlation + signal).clamp_min(0)
    active = signal / (2 * WINDOW) > 1e-5
    values = (10 * torch.log10((signal + 1e-12) / (error + 1e-12))).clamp(-60, 60)
    count = active.sum(0)
    stems = torch.where(active, values, 0).sum(0) / count.clamp_min(1)
    leakage = 10 * torch.log10(1 + prediction.clamp_min(0) / statistics["mixture_energy"][:, None].clamp_min(2 * WINDOW * 1e-5))
    absent_count = (~active).sum(0)
    absence = (torch.where(~active, leakage, 0).sum(0) / absent_count.clamp_min(1)).sum() / (absent_count > 0).sum().clamp_min(1)
    loss = -stems.sum() / (count > 0).sum().clamp_min(1) + .5 * absence + .05 * (coefficients - identity(torch)).square().mean()
    return loss, stems, absence


def check():
    import torch
    generator = torch.Generator().manual_seed(20261006)
    target = torch.randn(2, 4, 2, 2 * WINDOW, generator=generator) * .04
    target[0, 1, :, :WINDOW] = 0
    mixture = target.sum(1)
    deployed = target + torch.randn(target.shape, generator=generator) * .015
    deployed[:, 3] = mixture - deployed[:, :3].sum(1)
    contrast = torch.randn(target.shape, generator=generator) * .02
    features = torch.cat((deployed, contrast), 1)
    coefficients = identity(torch) + torch.randn(3, 8, dtype=torch.float64, generator=generator) * .02
    statistics = moments(features, target, mixture)
    _, calculated, _ = objective(coefficients, statistics)
    estimate = torch.einsum("os,bsct->boct", matrices(coefficients), features.double())
    signal = target.double().reshape(2, 4, 2, 2, WINDOW).square().sum((2, 4))
    error = (estimate - target.double()).reshape(2, 4, 2, 2, WINDOW).square().sum((2, 4))
    active = signal / (2 * WINDOW) > 1e-5
    values = (10 * torch.log10((signal + 1e-12) / (error + 1e-12))).clamp(-60, 60)
    expected = torch.where(active, values, 0).sum((0, 2)) / active.sum((0, 2)).clamp_min(1)
    discrepancy = float((calculated - expected).abs().max())
    require(discrepancy < 1e-5, "Covariance branch loss differs from actual waveform error")
    variable = coefficients.clone().requires_grad_()
    loss, _, _ = objective(variable, statistics)
    loss.backward()
    require(bool(torch.isfinite(variable.grad).all()) and torch.count_nonzero(variable.grad[:, 4:]) > 0,
            "Branch contrast lacks finite restoring gradients")
    return {"status": "pass", "covariance_vs_waveform_sdr_max_abs_db": discrepancy, "contrast_gradients_finite_and_nonzero": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CPU1 with CUDA hidden")
    import torch
    from research.direct.latency58_direct_sdr_checkpoint import load_model
    from research.direct.latency58_musdb_sdr_data import selection_contract, select_tracks
    from research.direct.latency58_sdr_checkpoint import require_space
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    functional = check()
    if args.check_only:
        print(json.dumps(functional), flush=True)
        return
    out = PHASE / "branch-balance-training-001"
    require(not out.exists(), "Preserve previous branch diagnostics")
    budget = read(PHASE / "wave-spectral-001/plan.json")
    counted = require_space(budget, 355_000_000)
    binding = read(PHASE / "c204-residual-model-001/checkpoint.json")
    parent, _ = load_model(binding)
    parent_state = state_sha256(parent.state_dict())
    contract = selection_contract()
    paths = [Path(__file__).resolve(), ROOT / "research/direct/latency58_direct_sdr_checkpoint.py",
             ROOT / "research/direct/latency58_musdb_sdr_data.py", ROOT / "research/direct/latency58_sdr_checkpoint.py",
             PRODUCTION / "train_production.py", PRODUCTION / "full_config.json", PRODUCTION / "manifests/combined.manifest.json",
             Path(binding["path"])]
    bindings = {**read(PHASE / "c204-residual-model-001/qualification.json")["source_bindings"], **{str(path): sha(path) for path in paths}}
    plan = {"schema": VERSION, "parent_checkpoint": binding, "parent_model_state_sha256": parent_state,
            "training_selection": contract, "fit_sample_indices": list(range(1_800_000, 1_800_256)),
            "probe_sample_indices": list(range(1_900_000, 1_900_064)), "data_seed": 60, "vocal_active_probability": .85,
            "crop_samples": 176384, "warmup_samples": 88064, "cpu_batch_size": 4, "optimization_steps": 2000,
            "maximum_lr": .01, "minimum_lr": .0001, "coefficient_displacement_bounds": [-.25, .25],
            "features": "four complete parent deployed stems followed by four native spectral-minus-waveform stem contrasts",
            "loss": "equal-stem windowed SD-SDR + 0.5 relative absence energy + 0.05 mean coefficient displacement squared",
            "probe_minimum_improvement_db": .1, "probe_scope": "different crops in the recorded training split, not unseen-track validation",
            "fit_used_validation_or_test_audio": False, "full14_executed": False,
            "source_bindings": bindings, "output_directory": str(out)}
    verify_inputs(plan)
    out.mkdir()
    write(out / "plan.json", plan)
    write(out / "functional.json", functional)
    sys.path.insert(0, str(PRODUCTION))
    import train_production as production
    source_config = read(PRODUCTION / "full_config.json")
    _, tracks, manifest_hash, _ = production.load_corpus_manifest(PRODUCTION / "manifests/combined.manifest.json",
        expected_file_sha256=contract["source_manifest_sha256"], config=source_config)
    require(manifest_hash == contract["source_manifest_sha256"] and source_config["seed"] == plan["data_seed"], "Corpus changed")
    tracks = select_tracks(tracks, contract)
    dataset = production.CounterAddressedCropDataset(tracks, root_weights={"musdb18hq_train": 1.}, seed=plan["data_seed"],
        crop_samples=plan["crop_samples"], vocal_active_probability=plan["vocal_active_probability"],
        final_sample_index=max(plan["probe_sample_indices"]) + 1)
    statistics = {}
    began = time.monotonic()
    with (out / "progress.jsonl").open("x", buffering=1) as journal, torch.inference_mode():
        for role in ("fit", "probe"):
            rows = []
            indices = plan[role + "_sample_indices"]
            for first in range(0, len(indices), 4):
                addresses = indices[first:first + 4]
                inputs = [dataset[index] for index in addresses]
                mixture = torch.stack([x for x, y in inputs])
                truth = torch.stack([y for x, y in inputs])
                digest = hashlib.sha256()
                for tensor in (mixture, truth):
                    digest.update(tensor.numpy().tobytes())
                warm = plan["warmup_samples"]
                state = parent.render(mixture[..., :warm]).state.detached()
                suffix = mixture[..., warm:]
                output = parent.render(torch.nn.functional.pad(suffix, (0, 128)), state)
                score = slice(128, 128 + suffix.shape[-1])
                require(torch.equal(output.delayed_mixture[..., score], suffix), "Physical branch alignment changed")
                deployed = output.deployed[..., score]
                contrast = (output.spectral - output.waveform)[..., score]
                features = torch.cat((deployed, contrast), 1)
                rows.append(moments(features, truth[..., warm:], suffix))
                row = {"role": role, "sample_indices": addresses, "input_sha256": digest.hexdigest(),
                       "completed_examples": first + len(addresses), "elapsed_seconds": time.monotonic() - began}
                journal.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                del inputs, mixture, truth, state, suffix, output, deployed, contrast, features
            statistics[role] = {key: torch.cat([row[key] for row in rows]) for key in rows[0]}
    statistics = {role: {key: value.clone() for key, value in rows.items()} for role, rows in statistics.items()}
    with (out / "training-covariances.pt").open("xb") as stream:
        torch.save(statistics, stream)
    initial = identity(torch)
    coefficients = initial.clone().requires_grad_()
    optimizer = torch.optim.Adam([coefficients], lr=.01, foreach=False)
    for step in range(plan["optimization_steps"]):
        optimizer.param_groups[0]["lr"] = .0001 + .5 * (.01 - .0001) * (1 + math.cos(math.pi * step / (plan["optimization_steps"] - 1)))
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = objective(coefficients, statistics["fit"])
        loss.backward()
        require(bool(torch.isfinite(loss)) and bool(torch.isfinite(coefficients.grad).all()), "Nonfinite branch fit")
        optimizer.step()
        with torch.no_grad():
            coefficients.clamp_(initial - .25, initial + .25)
    coefficients = coefficients.detach().float()
    summaries = {}
    for role, values in statistics.items():
        _, before, absence_before = objective(initial, values)
        _, after, absence_after = objective(coefficients.double(), values)
        summaries[role] = {"parent_full_sdr_db": float(before.mean()), "candidate_full_sdr_db": float(after.mean()),
            "full_sdr_improvement_db": float((after - before).mean()), "per_stem_parent_db": before.tolist(),
            "per_stem_candidate_db": after.tolist(), "per_stem_delta_db": (after - before).tolist(),
            "parent_absence": float(absence_before), "candidate_absence": float(absence_after)}
    require(state_sha256(parent.state_dict()) == parent_state and not torch.cuda.is_initialized(), "Parent or CPU scope changed")
    verify_inputs(plan)
    write(out / "result.json", {"status": "complete", "coefficients": coefficients.tolist(), "summaries": summaries,
        "probe_passed": summaries["probe"]["full_sdr_improvement_db"] >= plan["probe_minimum_improvement_db"],
        "elapsed_seconds": time.monotonic() - began, "gpu_used": False, "full14_executed": False,
        "fit_used_validation_or_test_audio": False, "source_bindings_unchanged": True, "parent_unchanged": True,
        "counted_bytes_before": counted, "counted_bytes_after": require_space(budget, 351_000_000),
        "quality_claimed": False, "plan_sha256": sha(out / "plan.json"),
        "statistics_sha256": sha(out / "training-covariances.pt")})
    print(json.dumps(read(out / "result.json")), flush=True)


if __name__ == "__main__":
    main()
