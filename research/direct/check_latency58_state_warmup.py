"""CPU parity and timing checks for the training-only final-tail warmup."""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import sys
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import PRODUCTION, state_sha256, verify_inputs


def main():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Run CPU1 checks with CUDA hidden")
    import torch
    from research.direct.latency58_magnitude_checkpoint import load_model
    from research.direct.latency58_recorded301_data import select_tracks
    from research.direct.latency58_sdr_context import render_scored_context
    from research.direct.latency58_sdr_checkpoint import require_space
    from research.direct.latency58_state_warmup import warm_state, VERSION
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    source_path = PHASE / "full-magnitude-001/plan.json"
    source = read(source_path)
    verify_inputs(source)
    require_space(source, 370_000_000)
    out = PHASE / "state-warmup-cpu-001"
    require(not out.exists(), "Preserve warmup diagnostics")
    trained_path = PHASE / "magnitude-sdr-001/result.json"
    checkpoint = read(trained_path)["checkpoint"]
    model, _ = load_model(checkpoint)
    require(torch.count_nonzero(model.spec_encode.magnitude_projection.weight) > 0,
            "Check actual nonzero trained magnitude features")
    initial_sha = state_sha256(model.state_dict())
    bindings = {**source["source_bindings"], str(source_path): sha(source_path),
                str(trained_path): sha(trained_path), checkpoint["path"]: checkpoint["sha256"],
                str(Path(__file__).resolve()): sha(__file__),
                str(ROOT / "research/direct/latency58_state_warmup.py"): sha(ROOT / "research/direct/latency58_state_warmup.py"),
                str(ROOT / "research/direct/latency58_magnitude_checkpoint.py"): sha(ROOT / "research/direct/latency58_magnitude_checkpoint.py")}
    sys.path.insert(0, str(PRODUCTION))
    import train_production as production
    config = source["config"]
    _, tracks, _, _ = production.load_corpus_manifest(PRODUCTION / "manifests/combined.manifest.json",
        expected_file_sha256=source["manifest_sha256"], config=read(PRODUCTION / "full_config.json"))
    tracks = select_tracks(tracks, source["training_selection"])
    dataset = production.CounterAddressedCropDataset(tracks, root_weights=config["root_weights"], seed=config["data_seed"],
        crop_samples=config["crop_samples"], vocal_active_probability=config["vocal_active_probability"],
        final_sample_index=config["data_start"] + 4)
    recorded = torch.stack([dataset[i][0] for i in range(config["data_start"], config["data_start"] + 4)])
    generator = torch.Generator().manual_seed(20261008)
    noise = torch.randn(2, 2, 8 * 128, generator=generator) * .05
    impulse = torch.zeros_like(noise)
    impulse[..., -130] = .2
    out.mkdir()
    write(out / "plan.json", {"source_bindings": bindings, "checkpoint": checkpoint, "version": VERSION,
          "cpu_only": True, "validation_used": False, "state_atol": 5e-5,
          "following_audio_atol": 5e-6, "relative_gradient_norm_tolerance": 1e-3})
    began, cases = time.monotonic(), []
    with torch.no_grad():
        incoming = model.render(noise).state.detached()
        for name, audio, state in (("one_hop", noise[..., :128], None), ("noise", noise, None),
                ("silence", noise * 0, None), ("impulse", impulse, None), ("carried_state", noise, incoming),
                ("recorded_micro4_full_warmup", recorded[..., :88064], None)):
            expected = model.render(audio, state).state
            actual = warm_state(model, audio, state)
            errors = {key: float((a - b).abs().max()) for key, a, b in zip(expected._fields, expected, actual, strict=True)}
            require(type(actual) is type(expected) and all(torch.isfinite(v).all() and not v.requires_grad for v in actual)
                    and max(errors.values()) < 5e-5, "Warm state differs: " + name)
            suffix = torch.randn(audio.shape[0], 2, 8 * 128, generator=generator) * .03
            left, right = model.render(suffix, expected), model.render(suffix, actual)
            output_error = max(float((a - b).abs().max()) for a, b in ((left.raw, right.raw), (left.deployed, right.deployed)))
            require(output_error < 5e-6 and torch.equal(actual.audio_history, expected.audio_history)
                    and torch.equal(actual.fusion_hidden, expected.fusion_hidden), "Following output or recurrent state differs")
            row = {"case": name, "state_max_abs": errors, "following_output_max_abs": output_error,
                   "audio_and_recurrent_state_exact": True}
            cases.append(row)
            print(json.dumps(row), flush=True)
        timings = {"full_render": [], "state_only": []}
        for repeat in range(4):
            for label in (("full_render", "state_only") if repeat % 2 == 0 else ("state_only", "full_render")):
                started = time.perf_counter()
                if label == "full_render":
                    model.render(recorded[..., :88064]).state
                else:
                    warm_state(model, recorded[..., :88064])
                timings[label].append(time.perf_counter() - started)
            print(json.dumps({"timing_repeat": repeat + 1}), flush=True)
    model.train().requires_grad_(True)
    warm, score = 8 * 128, 8 * 128
    audio = torch.randn(1, 2, warm + score, generator=generator) * .04
    target = torch.randn(1, 4, 2, score, generator=generator) * .01
    gradients, estimates = [], []
    for fast in (False, True):
        model.zero_grad(set_to_none=True)
        probe = audio.clone().requires_grad_()
        if fast:
            state = warm_state(model, probe[..., :warm])
            rendered = model.render(torch.nn.functional.pad(probe[..., warm:], (0, 128)), state)
            raw, deployed = (v[..., 128:128 + score] for v in (rendered.raw, rendered.deployed))
        else:
            rendered = render_scored_context(model, probe, warmup_samples=warm, carry_state=True)
            raw, deployed = rendered.raw, rendered.deployed
        loss = (deployed - target).square().mean() + .25 * (raw - target).square().mean()
        loss.backward()
        require(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max() > 0 for p in model.parameters())
                and torch.count_nonzero(probe.grad[..., :warm]) == 0
                and torch.count_nonzero(probe.grad[..., warm:]) > 0, "Detached prefix or live scored gradients differ")
        gradients.append({name: p.grad.clone() for name, p in model.named_parameters()})
        estimates.append(deployed.detach().clone())
    gradient_errors = {name: float((value - gradients[1][name]).norm() / value.norm().clamp_min(1e-10))
                       for name, value in gradients[0].items()}
    require(max(gradient_errors.values()) < 1e-3 and torch.max((estimates[0] - estimates[1]).abs()) < 5e-6,
            "Scored gradients or predictions differ")
    require(state_sha256(model.state_dict()) == initial_sha and not torch.cuda.is_initialized(), "Model or CPU scope changed")
    verify_inputs({"source_bindings": bindings})
    result = {"status": "pass", "version": VERSION, "cases": cases, "cpu_seconds": timings,
              "cpu_state_only_over_full_render_median_ratio": statistics.median(timings["state_only"]) / statistics.median(timings["full_render"]),
              "all22_parameter_gradient_relative_errors": gradient_errors,
              "warmup_input_gradients_zero": True, "scored_input_gradients_nonzero": True,
              "model_and_source_bindings_unchanged": True, "gpu_used": False, "gpu_speedup_measured": False,
              "quality_measured": False, "elapsed_seconds": time.monotonic() - began,
              "counted_bytes_after": require_space(source, 370_000_000)}
    write(out / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "all22_parameter_gradient_relative_errors", "cpu_seconds")}), flush=True)


if __name__ == "__main__":
    main()
