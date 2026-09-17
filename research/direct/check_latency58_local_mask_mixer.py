"""CPU feasibility only: identity, locality, causality, gradients and added cost."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import statistics
import time

from research.direct.run_latency58_quality import read, require, sha, write
from research.direct.train_latency58 import verify_inputs, state_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Mixer feasibility plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-local-mask-mixer-functional-plan-v1"
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require a bounded CUDA-hidden CPU fixture")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve previous proof")
    import torch
    from research.direct.latency58_drum_accum_parent import load_parent
    from research.direct.latency58_local_mask_mixer import SpectralHeadWithLocalMixer, VERSION
    from research.direct.latency58_sdr_context import render_scored_context, physical_context_teacher
    from research.direct.latency58_sdr_teacher import load_teacher
    from research.direct.latency58_drum_emphasis import drum_emphasized_objective
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260917)
    torch.use_deterministic_algorithms(True)
    began = time.monotonic()
    binding = plan["parent_plan"]
    require(sha(binding["path"]) == binding["sha256"], "Parent plan changed")
    parent_plan = read(binding["path"])
    verify_inputs(parent_plan)
    model = load_parent(parent_plan).eval().requires_grad_(False)
    teacher, teacher_identity = load_teacher("c91", parent_plan["teacher"])
    fingerprint = state_sha256(model.state_dict())
    require(fingerprint == parent_plan["parent"]["model_state_sha256"], "Different preparation parent")
    original_head = model.to_spec_masks
    wrapper = SpectralHeadWithLocalMixer(original_head).eval()
    params = dict(wrapper.mixer.named_parameters())
    require(sum(p.numel() for p in params.values()) == 536 and len(params) == 4, "Unexpected mixer size")
    generator = torch.Generator().manual_seed(20260917)
    audio = torch.randn(2, 2, 2048, generator=generator) * .03
    logits = torch.randn(2, 7, 2 * 513 * 2 * 4, generator=generator) * .2
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        baseline = model.render(audio)
        model.to_spec_masks = wrapper
        identity = model.render(audio)
        require(torch.equal(wrapper.mixer(logits), logits)
                and all(torch.equal(getattr(baseline, k), getattr(identity, k))
                        for k in ("raw", "deployed", "spectral", "waveform", "delayed_mixture"))
                and all(torch.equal(a, b) for a, b in zip(baseline.state, identity.state, strict=True)),
                "Zero mixer does not preserve exact native output and state")
    reference = torch.randn(4, 4, 2, 2048, generator=generator) * .02
    reference[0, 0].zero_()
    reference[1, 2].zero_()
    mixture = reference.sum(dim=1)
    flags = torch.tensor([False, True, False, True])
    target = physical_context_teacher(teacher, mixture, kind="c91", warmup_samples=1024)

    def gradients():
        model.zero_grad(set_to_none=True)
        output = render_scored_context(model, mixture, warmup_samples=1024, carry_state=True)
        require(output.initial_state_detached and torch.equal(output.physical_mixture, mixture[..., 1024:]),
                "Native gradient fixture lost alignment")
        terms = drum_emphasized_objective(output.raw, output.deployed, reference[..., 1024:], target, flags)
        terms.total.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params.values()),
                "Missing or nonfinite mixer gradients")
        return {name: {"max_absolute": float(p.grad.abs().max()), "nonzero": bool((p.grad != 0).any())}
                for name, p in params.items()}

    initial_gradients = gradients()
    require(all(initial_gradients[n]["nonzero"] for n in ("out.weight", "out.bias"))
            and all(initial_gradients[n]["max_absolute"] == 0 for n in ("local.weight", "local.bias")),
            "Zero-initialized branch gradients differ from the expected two-stage activation")
    # Fixed nonzero coefficients test the mechanism without an optimizer.
    # They are synthetic fixtures, not learned or retained candidate weights.
    with torch.no_grad():
        wrapper.mixer.out.weight.copy_(torch.randn(wrapper.mixer.out.weight.shape, generator=generator) * .003)
    active_gradients = gradients()
    require(all(row["nonzero"] for row in active_gradients.values()), "An active mixer parameter is disconnected")
    model.zero_grad(set_to_none=True)
    with torch.inference_mode():
        output = wrapper.mixer(logits)
        future = logits.clone()
        future[:, 3:] += .1
        changed_future = wrapper.mixer(future)
        require(torch.equal(output[:, :3], changed_future[:, :3])
                and not torch.equal(output[:, 3:], changed_future[:, 3:]), "Mixer uses future frames")
        shaped = logits.reshape(2, 7, 2, 513, 2, 4)
        swapped = wrapper.mixer(shaped.flip(2).reshape_as(logits)).reshape_as(shaped)
        require(torch.equal(swapped, output.reshape_as(shaped).flip(2)), "Stereo weight sharing differs")
        perturbed = shaped.clone()
        perturbed[:, :, :, 257] += .1
        localized = wrapper.mixer(perturbed.reshape_as(logits)).reshape_as(shaped)
        keep = torch.ones(513, dtype=torch.bool)
        keep[256:259] = False
        require(torch.equal(localized[:, :, :, keep], output.reshape_as(shaped)[:, :, :, keep])
                and not torch.equal(localized[:, :, :, 256], output.reshape_as(shaped)[:, :, :, 256])
                and not torch.equal(localized[:, :, :, 258], output.reshape_as(shaped)[:, :, :, 258]),
                "Frequency support is not the intended three bins")
        native = model.render(audio)
        future_audio = audio.clone()
        future_audio[..., 1024:] += .1
        native_future = model.render(future_audio)
        require(torch.equal(native.raw[..., :1024], native_future.raw[..., :1024])
                and torch.equal(native.deployed[..., :1024], native_future.deployed[..., :1024]),
                "Nonzero mixer adds future-audio dependence")
        state, raw, deployed = None, [], []
        for chunk in audio.split(128, dim=-1):
            piece = model.render(chunk, state)
            state = piece.state
            raw.append(piece.raw)
            deployed.append(piece.deployed)
        raw_error = float((torch.cat(raw, dim=-1) - native.raw).abs().max())
        deployed_error = float((torch.cat(deployed, dim=-1) - native.deployed).abs().max())
        closure = float((native.deployed.sum(dim=1) - native.delayed_mixture).abs().max())
        require(raw_error <= 1e-6 and deployed_error <= 1e-6 and closure <= 1e-6
                and all(a.shape == b.shape for a, b in zip(state, baseline.state, strict=True))
                and not torch.equal(native.raw, baseline.raw), "Streaming, closure, state shape or nonzero effect differs")
    print({"event": "identity_causality_locality_gradients_pass", "parameters": 536}, flush=True)

    # Paired process-local CPU timing is a feasibility observation. It is not
    # the native M4 callback benchmark and has concurrent host workloads.
    timing_audio = torch.randn(1, 2, 4096, generator=generator) * .03
    timing = []
    with torch.inference_mode():
        for pair in range(8):
            times = {}
            order = ("baseline", "mixer") if pair % 2 == 0 else ("mixer", "baseline")
            for kind in order:
                model.to_spec_masks = original_head if kind == "baseline" else wrapper
                state = model.initial_state(1)
                start = time.perf_counter()
                for chunk in timing_audio.split(128, dim=-1):
                    state = model.render(chunk, state).state
                times[kind] = (time.perf_counter() - start) * 1000 / 32
            if pair:
                timing.append({"pair": pair, "order": list(order), "milliseconds_per_hop": times,
                               "delta_ms": times["mixer"] - times["baseline"]})
    model.to_spec_masks = original_head
    require(fingerprint == state_sha256(model.state_dict())
            and state_sha256(teacher.state_dict()) == teacher_identity["model_state_sha256"]
            and all(p.grad is None for p in model.parameters())
            and all(not p.requires_grad and p.grad is None for p in teacher.parameters())
            and torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized(),
            "CPU preparation changed the parent, teacher, RNG or device scope")
    verify_inputs(plan)
    result = {"schema": "latency58-local-mask-mixer-functional-result-v1", "status": "pass",
              "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"],
              "source_bindings_unchanged": True, "version": VERSION, "parent_model_state_sha256": fingerprint,
              "additional_parameters": 536, "parameter_tensors": 4,
              "extra_convolution_macs_per_stereo_hop": 2 * 513 * (8 * 16 * 3 + 16 * 8),
              "zero_initial_native_output_and_states_exact": True, "stereo_swap_equivariant": True,
              "additional_time_context_samples": 0, "new_streaming_states": 0,
              "frequency_kernel_bins": 3, "future_frame_and_audio_invariance_passed": True,
              "initial_gradients": initial_gradients, "active_fixture_gradients": active_gradients,
              "literal_raw_max_abs": raw_error, "literal_deployed_max_abs": deployed_error,
              "closure_max_abs": closure, "paired_cpu_timing": timing,
              "median_paired_delta_ms": statistics.median(r["delta_ms"] for r in timing),
              "median_cpu_ms_per_hop": {k: statistics.median(r["milliseconds_per_hop"][k] for r in timing)
                                        for k in ("baseline", "mixer")},
              "parent_teacher_and_rng_unchanged": True, "optimizer_instances": 0,
              "training_updates_executed": 0, "checkpoint_written": False,
              "cuda_initialized": False, "validation_material_used": False, "quality_selected": False,
              "elapsed_seconds": time.monotonic() - began,
              "limitations": ["Synthetic CPU FP32 fixture; no evidence of SDR or audible improvement.",
                              "No full-crop BF16 GPU, ONNX or native M4 qualification.",
                              "CPU timings include this process and concurrent workloads; no M4 deadline claim.",
                              "The future trial parent and training recipe are not selected by this fixture."]}
    write(out / "result.json", result)
    print({"status": "pass", "elapsed_seconds": result["elapsed_seconds"],
           "median_paired_delta_ms": result["median_paired_delta_ms"]}, flush=True)


if __name__ == "__main__":
    main()
