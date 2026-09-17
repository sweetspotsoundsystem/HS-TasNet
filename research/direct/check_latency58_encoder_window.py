"""CPU-only check of the proposed encoder/window conversion; no training.

The independent cosine/sine matrix checks every input coordinate. Actual
FP32 FFT/linear calls then check impulses, tones, random frames and music.
This is neither a separation-quality test nor a new model checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

from research.direct.latency58_checkpoint import require, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Plan changed")
    plan = json.loads(args.plan.read_text())
    require(plan["schema"] == "latency58-encoder-window-check-plan-v1"
            and all(sha(p) == h for p, h in plan["source_bindings"].items()), "Check source or parent changed")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and all(os.environ.get(n) == "1"
            for n in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CUDA-hidden CPU1")
    output = Path(plan["output"])
    require(output.parent.is_dir() and not output.exists(), "Use a prepared directory and fresh result path")
    import torch
    import soundfile as sf
    from torch.nn import functional as F
    from research.direct.latency58 import Latency58Model
    from research.direct.latency58_checkpoint import load_model_state
    from research.direct.latency58_evaluate import model_state_sha256
    from research.direct.latency58_encoder_window import asymmetric_windows, convert_encoder_weight

    require(torch.__version__ == plan["torch_version"], "CPU Torch runtime differs")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model = Latency58Model.from_accepted().eval().requires_grad_(False)
    step = load_model_state(model, plan["checkpoint"])
    fingerprint = model_state_sha256(model)
    require(fingerprint == plan["model_state_sha256"] and step == plan["step"], "Loaded parent differs")
    rng = torch.get_rng_state().clone()
    started = time.monotonic()
    old = model.analysis_window
    new, synthesis = asymmetric_windows()
    original_weight = model.spec_encode.weight.detach().clone()
    with torch.inference_mode():
        converted = convert_encoder_weight(original_weight, old, new)
        product = new[768:] * synthesis
        # Compare the product with the mathematical prototype independently.
        # Torch's FP32 cosine calculation has its own rounding error, so its
        # difference is a separate diagnostic rather than the reference value.
        expected = torch.tensor([0.5 * (1 - math.cos(math.pi * n / 128))
                                 for n in range(256)], dtype=torch.float64)
        product_error = float((product.double() - expected).abs().max())
        torch_prototype_difference = float((product - torch.hann_window(256, periodic=True)).abs().max())
        overlap_error = float((product[:128] + product[128:] - 1).abs().max())
        require(product_error <= 2e-7 and overlap_error <= 2e-7, "Window pair fails its prototype/overlap identity")

        # Independent from the FFT-based conversion: explicit real DFT rows.
        n = torch.arange(1024, dtype=torch.float64)
        k = torch.arange(513, dtype=torch.float64)
        angles = 2 * torch.pi * k[:, None] * n[None] / 1024
        cosine, sine = angles.cos(), angles.sin()
        sine[0].zero_()
        sine[-1].zero_()
        def kernel(weight, window):
            w = weight.double().reshape(500, 2, 513, 2)
            return (w[..., 0] @ cosine - w[..., 1] @ sine) * window.double()
        old_kernel, new_kernel = kernel(original_weight, old), kernel(converted, new)
        difference = (new_kernel - old_kernel).abs()
        coordinate_error = float(difference.max())
        bounded_input_error = float(difference.sum(dim=(1, 2)).max())
        require(bounded_input_error <= 1e-4, "Converted linear map exceeds the predeclared bounded-input error")

        rows = []
        def check(name, frames):
            require(frames.ndim == 3 and frames.shape[1:] == (2, 1024) and frames.dtype == torch.float32,
                    "Feature fixture shape/dtype differs")
            def features(window, weight):
                spectrum = torch.fft.rfft(frames * window, n=1024, dim=-1)
                packed = torch.view_as_real(spectrum).flatten(1)
                return F.linear(packed, weight, model.spec_encode.bias)
            reference, candidate = features(old, original_weight), features(new, converted)
            require(torch.isfinite(reference).all().item() and torch.isfinite(candidate).all().item()
                    and torch.allclose(reference, candidate, atol=3e-5, rtol=1e-5),
                    "Actual FP32 features differ beyond the predeclared tolerance: " + name)
            rows.append({"case": name, "frames": frames.shape[0],
                         "max_abs_error": float((reference - candidate).abs().max()),
                         "reference_abs_max": float(reference.abs().max())})
        check("silence", torch.zeros(1, 2, 1024))
        check("stereo_dc", torch.tensor([0.125, -0.25])[None, :, None].expand(1, 2, 1024))
        check("nyquist", torch.tensor([0.125, -0.25])[None, :, None]
              * (1 - 2 * torch.arange(1024).remainder(2)).float()[None, None])
        generator = torch.Generator(device="cpu").manual_seed(20260907)
        check("random_nonzero_history", torch.randn((32, 2, 1024), generator=generator) * 0.1)
        tones = torch.stack([torch.sin(2 * torch.pi * frequency * n / 44100)
                             for frequency in (43.06640625, 61.735, 997.25, 22049.0)]).float() * 0.2
        check("on_and_off_bin_tones", torch.stack((tones, -tones), dim=1))
        for channel in range(2):
            impulses = torch.zeros(1024, 2, 1024)
            impulses[torch.arange(1024), channel, torch.arange(1024)] = 0.125
            check("all_impulse_positions_channel_" + str(channel), impulses)
            del impulses
        audio_path = Path(plan["audio"]["path"])
        require(sha(audio_path) == plan["audio"]["sha256"], "Music source changed")
        audio, rate = sf.read(audio_path, start=60 * 44100 - 896, stop=60 * 44100 + 1024,
                              dtype="float32", always_2d=True)
        require(rate == 44100 and audio.shape == (1920, 2), "Music frame context differs")
        frames = torch.from_numpy(audio.T.copy()).unfold(-1, 1024, 128).permute(1, 0, 2).contiguous()
        check("actions_real_nonzero_history", frames)

    require(torch.equal(original_weight, model.spec_encode.weight)
            and fingerprint == model_state_sha256(model) and torch.equal(rng, torch.get_rng_state())
            and not torch.cuda.is_initialized()
            and all(sha(p) == h for p, h in plan["source_bindings"].items()),
            "Check changed its model, RNG, CPU scope or inputs")
    report = {"schema": "latency58-encoder-window-check-v1", "status": "pass",
              "checkpoint": plan["checkpoint"], "model_state_sha256": fingerprint,
              "window_product_max_error": product_error, "overlap_sum_max_error": overlap_error,
              "window_product_difference_from_torch_fp32_hann": torch_prototype_difference,
              "independent_effective_kernel_max_error": coordinate_error,
              "independent_kernel_l1_bound_for_unit_bounded_stereo_input": bounded_input_error,
              "feature_checks": rows, "feature_atol": 3e-5, "feature_rtol": 1e-5,
              "cuda_initialized": False, "optimizer_instances": 0, "training_updates_executed": 0,
              "weights_saved": False, "model_unchanged": True, "source_bindings_unchanged": True,
              "source_bindings": plan["source_bindings"], "plan_sha256": args.plan_sha256,
              "torch_version": torch.__version__,
              "elapsed_seconds": time.monotonic() - started,
              "scope": "Window product and existing encoder features only; no new separation model, BF16, quality or native timing claim"}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
