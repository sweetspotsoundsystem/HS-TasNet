"""Bounded CPU functional checks for the separate hop128 research family.

No optimizer, training, CUDA, music render, or checkpoint write. Run with
CUDA hidden and one CPU thread; the caller retains this program's exit status.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve existing result")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(name) == "1" for name in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use the CUDA-hidden CPU1 environment")
    import torch
    from research.direct import latency58 as model_source

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(), "CPU fixture must not initialize CUDA")
    torch.manual_seed(20260907)
    start = time.monotonic()
    checks = {}
    hop = model_source.HOP

    # The reference is a physical input delayed by 128, with no separator and
    # no model state involved in construction of the expected waveform.
    for dtype, tolerance in ((torch.float32, 2e-6), (torch.float64, 2e-12)):
        window = torch.hann_window(1024, periodic=True, dtype=dtype)
        synth = model_source.Cropped256Synthesis(window)
        denominator = torch.tensor([
            (0.5 - 0.5 * math.cos(2 * math.pi * (768 + p) / 1024))
            * (0.5 - 0.5 * math.cos(2 * math.pi * p / 256))
            + (0.5 - 0.5 * math.cos(2 * math.pi * (896 + p) / 1024))
            * (0.5 - 0.5 * math.cos(2 * math.pi * (128 + p) / 256))
            for p in range(hop)], dtype=dtype)
        require(torch.allclose(synth.spectral_denominator, denominator, atol=2e-7, rtol=0),
                "Independent scalar denominator mismatch")
        samples = torch.arange(4 * hop, dtype=dtype)
        cases = [torch.randn(2, 4 * hop, dtype=dtype), torch.zeros(2, 4 * hop, dtype=dtype),
                 torch.ones(2, 4 * hop, dtype=dtype),
                 torch.cos(math.pi * samples).repeat(2, 1),
                 (0.3 * torch.sin(2 * math.pi * 43.7 * samples / 44100)
                  + 0.2 * torch.cos(2 * math.pi * 1003.2 * samples / 44100)).repeat(2, 1)]
        for phase in range(hop):
            impulse = torch.zeros(2, 4 * hop, dtype=dtype)
            impulse[0, hop + phase] = 1
            impulse[1, 2 * hop + phase] = -0.7
            cases.append(impulse)
        worst = 0.0
        for signal in cases:
            padded = torch.nn.functional.pad(signal, (896, hop))
            frames = padded.unfold(-1, 1024, hop)
            spectrum = torch.fft.rfft(frames * window, dim=-1)
            audio, _ = synth.spectral(spectrum, torch.zeros(2, hop, dtype=dtype))
            expected = torch.nn.functional.pad(signal, (hop, 0))
            error = float((audio - expected).abs().max())
            worst = max(worst, error)
            require(error <= tolerance, "Unity physical alignment or overlap reconstruction failed")
        checks[str(dtype)] = {"unity_cases": len(cases), "maximum_sample_error": worst,
                              "spectral_denominator_min": float(synth.spectral_denominator.min()),
                              "spectral_denominator_max": float(synth.spectral_denominator.max())}

    rng = torch.get_rng_state().clone()
    model = model_source.Latency58Model.from_accepted().eval()
    require(torch.equal(torch.get_rng_state(), rng), "Initializer changed caller RNG")
    original = torch.load(model_source.ACCEPTED_CHECKPOINT, map_location="cpu", weights_only=True)["model"]
    for name, value in model.state_dict().items():
        if name.startswith("synthesis."):
            continue
        expected = original[name][..., -256:] if name == "waveform_decoder_weight" else original[name]
        require(torch.equal(value, expected), "Accepted tensor transfer differs: " + name)
    del original
    before = {name: value.clone() for name, value in model.state_dict().items()}
    require(len(tuple(model.parameters())) == 21 and len(tuple(model.buffers())) == 5,
            "Model tensor inventory differs")
    audio = torch.randn(1, 2, 5 * hop) * 0.15
    initial = model.initial_state(1)
    nonzero = model_source.Latency58State(
        torch.randn_like(initial.audio_history) * 0.1,
        torch.randn_like(initial.fusion_hidden) * model_source.PUBLIC_FUSION_SCALE * 0.01,
        torch.randn_like(initial.spectral_numerator_tail) * 0.01,
        torch.randn_like(initial.waveform_tail) * 0.01)
    maximum = 0.0
    with torch.no_grad():
        for state in (initial, nonzero):
            grouped = model.render(audio, state)
            current = state
            chunks = []
            for offset in range(0, audio.shape[-1], hop):
                out = model.render(audio[..., offset:offset + hop], current)
                chunks.append(out)
                current = out.state
            for field in ("raw", "deployed", "spectral", "waveform", "delayed_mixture"):
                literal = torch.cat([getattr(item, field) for item in chunks], dim=-1)
                value = getattr(grouped, field)
                maximum = max(maximum, float((literal - value).abs().max()))
                require(torch.allclose(literal, value, atol=5e-5, rtol=5e-5),
                        "Grouped/literal output mismatch: " + field)
            for index, (left, right) in enumerate(zip(grouped.state, current, strict=True)):
                # Compare decoded GRU state in physical units.
                scale = model_source.PUBLIC_FUSION_SCALE if index == 1 else 1.0
                require(torch.allclose(left / scale, right / scale, atol=5e-5, rtol=5e-5),
                        "Grouped/literal carried state mismatch")
            require(float((grouped.deployed.sum(dim=1) - grouped.delayed_mixture).abs().max()) <= 1e-6,
                    "Four-stem reconstruction failed")

        for length in (1, 127, 128, 129, 255, 256, 257, 641):
            real = torch.randn(1, 2, length) * 0.1
            real[..., -1] = torch.tensor([0.37, -0.29])
            padded = torch.nn.functional.pad(real, (0, (-length) % hop))
            state, delayed, stems = model.initial_state(1), [], []
            for offset in range(0, padded.shape[-1], hop):
                out = model.render(padded[..., offset:offset + hop], state)
                state = out.state
                delayed.append(out.delayed_mixture)
                stems.append(out.deployed)
            # Exactly one graph flush. Compare its API with an explicit zero.
            tail, tail_state = model.flush(state)
            zero = model.render(torch.zeros(1, 2, hop), state)
            require(torch.equal(tail, zero.deployed), "Flush differs from one zero graph hop")
            require(all(torch.equal(a, b) for a, b in zip(tail_state, zero.state, strict=True)),
                    "Flush state differs from one zero graph hop")
            delayed.append(zero.delayed_mixture)
            stems.append(tail)
            physical = torch.cat(delayed, dim=-1)[..., hop:hop + length]
            deployed = torch.cat(stems, dim=-1)[..., hop:hop + length]
            require(torch.equal(physical, real), "Partial EOF lost or shifted real input")
            require(torch.isfinite(deployed).all().item()
                    and float((deployed.sum(dim=1) - real).abs().max()) <= 1e-6,
                    "Partial EOF four-stem closure failed")

        prefix = audio[..., :2 * hop]
        a, b = model.render(prefix), model.render(prefix)
        require(torch.equal(a.deployed, b.deployed)
                and all(torch.equal(x, y) for x, y in zip(a.state, b.state, strict=True)),
                "Reset replay is not exact")
        future_a, future_b = audio.clone(), audio.clone()
        future_b[..., 2 * hop:] *= -2
        a, b = model.render(future_a), model.render(future_b)
        require(torch.equal(a.deployed[..., :2 * hop], b.deployed[..., :2 * hop]),
                "Prefix causality failed")
        # State serialization needs no new 111 MB checkpoint artifact.
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = torch.load(buffer, map_location="cpu", weights_only=True)
        model.load_state_dict(restored, strict=True)
        require(torch.equal(model.render(audio).deployed, a.deployed), "Saved tensor replay differs")
        del restored, buffer

    model.zero_grad(set_to_none=True)
    grad_audio = (torch.randn(1, 2, 3 * hop) * 0.1).requires_grad_()
    state = model_source.Latency58State(*(x.detach().clone().requires_grad_() for x in nonzero))
    outputs = []
    for offset in range(0, grad_audio.shape[-1], hop):
        out = model.render(grad_audio[..., offset:offset + hop], state)
        state = out.state
        if offset == 0:
            carried = state
            for value in carried:
                value.retain_grad()
        outputs.append(out.raw)
    last = model.render(torch.zeros(1, 2, hop), state)
    outputs.append(last.raw)
    real_output = torch.cat(outputs, dim=-1)[..., hop:hop + grad_audio.shape[-1]]
    real_output.square().mean().backward()
    require(grad_audio.grad is not None and torch.isfinite(grad_audio.grad).all().item()
            and float(grad_audio.grad[..., -1].abs().sum()) > 0,
            "Final real sample lost its gradient through the final flush")
    require(all(p.grad is not None and torch.isfinite(p.grad).all().item()
                and float(p.grad.abs().sum()) > 0 for p in model.parameters()),
            "A learned tensor has missing, nonfinite or zero gradients")
    require(all(value.grad is not None and torch.isfinite(value.grad).all().item()
                and float(value.grad.abs().sum()) > 0 for value in carried),
            "A carried state lost its finite nonzero cross-call gradient")
    require(all(torch.equal(value, before[name]) for name, value in model.state_dict().items()),
            "Functional fixture mutated model weights or buffers")
    require(not torch.cuda.is_initialized(), "Functional fixture initialized CUDA")
    checks["model"] = {"parameters": sum(p.numel() for p in model.parameters()),
                       "parameter_tensors": 21, "buffer_tensors": 5,
                       "grouped_literal_max_sample_error": maximum,
                       "partial_eof_cases": 8, "gradient_parameter_tensors": 21,
                       "gradient_carried_state_tensors": 4,
                       "training_updates": 0, "optimizer_instances": 0,
                       "prefix_causality": True, "reset_replay_exact": True,
                       "serialization_exact": True, "model_unchanged": True}
    result = {"status": "pass", "family": model_source.VERSION,
              "torch_version": torch.__version__, "elapsed_seconds": time.monotonic() - start,
              "graph_delay_samples": hop, "intended_plugin_latency_samples": 2 * hop,
              "plugin_latency_qualified": False, "cuda_initialized": False,
              "model_source_sha256": model_source.file_sha256(Path(model_source.__file__)),
              "fixture_source_sha256": model_source.file_sha256(Path(__file__)),
              "accepted_checkpoint_sha256": model_source.file_sha256(model_source.ACCEPTED_CHECKPOINT),
              "checks": checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
