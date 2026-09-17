"""CPU functional and parent-isolation checks for the unqualified asymmetric model."""
from __future__ import annotations

import argparse
import io
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
    require(sha(args.plan) == args.plan_sha256, "Functional plan changed")
    plan = json.loads(args.plan.read_text())
    require(plan["schema"] == "latency58-asymmetric-functional-plan-v1"
            and all(sha(path) == digest for path, digest in plan["source_bindings"].items()),
            "Functional plan or inputs differ")
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve prior result")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and all(os.environ.get(key) == "1" for key in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CUDA-hidden CPU1")
    import torch
    from research.direct.latency58 import Cropped256Synthesis, Latency58Model, PUBLIC_FUSION_SCALE
    from research.direct.latency58_checkpoint import load_model_state
    from research.direct.latency58_evaluate import model_state_sha256
    from research.direct.latency58_asymmetric import Latency58AsymmetricModel, AsymmetricState, AsymmetricSynthesis, VERSION
    from research.direct.latency58_encoder_window import asymmetric_windows

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260907)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    checks = {}
    for dtype, tolerance in ((torch.float32, 2e-6), (torch.float64, 2e-12)):
        window, spectral = asymmetric_windows(dtype=dtype)
        synthesis = AsymmetricSynthesis(window, spectral)
        independent = torch.tensor([.5 * (1 - math.cos(math.pi * p / 128))
                                    for p in range(256)], dtype=torch.float64)
        product_error = float((window[768:].double() * spectral.double() - independent).abs().max())
        overlap_error = float((synthesis.spectral_denominator - 1).abs().max())
        require(product_error <= (2e-7 if dtype == torch.float32 else 2e-15)
                and overlap_error <= (2e-7 if dtype == torch.float32 else 2e-15),
                "Independent analysis-synthesis product or overlap differs")
        t = torch.arange(512, dtype=dtype)
        cases = [torch.randn(2, 512, dtype=dtype), torch.zeros(2, 512, dtype=dtype),
                 torch.ones(2, 512, dtype=dtype), torch.cos(math.pi * t).repeat(2, 1),
                 (.3 * torch.sin(2 * math.pi * 43.7 * t / 44100)
                  + .2 * torch.cos(2 * math.pi * 1003.2 * t / 44100)).repeat(2, 1)]
        for phase in range(128):
            impulse = torch.zeros(2, 512, dtype=dtype)
            impulse[0, 128 + phase], impulse[1, 256 + phase] = 1, -.7
            cases.append(impulse)
        maximum = 0.0
        for signal in cases:
            frames = torch.nn.functional.pad(signal, (896, 128)).unfold(-1, 1024, 128)
            spectrum = torch.fft.rfft(frames * window, n=1024, dim=-1)
            actual, _ = synthesis.spectral(spectrum, torch.zeros(2, 128, dtype=dtype))
            expected = torch.nn.functional.pad(signal, (128, 0))
            error = float((actual - expected).abs().max())
            maximum = max(maximum, error)
            require(error <= tolerance, "Unity reconstruction or physical alignment differs")
        checks[str(dtype)] = {"unity_cases": len(cases), "maximum_sample_error": maximum,
                              "independent_product_max_error": product_error, "overlap_max_error": overlap_error,
                              "denominator_min": float(synthesis.spectral_denominator.min()),
                              "denominator_max": float(synthesis.spectral_denominator.max())}

    parent = Latency58Model.from_accepted().eval().requires_grad_(False)
    parent_step = load_model_state(parent, plan["checkpoint"])
    parent_hash = model_state_sha256(parent)
    rng = torch.get_rng_state().clone()
    model = Latency58AsymmetricModel.from_hann_model(parent).eval()
    require(torch.equal(rng, torch.get_rng_state()), "Window variant construction changed RNG")
    require(len(list(model.parameters())) == 21 and len(list(model.buffers())) == 6, "Tensor inventory differs")
    for name, value in parent.state_dict().items():
        if name not in ("analysis_window", "synthesis.spectral_denominator", "spec_encode.weight"):
            require(torch.equal(value, model.state_dict()[name]), "A copied parent tensor differs: " + name)
    require(parent_hash == plan["parent_model_state_sha256"] and parent_step == plan["parent_step"],
            "Frozen parent state or update count differs")
    require(not torch.equal(parent.spec_encode.weight, model.spec_encode.weight), "Encoder was not transferred")
    require(model.architecture_metadata["source_order"] == ["drums", "bass", "vocals", "other"]
            and torch.equal(model.output_source_scales, parent.output_source_scales), "Native gains/order differ")
    model_hash = model_state_sha256(model)
    audio = torch.randn(1, 2, 640) * .1
    initial = model.initial_state(1)
    nonzero = AsymmetricState(*(torch.randn_like(value) * scale for value, scale in zip(initial,
                            (.04, .01 * PUBLIC_FUSION_SCALE, .001, .001), strict=True)))
    grouped_error = closure = 0.0
    with torch.no_grad():
        for state in (initial, nonzero):
            grouped = model.render(audio, state)
            current, rows = state, []
            for offset in range(0, 640, 128):
                item = model.render(audio[..., offset:offset + 128], current)
                current = item.state
                rows.append(item)
            for field in ("raw", "deployed", "spectral", "waveform", "delayed_mixture"):
                literal = torch.cat([getattr(row, field) for row in rows], dim=-1)
                error = float((literal - getattr(grouped, field)).abs().max())
                grouped_error = max(grouped_error, error)
                require(torch.allclose(literal, getattr(grouped, field), atol=5e-5, rtol=5e-5),
                        "Literal/grouped output differs: " + field)
            for index, (a, b) in enumerate(zip(grouped.state, current, strict=True)):
                scale = PUBLIC_FUSION_SCALE if index == 1 else 1.0
                require(torch.allclose(a / scale, b / scale, atol=5e-5, rtol=5e-5), "Literal/grouped state differs")
            closure = max(closure, float((grouped.deployed.sum(1) - grouped.delayed_mixture).abs().max()))

        # One FFT feeds the changed carrier and the transferred feature encoder.
        from unittest.mock import patch
        with patch("torch.fft.rfft", wraps=torch.fft.rfft) as fft_call:
            model.render(audio)
            require(fft_call.call_count == 1, "The model duplicated the analysis FFT")
        from research.direct.latency58 import Latency58State
        shared_errors = {}
        for state in (initial, nonzero):
            old, new = parent.render(audio, Latency58State(*state)), model.render(audio, state)
            for name, a, b in (("waveform", old.waveform, new.waveform),
                               ("hidden_physical", old.state[1] / PUBLIC_FUSION_SCALE,
                                new.state[1] / PUBLIC_FUSION_SCALE),
                               ("waveform_tail", old.state[3], new.state[3])):
                shared_errors[name] = max(shared_errors.get(name, 0.), float((a - b).abs().max()))
                require(torch.allclose(a, b, atol=3e-5, rtol=1e-5), "Transferred feature path differs: " + name)
            require(torch.equal(old.state[0], new.state[0]), "Audio history changed")
        # Restore the original analysis, spectral synthesis and encoder together
        # in this disposable object. Every output/state must match the parent.
        restored_names = ("analysis_window", "synthesis.spectral_window", "synthesis.spectral_denominator",
                          "spec_encode.weight")
        saved = {name: model.state_dict()[name].clone() for name in restored_names}
        model.analysis_window.copy_(parent.analysis_window)
        model.synthesis.spectral_window.copy_(parent.synthesis.window)
        model.spec_encode.weight.copy_(parent.spec_encode.weight)
        model.synthesis.rebuild_denominator(model.analysis_window)
        for state in (initial, nonzero):
            old, new = parent.render(audio, Latency58State(*state)), model.render(audio, state)
            require(all(torch.equal(getattr(old, field), getattr(new, field)) for field in
                        ("raw", "deployed", "spectral", "waveform", "delayed_mixture"))
                    and all(torch.equal(a, b) for a, b in zip(old.state, new.state, strict=True)),
                    "Restored Hann implementation does not exactly reproduce parent")
        for name, value in saved.items():
            model.state_dict()[name].copy_(value)
        for length in (1, 127, 128, 129, 255, 256, 257, 641):
            real = torch.randn(1, 2, length) * .1
            real[..., -1] = torch.tensor([.37, -.29])
            padded = torch.nn.functional.pad(real, (0, (-length) % 128))
            state, pieces = model.initial_state(1), []
            for offset in range(0, padded.shape[-1], 128):
                row = model.render(padded[..., offset:offset + 128], state)
                state = row.state
                pieces.append(row.deployed)
            tail, final = model.flush(state)
            zero = model.render(torch.zeros(1, 2, 128), state)
            require(torch.equal(tail, zero.deployed) and all(torch.equal(a, b) for a, b in
                    zip(final, zero.state, strict=True)), "Flush differs from one explicit zero hop")
            pieces.append(tail)
            recovered = torch.cat(pieces, -1)[..., 128:128 + length].sum(1)
            closure = max(closure, float((recovered - real).abs().max()))
        changed = audio.clone()
        changed[..., 256:] *= -2
        a, b = model.render(audio), model.render(changed)
        require(torch.equal(a.deployed[..., :256], b.deployed[..., :256]), "Prefix causality failed")
        replay = model.render(audio)
        require(torch.equal(a.deployed, replay.deployed) and all(torch.equal(x, y) for x, y in
                zip(a.state, replay.state, strict=True)), "Complete reset replay differs")
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        model.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True), strict=True)
        require(torch.equal(a.deployed, model.render(audio).deployed), "Serialized tensor replay differs")
        for receiver, wrong_state in ((model, parent.initial_state(1)), (parent, initial)):
            try:
                receiver.render(audio, wrong_state)
            except ValueError:
                pass
            else:
                raise RuntimeError("Cross-family state was accepted")
    for bad_audio, bad_state in ((audio.double(), initial),
                                 (audio, AsymmetricState(*(v.double() for v in initial))),
                                 (audio[..., :127], initial)):
        try:
            model.render(bad_audio, bad_state)
        except ValueError:
            pass
        else:
            raise RuntimeError("Invalid audio/state contract was accepted")
    require(torch.__version__ == plan["torch_version"], "Unexpected Torch runtime")
    require(closure <= 1e-6, "Four-stem physical reconstruction failed")
    model.zero_grad(set_to_none=True)
    grad_audio = (torch.randn(1, 2, 384) * .1).requires_grad_()
    state, pieces = nonzero, []
    for offset in range(0, 384, 128):
        item = model.render(grad_audio[..., offset:offset + 128], state)
        state = item.state
        if offset == 0:
            carried = state
            for value in carried:
                value.retain_grad()
        pieces.append(item.raw)
    pieces.append(model.render(torch.zeros(1, 2, 128), state).raw)
    torch.cat(pieces, -1)[..., 128:512].square().mean().backward()
    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().sum()) > 0
                for p in model.parameters()), "Missing, zero or nonfinite parameter gradient")
    require(all(v.grad is not None and bool(torch.isfinite(v.grad).all()) and float(v.grad.abs().sum()) > 0
                for v in carried), "Missing, zero or nonfinite cross-call state gradient")
    require(grad_audio.grad is not None and bool(torch.isfinite(grad_audio.grad).all())
            and float(grad_audio.grad[..., -1].abs().sum()) > 0, "Final real input lost its flush gradient")
    require(model_state_sha256(parent) == parent_hash and model_state_sha256(model) == model_hash
            and all(sha(path) == digest for path, digest in plan["source_bindings"].items())
            and not torch.cuda.is_initialized(), "Parent, variant, bound inputs or CPU scope changed")
    checks["model"] = {"parameter_tensors": 21, "buffer_tensors": 6, "unchanged_learned_parent_tensors_exact": 20,
                       "encoder_weight_transformed": True, "forward_analysis_ffts_per_group": 1,
                       "shared_path_max_errors": shared_errors, "restored_hann_parent_replay_exact": True,
                       "invalid_dtype_and_geometry_rejected": True, "cross_family_states_rejected": True,
                       "grouped_literal_max_sample_error": grouped_error, "reconstruction_max_abs": closure,
                       "partial_eof_cases": 8, "one_flush_final_real_input_gradient": True,
                       "finite_nonzero_parameter_gradients": 21, "finite_nonzero_carried_state_gradients": 4,
                       "prefix_causality": True, "reset_replay_exact": True, "serialization_exact": True}
    result = {"status": "pass", "family": VERSION, "plan_sha256": args.plan_sha256,
              "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
              "checkpoint": plan["checkpoint"], "parent_step": parent_step, "parent_model_state_sha256": parent_hash,
              "model_state_sha256": model_hash, "checks": checks, "elapsed_seconds": time.monotonic() - started,
              "torch_version": torch.__version__, "graph_alignment_samples": 128,
              "intended_plugin_latency_samples": 256, "plugin_latency_qualified": False,
              "training_updates_executed": 0, "optimizer_instances": 0, "cuda_initialized": False}
    with (out / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
