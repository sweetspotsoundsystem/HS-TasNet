"""CPU functional and parent-isolation checks for the unqualified plateau model."""
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
    require(plan["schema"] == "latency58-plateau-functional-plan-v1"
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
    from research.direct.latency58_plateau import Latency58PlateauModel, PlateauState, VERSION, carrier_window

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260907)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    checks = {}
    for dtype, tolerance in ((torch.float32, 2e-6), (torch.float64, 2e-12)):
        window = carrier_window(dtype=dtype)
        synthesis = Cropped256Synthesis(window)
        denominator = torch.tensor([1.0 - math.cos(math.pi * p / 256.0) ** 2
                                    + math.cos(math.pi * p / 256.0) ** 4 for p in range(128)], dtype=dtype)
        require(torch.allclose(synthesis.spectral_denominator, denominator, atol=2e-7, rtol=0),
                "Independent plateau overlap formula differs")
        require(float(synthesis.spectral_denominator.min()) >= 0.75 - 2e-7
                and float(synthesis.spectral_denominator.max()) <= 1 + 2e-7, "Overlap bounds failed")
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
                              "denominator_min": float(synthesis.spectral_denominator.min()),
                              "denominator_max": float(synthesis.spectral_denominator.max())}

    parent = Latency58Model.from_accepted().eval().requires_grad_(False)
    parent_step = load_model_state(parent, plan["checkpoint"])
    parent_hash = model_state_sha256(parent)
    rng = torch.get_rng_state().clone()
    model = Latency58PlateauModel.from_hann_model(parent).eval()
    require(torch.equal(rng, torch.get_rng_state()), "Window variant construction changed RNG")
    require(len(list(model.parameters())) == 21 and len(list(model.buffers())) == 6, "Tensor inventory differs")
    for name, value in parent.state_dict().items():
        if name != "synthesis.spectral_denominator":
            require(torch.equal(value, model.state_dict()[name]), "A copied parent tensor differs: " + name)
    model_hash = model_state_sha256(model)
    audio = torch.randn(1, 2, 640) * .1
    initial = model.initial_state(1)
    nonzero = PlateauState(*(torch.randn_like(value) * scale for value, scale in zip(initial,
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

        # The extra FFT must not change the learned feature or waveform path.
        old, new = parent.render(audio), model.render(audio)
        require(torch.equal(old.waveform, new.waveform)
                and all(torch.equal(old.state[i], new.state[i]) for i in (0, 1, 3)),
                "Changing only the carrier affected a shared feature/state/waveform path")
        # Restore the original carrier in this disposable object: every output
        # and state must then match the parent exactly, including nonzero state.
        saved_window, saved_divisor = model.carrier_window.clone(), model.synthesis.spectral_denominator.clone()
        model.carrier_window.copy_(model.analysis_window)
        model.synthesis.rebuild_denominator(model.carrier_window)
        for state in (initial, nonzero):
            from research.direct.latency58 import Latency58State
            old, new = parent.render(audio, Latency58State(*state)), model.render(audio, state)
            require(all(torch.equal(getattr(old, field), getattr(new, field)) for field in
                        ("raw", "deployed", "spectral", "waveform", "delayed_mixture"))
                    and all(torch.equal(a, b) for a, b in zip(old.state, new.state, strict=True)),
                    "Same-carrier implementation does not exactly reproduce parent")
        model.carrier_window.copy_(saved_window)
        model.synthesis.spectral_denominator.copy_(saved_divisor)
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
    checks["model"] = {"parameter_tensors": 21, "buffer_tensors": 6, "all_learned_parent_tensors_exact": True,
                       "shared_waveform_history_hidden_and_waveform_tail_exact": True,
                       "same_carrier_parent_replay_exact": True, "cross_family_states_rejected": True,
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
