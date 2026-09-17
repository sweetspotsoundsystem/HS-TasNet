"""Check new causal features, inherited identity, literal state and saved training."""
import tempfile
from pathlib import Path

import torch
from torch.nn import functional as F

from research.direct.run_latency58_quality import PHASE, read, require
from research.direct.train_latency58 import state_sha256
from research.direct.latency58 import FEATURE_HISTORY, PUBLIC_FUSION_SCALE
from research.direct.latency58_full_magnitude_checkpoint import load_model as load_parent
from research.direct.latency58_long_magnitude import Latency58LongMagnitudeModel, LongMagnitudeState, ADAPTER
from research.direct.latency58_long_magnitude_checkpoint import load_model, audit_live, save_generation


def check():
    torch.set_num_threads(1)
    binding = read(PHASE / "full-magnitude-001/result.json")["checkpoint"]
    parent, _ = load_parent(binding)
    parent_sha = state_sha256(parent.state_dict())
    model = Latency58LongMagnitudeModel.from_parent(parent)
    initial_sha = state_sha256(model.state_dict())
    generator = torch.Generator().manual_seed(20261011)
    audio = .03 * torch.randn(1, 2, 53 * 128, generator=generator)
    native_state = parent.initial_state(1)
    for value in native_state:
        value.copy_(.01 * torch.randn(value.shape, generator=generator))
    native_state.fusion_hidden.mul_(PUBLIC_FUSION_SCALE)
    history = torch.cat((.03 * torch.randn(1, 2, 3072, generator=generator), native_state.audio_history), -1)
    state = LongMagnitudeState(history, *native_state[1:])
    with torch.inference_mode():
        first, second = parent.render(audio, native_state), model.render(audio, state)
        require(torch.equal(first.raw, second.raw) and torch.equal(first.deployed, second.deployed)
                and torch.equal(first.state.audio_history, second.state.audio_history[..., -FEATURE_HISTORY:])
                and all(torch.equal(a, b) for a, b in zip(first.state[1:], second.state[1:], strict=True)),
                "Zero long projection changed inherited execution")
        joined = torch.cat((state.audio_history, audio), -1)
        actual = model.long_features(joined).double()
        # Independently form each trailing frame, its bins, and its band powers.
        expected = []
        for index in range(53):
            frame = joined[..., index * 128:index * 128 + 4096].double()
            fft = torch.fft.rfft(frame * model.long_window.double(), n=4096)
            power = fft.real.square() + fft.imag.square()
            bands = torch.stack([power[..., i] for i in range(128)] +
                                [power[..., i:i+8].mean(-1) for i in range(128, 2048, 8)] +
                                [power[..., 2048]], -1)
            scale = power.mean((1, 2), keepdim=True).clamp_min(1e-8).sqrt()
            expected.append(torch.log1p((bands + 1e-12).sqrt() / scale).flatten(1))
        feature_error = float((actual - torch.stack(expected, 1)).abs().max())
        require(feature_error < 2e-6, "Long feature frame alignment or normalization differs")
    model.train().requires_grad_(True)
    frozen = {n: v.clone() for n, v in model.named_buffers()}
    before = {n: v.detach().clone() for n, v in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, foreach=False)
    audit_live(model, optimizer, 0, frozen)
    output = model.render(audio)
    target = .02 * torch.randn(output.deployed.shape, generator=generator)
    loss = (output.deployed - target).square().mean() + .25 * (output.raw - target).square().mean()
    loss.backward()
    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and p.grad.abs().max() > 0
                for p in model.parameters()), "A neural tensor has no finite nonzero gradient")
    optimizer.step()
    audit_live(model, optimizer, 1, frozen)
    changed = [n for n, p in model.named_parameters() if not torch.equal(before[n], p)]
    require(len(changed) == 23 and ADAPTER in changed, "Every neural tensor must update")
    del before, output, loss
    model.eval()
    with torch.inference_mode():
        grouped = model.render(audio, state)
        replay_state, rows = state, []
        for chunk in audio.split(128, -1):
            result = model.render(chunk, replay_state)
            replay_state = result.state
            rows.append(result.deployed)
        error = (grouped.deployed - torch.cat(rows, -1)).abs()
        wave_max, wave_rms = float(error.max()), float(error.square().mean().sqrt())
        state_errors = {name: float((a-b).abs().max()) / (PUBLIC_FUSION_SCALE if name == "fusion_hidden" else 1.)
                        for name, a, b in zip(state._fields, grouped.state, replay_state, strict=True)}
        require(wave_max < 1e-4 and wave_rms < 1e-5 and max(state_errors.values()) < 5e-4,
                "Literal and grouped physical states or waveforms differ")
        changed_future = audio.clone()
        changed_future[..., 31 * 128:] += .1 * torch.randn(changed_future[..., 31 * 128:].shape, generator=generator)
        future = model.render(changed_future, state)
        require(torch.equal(future.deployed[..., :31 * 128], grouped.deployed[..., :31 * 128]),
                "Future callbacks affect earlier emissions")
        older = state.audio_history.clone()
        older[..., :3072] = 0
        altered = model.render(audio[..., :128], LongMagnitudeState(older, *state[1:]))
        original = model.render(audio[..., :128], state)
        require(not torch.equal(altered.deployed, original.deployed), "New past-history feature path is unexercised")
        padded = F.pad(audio[..., :-37], (0, 37))
        partial = model.render(padded)
        flushed, _ = model.flush(partial.state)
        require(flushed.shape == (1, 4, 2, 128) and bool(torch.isfinite(flushed).all()), "Partial EOF flush failed")
        closure = float((grouped.deployed.sum(1) - grouped.delayed_mixture).abs().max())
        require(closure < 1e-6 and model.algorithmic_latency_samples == 256, "Closure or latency differs")
    plan = {**read(PHASE / "full-magnitude-sdr-001/plan.json"), "parent_checkpoint": binding,
            "parent_model_state_sha256": initial_sha, "parent_training_updates": parent.provenance["training_updates"]}
    with tempfile.TemporaryDirectory(prefix="long-magnitude-cpu-", dir=PHASE) as temporary:
        run = Path(temporary)
        (run / "metrics.jsonl").write_text('{"cpu_test_step":1}\n')
        checkpoint = save_generation(model, optimizer, 1, plan, "cpu-functional-check", run)
        loaded, payload = load_model(checkpoint)
        resume = torch.load(run / "checkpoint/optimizer.pt", map_location="cpu", weights_only=True)
        require(len(resume["optimizer"]["state"]) == 23 and resume["model_state_sha256"] == payload["model_state_sha256"],
                "Saved optimizer inventory differs")
        with torch.inference_mode():
            replay = loaded.render(audio, state)
            require(torch.equal(grouped.raw, replay.raw) and torch.equal(grouped.deployed, replay.deployed)
                    and all(torch.equal(a,b) for a,b in zip(grouped.state, replay.state, strict=True)),
                    "Saved inference replay differs")
        sizes = {p.name: p.stat().st_size for p in (run / "checkpoint").iterdir()}
    require(state_sha256(parent.state_dict()) == parent_sha and not torch.cuda.is_initialized(),
            "CPU check changed the parent or initialized CUDA")
    return {"status": "pass", "zero_projection_parent_bit_exact": True,
            "independent_feature_max_abs": feature_error, "all_neural_tensors_updated": changed,
            "literal_waveform_max_abs": wave_max, "literal_waveform_rms": wave_rms,
            "literal_physical_state_max_abs": state_errors, "future_callback_causality_exact": True,
            "older_audio_feature_exercised": True, "partial_eof_flush": True, "closure_max_abs": closure,
            "algorithmic_latency_samples": 256, "saved_inference_replay_bit_exact": True,
            "saved_optimizer_tensor_count": 23, "temporary_checkpoint_bytes": sizes,
            "parent_state_unchanged": True, "gpu_used": False, "quality_measured": False}


if __name__ == "__main__":
    import json
    torch.set_num_interop_threads(1)
    print(json.dumps(check()), flush=True)
