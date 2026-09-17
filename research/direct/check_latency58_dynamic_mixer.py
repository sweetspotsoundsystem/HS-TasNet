"""Independent waveform-loss and streaming checks for dynamic source correction."""
from __future__ import annotations

import json

from research.direct.run_latency58_quality import PHASE, read, require
from research.direct.train_latency58 import state_sha256


def check():
    import torch
    from research.direct.latency58_dynamic_mixer import (
        HOP, WINDOW, FRAMES, Latency58DynamicMixer, apply_correction, coefficients,
        make_head, mixing_ramp, mixing_scale, frame_covariances, loss_from_frames,
    )
    from research.direct.latency58_direct_sdr_checkpoint import load_model
    torch.set_num_threads(1)
    torch.manual_seed(20261002)
    truth = torch.randn(2, 4, 2, FRAMES * HOP) * .04
    truth[0, 1, :, :WINDOW] = 0
    mixture = truth.sum(1)
    estimates = truth + torch.randn_like(truth) * .015
    estimates[:, 3] = mixture - ((estimates[:, 0] + estimates[:, 1]) + estimates[:, 2])
    features = torch.randn(2, FRAMES + 1, 1000) * .2
    head = make_head()
    with torch.no_grad():
        head.weight.normal_(0, .002)
        head.bias.normal_(0, .005)
    moments = frame_covariances(estimates, truth, mixture)
    loss, calculated, _ = loss_from_frames(head, features, moments)
    delta = coefficients(head, features, mixing_scale()).double()
    actual = apply_correction(estimates.double(), mixture.double(), delta, mixing_ramp().double())
    reference = truth[..., :2 * WINDOW].double().unflatten(-1, (2, WINDOW))
    error = (actual[..., :2 * WINDOW] - truth[..., :2 * WINDOW].double()).unflatten(-1, (2, WINDOW)).square().sum((2, 4))
    signal = reference.square().sum((2, 4))
    active = signal / (2 * WINDOW) > 1e-5
    expected = (10 * torch.log10((signal + 1e-12) / (error + 1e-12))).clamp(-60, 60)
    expected = torch.where(active, expected, 0).sum((0, 2)) / active.sum((0, 2)).clamp_min(1)
    discrepancy = float((calculated - expected).abs().max().detach())
    require(discrepancy < 1e-5, "Fragment covariance loss differs from direct waveform errors")
    loss.backward()
    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and p.grad.abs().max() > 0 for p in head.parameters()),
            "Dynamic head gradient is missing or nonfinite")
    parent, _ = load_model(read(PHASE / "c204-residual-model-001/checkpoint.json"))
    baseline = state_sha256(parent.state_dict())
    initial = Latency58DynamicMixer.from_parent(parent)
    model = Latency58DynamicMixer.from_parent(parent, head.state_dict())
    audio = torch.randn(1, 2, 11 * HOP) * .03
    with torch.inference_mode():
        native, unchanged = parent.render(audio), initial.render(audio)
        require(torch.equal(native.deployed, unchanged.deployed)
                and all(torch.equal(a, b) for a, b in zip(native.state, unchanged.state, strict=True)),
                "Zero head does not reproduce the deployed parent exactly")
        whole = model.render(audio)
        outputs, state, offset = [], None, 0
        for count in (2, 3, 6):
            rendered = model.render(audio[..., offset:offset + count * HOP], state)
            state = rendered.state
            outputs.append(rendered.deployed)
            offset += count * HOP
        partition = float((whole.deployed - torch.cat(outputs, -1)).abs().max())
        require(partition < 1e-4, "Dynamic coefficients depend on chunk partition")
        changed = audio.clone()
        changed[..., 6 * HOP:] *= -2
        require(torch.equal(whole.deployed[..., :6 * HOP], model.render(changed).deployed[..., :6 * HOP]),
                "Future audio changed an emitted sample")
        state, outputs = None, []
        for chunk in audio.split(HOP, -1):
            output, state = model.forward_chunk(chunk, state)
            outputs.append(output)
        flush, _ = model.flush(state)
        host = torch.cat([torch.zeros_like(outputs[0]), *outputs, flush], -1)
        closure = float((host[..., 256:256 + audio.shape[-1]].sum(1) - audio).abs().max())
        require(closure < 1e-6 and model.algorithmic_latency_samples == 256 and len(state) == 4,
                "Dynamic correction changes host delay or state count")
        require(torch.equal(whole.deployed, model.render(audio).deployed), "Reset replay differs")
    require(state_sha256(parent.state_dict()) == baseline and not torch.cuda.is_initialized(), "Parent or CPU scope changed")
    return {"status": "pass", "covariance_vs_waveform_sdr_max_abs_db": discrepancy,
            "zero_head_deployed_parent_bit_exact": True, "nonzero_head_partition_max_abs": partition,
            "future_input_independence": True, "saved_state_tensors": 4, "exact_reset_replay": True,
            "host_total_delay_samples": 256, "complete_recovery_max_abs": closure,
            "inherited_tensors_unchanged": True, "trainable_parameters": 12012, "gpu_used": False, "quality_measured": False}


if __name__ == "__main__":
    print(json.dumps(check()), flush=True)
