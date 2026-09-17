"""Functional checks for a trainable spectral-magnitude adapter; no quality claim."""
from __future__ import annotations

import json

from research.direct.run_latency58_quality import PHASE, read, require, write
from research.direct.train_latency58 import state_sha256


def check():
    import torch
    from research.direct.latency58 import PUBLIC_FUSION_SCALE
    from research.direct.latency58_direct_sdr_checkpoint import load_model
    from research.direct.latency58_magnitude import ADAPTER, MagnitudeEncoder, Latency58MagnitudeModel
    torch.set_num_threads(1)
    torch.manual_seed(20260928)
    parent, _ = load_model(read(PHASE / "c204-residual-model-001/checkpoint.json"))
    baseline = state_sha256(parent.state_dict())
    model = Latency58MagnitudeModel.from_parent(parent)
    audio = torch.randn(1, 2, 11 * 128) * .03
    with torch.inference_mode():
        original, initial = parent.render(audio), model.render(audio)
        require(torch.equal(original.raw, initial.raw) and torch.equal(original.deployed, initial.deployed)
                and all(torch.equal(a, b) for a, b in zip(original.state, initial.state, strict=True)),
                "Zero adapter does not reproduce the parent exactly")
        packed = torch.randn(2, 3, 2052)
        paired = packed.unflatten(-1, (1026, 2))
        rotated = torch.stack((-paired[..., 1], paired[..., 0]), -1).flatten(-2)
        require(torch.equal(MagnitudeEncoder.magnitude_features(packed), MagnitudeEncoder.magnitude_features(rotated)),
                "Magnitude features depend on complex phase")
        model.spec_encode.magnitude_projection.weight.normal_(0, .0001)
        whole = model.render(audio)
        state, outputs, offset = None, [], 0
        for hops in (1, 3, 2, 5):
            output = model.render(audio[..., offset:offset + hops * 128], state)
            outputs.append(output.deployed)
            state = output.state
            offset += hops * 128
        partition_error = float((whole.deployed - torch.cat(outputs, -1)).abs().max())
        state_errors = [float((a - b).abs().max()) / (PUBLIC_FUSION_SCALE if i == 1 else 1.)
                        for i, (a, b) in enumerate(zip(whole.state, state, strict=True))]
        require(partition_error < 1e-4 and max(state_errors) < 5e-4, "Partitioned stream differs")
        changed = audio.clone()
        changed[..., 6 * 128:] *= -2
        require(torch.equal(whole.deployed[..., :6 * 128], model.render(changed).deployed[..., :6 * 128]),
                "Future input changes emitted output")
        require(torch.equal(whole.deployed, model.render(audio).deployed), "Reset replay differs")
        state, outputs = None, []
        for chunk in audio.split(128, -1):
            output, state = model.forward_chunk(chunk, state)
            outputs.append(output)
        flushed, _ = model.flush(state)
        host = torch.cat([torch.zeros_like(outputs[0]), *outputs, flushed], -1)
        closure = float((host[..., 256:256 + audio.shape[-1]].sum(1) - audio).abs().max())
        require(closure < 1e-6 and model.algorithmic_latency_samples == 256, "Physical host delay changed")
    model.train_adapter_only()
    frozen = state_sha256({k: v for k, v in model.state_dict().items() if k != ADAPTER})
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    output = model.render(audio[..., :5 * 128])
    loss = (output.deployed.square() * torch.arange(1, 5)[None, :, None, None]).mean()
    loss.backward()
    parameter = model.spec_encode.magnitude_projection.weight
    require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            and parameter.grad.abs().max() > 0, "Adapter gradient is missing")
    require(all(p.grad is None for name, p in model.named_parameters() if name != ADAPTER), "Inherited weights have gradients")
    before = parameter.detach().clone()
    optimizer.step()
    require(not torch.equal(before, parameter) and frozen == baseline
            == state_sha256({k: v for k, v in model.state_dict().items() if k != ADAPTER}), "Update changed inherited tensors")
    require(state_sha256(parent.state_dict()) == baseline and not torch.cuda.is_initialized(), "Parent or CPU scope changed")
    return {"status": "pass", "quality_measured": False, "gpu_used": False,
            "zero_adapter_parent_bit_exact": True, "phase_rotation_invariant_features": True,
            "nonzero_adapter_partition_max_abs": partition_error, "state_partition_max_abs_decoded": state_errors,
            "future_input_independence": True, "exact_reset_replay": True,
            "host_total_delay_samples": 256, "complete_sample_recovery_max_abs": closure,
            "trainable_parameters": parameter.numel(), "adapter_updated": True,
            "inherited_tensors_unchanged": True, "parent_model_state_sha256": baseline}


if __name__ == "__main__":
    result = check()
    out = PHASE / "magnitude-functional-001"
    require(not out.exists(), "Preserve previous functional evidence")
    out.mkdir()
    write(out / "result.json", result)
    print(json.dumps(result), flush=True)
