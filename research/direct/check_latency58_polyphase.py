"""CPU-only state, gradient and latency checks for the untrained cadence prototype."""
from __future__ import annotations

import json

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write


def check():
    import torch
    from research.direct.latency58 import PUBLIC_FUSION_SCALE
    from research.direct.latency58_direct_sdr_checkpoint import load_model
    from research.direct.latency58_evaluate import model_state_sha256
    from research.direct.latency58_polyphase import InterleavedGRU, Latency58InterleavedModel
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260927)
    torch.use_deterministic_algorithms(True)
    recurrent_error = 0.0
    # Independent oracle: a literal hop loop with a fixed bank and phase index.
    # The implementation instead strides complete sequences and rotates the bank.
    for phases in (1, 2, 4):
        gru = InterleavedGRU(3, 5, num_layers=2, batch_first=True, phases=phases).double()
        oracle = torch.nn.GRU(3, 5, num_layers=2, batch_first=True).double()
        oracle.load_state_dict(gru.state_dict(), strict=True)
        for frames in (1, 2, 3, 5, 8, 11):
            samples = torch.randn(2, frames, 3, dtype=torch.float64)
            initial = torch.randn(2, 2, phases * 5, dtype=torch.float64)
            actual, actual_state = gru(samples, initial)
            slots = list(initial.reshape(2, 2, phases, 5).unbind(2))
            expected = []
            for index in range(frames):
                output, hidden = oracle(samples[:, index:index + 1], slots[index % phases].contiguous())
                slots[index % phases] = hidden
                expected.append(output)
            rotation = frames % phases
            expected_state = torch.stack(slots[rotation:] + slots[:rotation], dim=2).flatten(2)
            error = max(float((actual - torch.cat(expected, 1)).abs().max().detach()),
                        float((actual_state - expected_state).abs().max().detach()))
            require(error < 1e-12, "Grouped cadence disagrees with the literal independent oracle")
            recurrent_error = max(recurrent_error, error)
    checkpoint = read(PHASE / "c204-residual-model-001/checkpoint.json")
    parent, _ = load_model(checkpoint)
    before = model_state_sha256(parent)
    one = Latency58InterleavedModel.from_parent(parent, phases=1)
    four = Latency58InterleavedModel.from_parent(parent, phases=4)
    audio = torch.randn(1, 2, 17 * 128) * .03
    with torch.inference_mode():
        baseline, one_output = parent.render(audio), one.render(audio)
        require(torch.equal(baseline.raw, one_output.raw) and torch.equal(baseline.deployed, one_output.deployed)
                and all(torch.equal(a, b) for a, b in zip(baseline.state, one_output.state, strict=True)),
                "One-phase control differs from the original model")
        whole = four.render(audio)
        state, outputs, offset = None, [], 0
        for hops in (1, 3, 2, 7, 4):
            output = four.render(audio[..., offset:offset + hops * 128], state)
            outputs.append(output.deployed)
            state = output.state
            offset += hops * 128
        partition_error = float((whole.deployed - torch.cat(outputs, -1)).abs().max())
        state_errors = [float((a - b).abs().max()) / (PUBLIC_FUSION_SCALE if i == 1 else 1.)
                        for i, (a, b) in enumerate(zip(whole.state, state, strict=True))]
        require(partition_error < 1e-4 and max(state_errors) < 5e-4, "Arbitrary callback partitions change the stream")
        modified = audio.clone()
        modified[..., 8 * 128:] = torch.randn_like(modified[..., 8 * 128:]) * .1
        future = four.render(modified)
        require(torch.equal(whole.deployed[..., :8 * 128], future.deployed[..., :8 * 128]),
                "Future input changed an already emitted output")
        require(torch.equal(whole.deployed, four.render(audio).deployed), "Reset replay differs")
        state, callbacks = None, []
        for chunk in audio.split(128, dim=-1):
            output, state = four.forward_chunk(chunk, state)
            callbacks.append(output)
        flushed, _ = four.flush(state)
        callbacks.append(flushed)
        # One actual host hop queues the graph output; drain that queue once.
        host = torch.cat([torch.zeros_like(callbacks[0]), *callbacks], dim=-1)
        recovered = host[..., 256:256 + audio.shape[-1]].sum(1)
        closure_error = float((recovered - audio).abs().max())
        require(closure_error < 1e-6 and four.algorithmic_latency_samples == 256,
                "Host queue failed complete physical sample recovery at 256 samples")
    four.train().requires_grad_(True)
    state = four.initial_state(1)
    for value in state:
        value.requires_grad_()
    samples = audio[..., :10 * 128].clone().requires_grad_()
    first = four.render(samples[..., :3 * 128], state)
    second = four.render(samples[..., 3 * 128:], first.state)
    estimates = torch.cat((first.deployed, second.deployed), dim=-1)
    loss = (estimates.square() * torch.arange(1, 5, dtype=torch.float32)[None, :, None, None]).mean()
    loss.backward()
    parameters = list(four.parameters())
    require(len(parameters) == 21 and all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                and bool(p.grad.abs().max() > 0) for p in parameters), "A neural parameter lost its finite nonzero gradient")
    require(samples.grad is not None and bool(torch.isfinite(samples.grad).all())
            and all(v.grad is not None and bool(torch.isfinite(v.grad).all()) and bool(v.grad.abs().max() > 0) for v in state),
            "A carried state or input lost its gradient")
    require(all(not v.requires_grad and v.grad_fn is None for v in second.state.detached()), "Detached warmup retained a graph")
    require(model_state_sha256(parent) == model_state_sha256(one) == model_state_sha256(four) == before
            and not torch.cuda.is_initialized(), "Functional checks changed model tensors or initialized CUDA")
    return {"status": "pass", "quality_measured": False, "training_updates": 0, "gpu_used": False,
            "checkpoint_parent": checkpoint, "unchanged_neural_model_state_sha256": before,
            "independent_recurrence_cases": 18, "independent_recurrence_max_abs": recurrent_error,
            "one_phase_original_bit_exact": True, "four_phase_partition_max_abs": partition_error,
            "state_partition_max_abs_decoded": state_errors, "future_input_independence": True,
            "reset_replay_exact": True, "host_total_delay_samples": 256, "complete_sample_recovery_max_abs": closure_error,
            "finite_nonzero_parameter_and_state_gradients": True, "parameter_tensors": 21,
            "architecture": four.architecture_metadata,
            "source_bindings": {str(p): sha(p) for p in (ROOT / "research/direct/latency58_polyphase.py",
                ROOT / "research/direct/check_latency58_polyphase.py")}}


if __name__ == "__main__":
    result = check()
    out = PHASE / "polyphase-functional-001"
    require(not out.exists(), "Preserve existing functional results")
    out.mkdir()
    write(out / "result.json", result)
    print(json.dumps(result), flush=True)
