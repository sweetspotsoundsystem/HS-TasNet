"""Current native model contracts, streaming trajectories and training context."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from stemgenrt.model import StemgenRT58, StreamingState, render_scored_context


def new_model(window=32, past_filter=True):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(617)
        model = StemgenRT58(attention_window=window, past_filter=past_filter)
        # Exercise paths that deliberately start with zero output projections.
        with torch.no_grad():
            for parameter in model.parameters():
                if torch.count_nonzero(parameter) == 0:
                    parameter.normal_(std=.001)
    return model


def test_default_model_exposes_current_eight_state_interface():
    model = StemgenRT58()
    state = model.initial_state(1)
    assert type(state) is StreamingState and len(state) == 8
    assert model.attention_window == 32 and not model.has_past_filter
    assert model.parameter_tensor_count == 40 and model.training_precision == 'fp32'
    assert model.algorithmic_latency_samples == 256


def test_public_import_keeps_torch_lazy_and_has_no_research_dependency():
    script = """
import sys
import stemgenrt
assert 'torch' not in sys.modules
assert set(stemgenrt.__all__) == {
    'StreamingSeparator', 'StemgenRT58', 'StreamingState', 'render_scored_context'}
from stemgenrt import StemgenRT58
assert StemgenRT58.__module__ == 'stemgenrt.model'
assert not any(name == 'research' or name.startswith('research.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize("window,past_filter", ((32, False), (128, False), (32, True)))
def test_native_matches_checkpoint_geometry_and_preserves_released_interface(window, past_filter):
    model = new_model(window, past_filter).eval()
    release = json.loads((Path(__file__).resolve().parents[1]
                          / "stemgenrt/streaming_models.json").read_text())["current"]
    state = model.initial_state(1)
    assert (type(state) is StreamingState) == (not past_filter)
    assert tuple(release["states"]) == state._fields[:8]
    expected = dict(release["states"])
    expected["attention_keys"] = [1, window - 1, 64]
    expected["attention_values"] = [1, window - 1, 128]
    if past_filter:
        expected["past_carrier_history"] = [1, 2, 2, 513, 2]
    assert [list(value.shape) for value in state] == list(expected.values())
    assert len(list(model.parameters())) == 40 + int(past_filter)
    assert sum(value.numel() for value in model.parameters()) == 32_775_840 + 16_000 * past_filter
    assert len(list(model.buffers())) == 7
    assert model.sample_rate == 44100
    assert model.hop_samples == model.graph_alignment_samples == model.host_queue_samples == 128
    assert model.algorithmic_latency_samples == 256
    assert model.architecture_metadata["state_names"] == list(state._fields)
    assert model.architecture_metadata["state_family"] == (
        "stemgenrt-shared-mask-past1-2-v1" if past_filter else
        "latency58-attention-private-branch-gru500-zero-projections-v1" if window == 32
        else "stemgenrt-branch-memory-attention128-v1")
    assert all(value.dtype == torch.float32 for value in state)
    assert not hasattr(model, "from_accepted") and not hasattr(model, "from_parent")


def test_streaming_carry_reset_closure_and_warmup_shortcut():
    model = new_model().eval()
    generator = torch.Generator().manual_seed(991)
    audio = torch.randn(1, 2, 6 * 128, generator=generator) * .03
    with torch.no_grad():
        block = model.render(audio)
        state, chunks = model.initial_state(1), []
        for hop in audio.split(128, dim=-1):
            output = model.render(hop, state)
            state = output.state
            chunks.append(output.deployed)
        torch.testing.assert_close(torch.cat(chunks, dim=-1), block.deployed,
                                   atol=1e-6, rtol=1e-5)
        for actual, expected in zip(state, block.state, strict=True):
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
        warm = model.warm_state(audio)
        for actual, expected in zip(warm, block.state, strict=True):
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(block.deployed.sum(1), block.delayed_mixture,
                                   atol=1e-6, rtol=0)
        assert torch.equal(model.render(audio).deployed, block.deployed)
        flushed, _ = model.flush(block.state)
        assert flushed.shape == (1, 4, 2, 128)


def test_detached_warmup_preserves_scored_input_and_all_parameter_gradients():
    model = new_model().train()
    audio = (torch.randn(1, 2, 6 * 128) * .03).requires_grad_()
    scored = render_scored_context(model, audio, warmup_samples=256, carry_state=True)
    assert scored.flush_hops == 1 and scored.initial_state_detached
    assert torch.equal(scored.physical_mixture, audio[..., 256:])
    (scored.raw.square().mean() + scored.deployed.square().mean()).backward()
    assert torch.count_nonzero(audio.grad[..., :256]) == 0
    assert torch.count_nonzero(audio.grad[..., 256:]) > 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


def test_invalid_stream_geometry_is_rejected():
    model = new_model().eval()
    with pytest.raises(ValueError):
        model.render(torch.zeros(1, 2, 129))
    with pytest.raises(ValueError):
        model.render(torch.zeros(1, 2, 128, dtype=torch.float64))
    with pytest.raises(ValueError):
        model.render(torch.zeros(1, 2, 128), tuple(model.initial_state(1)))
    with pytest.raises(ValueError):
        model.render(torch.zeros(1, 2, 128), new_model(128, False).initial_state(1))


def test_explicit_window_conversion_preserves_weights_rng_and_latency():
    from stemgenrt.checkpoint import state_sha256
    parent = new_model(32, False).eval()
    before = state_sha256(parent.state_dict())
    rng = torch.get_rng_state().clone()
    candidate = parent.with_attention_window(128)
    assert candidate.attention_window == 128 and parent.attention_window == 32
    assert state_sha256(candidate.state_dict()) == state_sha256(parent.state_dict()) == before
    assert candidate.provenance["window_parent_state_sha256"] == before
    assert candidate.provenance["quality_measured"] is False
    assert torch.equal(rng, torch.get_rng_state())
    assert candidate.algorithmic_latency_samples == parent.algorithmic_latency_samples == 256
    assert sum(s.numel() * s.element_size() for s in candidate.initial_state(1)) - sum(
        s.numel() * s.element_size() for s in parent.initial_state(1)) == 73728
    with pytest.raises(ValueError):
        parent.with_attention_window(64)
