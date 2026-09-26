"""Current-model initialization, causality, and complete trainable inventory."""
import pytest
import torch

from stemgenrt.model import StemgenRT58, StreamingState
from stemgenrt.checkpoint import ParameterEMA, state_sha256, _validate_geometry_config


def test_parent_conversion_preserves_weights_rng_and_zero_head_output():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(103)
        parent = StemgenRT58(attention_window=32, past_filter=False).eval()
        with torch.no_grad():
            parent.waveform_decoder_weight.normal_(std=.001)
    before = state_sha256(parent.state_dict())
    rng = torch.get_rng_state().clone()
    model = parent.with_past_filter()
    assert torch.equal(rng, torch.get_rng_state())
    assert state_sha256(parent.state_dict()) == before
    assert state_sha256({k: v for k, v in model.state_dict().items() if k != 'past_filter.gate.weight'}) == before
    assert state_sha256(model.state_dict()) != before
    assert torch.count_nonzero(model.past_filter.gate.weight) == 0
    assert model.provenance['past_filter_parent_model_state_sha256'] == before
    assert model.parameter_tensor_count == len(list(model.parameters())) == 41
    assert model.algorithmic_latency_samples == parent.algorithmic_latency_samples == 256
    assert sum(v.numel() * v.element_size() for v in model.initial_state(1)) - sum(
        v.numel() * v.element_size() for v in parent.initial_state(1)) == 16416
    audio = torch.randn(1, 2, 5 * 128) * .03
    with torch.no_grad():
        expected, actual = parent.render(audio), model.render(audio)
        for name in ('raw', 'deployed', 'spectral', 'waveform', 'delayed_mixture', 'native_raw'):
            assert torch.equal(getattr(actual, name), getattr(expected, name))
        for a, b in zip(actual.state[:8], expected.state, strict=True):
            assert torch.equal(a, b)
    ema = ParameterEMA(model)
    assert ema.base_state_sha256 == state_sha256(model.state_dict()) != before
    assert len(ema.parameters) == 41 and len(model.initial_state(1)) == 9
    with pytest.raises(ValueError, match='base model'):
        ParameterEMA(model, base_state_sha256=before)
    with pytest.raises(ValueError, match='past filter'):
        _validate_geometry_config({'attention_window': 32}, model)
    _validate_geometry_config({'attention_window': 32, 'past_filter': True}, model)
    with pytest.raises(ValueError):
        model.render(audio, expected.state)
    with pytest.raises(ValueError):
        parent.render(audio, actual.state)
    with pytest.raises(ValueError):
        model.with_attention_window(128)


@pytest.mark.parametrize('configuration', [dict(attention_window=128, past_filter=True), dict(past_filter=1)])
def test_unqualified_past_filter_geometry_is_rejected(configuration):
    with pytest.raises(ValueError):
        StemgenRT58(**configuration)
