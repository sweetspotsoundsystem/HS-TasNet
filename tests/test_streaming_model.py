"""Check the trainable model against the separately recorded release oracle."""
from pathlib import Path

import numpy as np
import pytest
import torch

from hs_tasnet import StreamingHSTasNet, StreamingState
from hs_tasnet.streaming_checkpoint import (
    RELEASED_STATE_SHA256, import_released_onnx, load_streaming_checkpoint,
    save_streaming_checkpoint, state_sha256,
)
from test_streaming import references

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(1)
    return import_released_onnx(ROOT / "models/hop128.onnx")


@pytest.mark.parametrize("audio,expected", references(), ids=lambda value: str(value.shape))
def test_original_reference_and_partial_alignment(model, audio, expected):
    with torch.no_grad():
        output = model.separate(torch.from_numpy(audio.copy())[None])[0].numpy()
    np.testing.assert_allclose(output, expected, atol=1e-5, rtol=0)
    np.testing.assert_allclose(output.sum(axis=0), audio, atol=1e-6, rtol=0)
    assert output.shape == expected.shape


def test_all_states_and_chunk_partition(model):
    generator = torch.Generator().manual_seed(37)
    state = StreamingState(*(torch.randn(value.shape, generator=generator) * scale
                             for value, scale in zip(model.initial_state(2), (.01, 2**-21, .001, .001))))
    audio = torch.randn(2, 2, 640, generator=generator) * .03
    with torch.no_grad():
        whole = model.render(audio, state)
        current, pieces = state, []
        for chunk in audio.split(128, dim=-1):
            result = model.render(chunk, current)
            current = result.state
            pieces.append(result.deployed)
        torch.testing.assert_close(torch.cat(pieces, dim=-1), whole.deployed, atol=1e-5, rtol=0)
        for actual, expected in zip(current, whole.state):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
        torch.testing.assert_close(whole.deployed.sum(dim=1), whole.delayed_mixture, atol=1e-6, rtol=0)
        replay = model.render(audio, state)
        assert torch.equal(whole.deployed, replay.deployed)


def test_checkpoint_import_roundtrip_preserves_all_parameters(model, tmp_path):
    assert state_sha256(model.state_dict()) == RELEASED_STATE_SHA256
    assert len(list(model.parameters())) == 21 and len(list(model.buffers())) == 6
    before = torch.get_rng_state().clone()
    path = save_streaming_checkpoint(model, tmp_path / "weights.pt")
    try:
        loaded = StreamingHSTasNet.from_checkpoint(path)
        assert state_sha256(loaded.state_dict()) == RELEASED_STATE_SHA256
        assert torch.equal(torch.get_rng_state(), before)
        with pytest.raises(FileExistsError):
            save_streaming_checkpoint(model, path)
    finally:
        path.unlink()


def test_invalid_input_and_bf16_cpu_rejected(model):
    for audio in (torch.zeros(1, 1, 128), torch.zeros(1, 2, 127), torch.zeros(1, 2, 128, dtype=torch.float64),
                  torch.full((1, 2, 128), float("nan"))):
        with pytest.raises(ValueError):
            model.render(audio)
    with pytest.raises(ValueError):
        model.render(torch.zeros(1, 2, 128), model.initial_state(2))
    model.train()
    model.training_precision = "bf16"
    try:
        with pytest.raises(ValueError, match="CUDA"):
            model.render(torch.zeros(1, 2, 128))
    finally:
        model.training_precision = "fp32"
        model.eval()
    assert model.separate(torch.empty(1, 2, 0)).shape == (1, 4, 2, 0)
