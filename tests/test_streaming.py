"""Compare deployed ONNX output with an independent CPU PyTorch oracle."""

import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from hs_tasnet.streaming import StreamingSeparator


ROOT = Path(__file__).resolve().parents[1]


def references():
    path = ROOT / "tests/fixtures/hop128-pytorch.bin"
    data = path.read_bytes()
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert hashlib.sha256(data).hexdigest() == metadata["fixture_sha256"]
    assert data[:8] == b"SGRTG001"
    count, = struct.unpack_from("<I", data, 8)
    offset = 12
    result = []
    for _ in range(count):
        frames, = struct.unpack_from("<I", data, offset)
        offset += 4
        audio = np.frombuffer(data, dtype="<f4", count=2 * frames, offset=offset).reshape(2, frames)
        offset += 8 * frames
        expected = np.frombuffer(data, dtype="<f4", count=8 * frames, offset=offset).reshape(4, 2, frames)
        offset += 32 * frames
        result.append((audio, expected))
    assert offset == len(data)
    assert [audio.shape[-1] for audio, _ in result] == metadata["frames"]
    return result


@pytest.fixture(scope="module")
def separator():
    return StreamingSeparator(ROOT / "models/hop128.onnx")


@pytest.mark.parametrize("audio,expected", references(), ids=lambda x: str(x.shape))
def test_partial_eof_and_all_four_stems(separator, audio, expected):
    output = separator.separate(audio)
    assert output.shape == expected.shape
    assert np.isfinite(output).all()
    np.testing.assert_allclose(output, expected, rtol=0, atol=1e-5)
    np.testing.assert_allclose(output.sum(axis=0), audio, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(separator.separate(audio), output)
    assert separator.flush() is None


def test_chunk_alignment_flush_and_reset(separator):
    audio, expected = references()[4]  # 255 frames: one complete and one partial hop.
    separator.reset()
    assert separator.process_chunk(audio[:, :128]) is None
    first = separator.process_chunk(np.pad(audio[:, 128:], ((0, 0), (0, 1))))
    final = separator.flush()
    np.testing.assert_allclose(first, expected[..., :128], rtol=0, atol=1e-5)
    np.testing.assert_allclose(final[..., :127], expected[..., 128:], rtol=0, atol=1e-5)
    assert separator.flush() is None
    assert separator.process_chunk(audio[:, :128]) is None
    separator.reset()
    assert separator.flush() is None


def test_invalid_audio_resets_stream(separator):
    valid = np.ones((2, 128), dtype=np.float32) * .1
    for invalid in (valid.astype(np.float64), valid[:, :127], valid[:1], valid * np.nan):
        separator.reset()
        assert separator.process_chunk(valid) is None
        with pytest.raises(ValueError):
            separator.process_chunk(invalid)
        assert separator.flush() is None
    assert separator.separate(np.empty((2, 0), dtype=np.float32)).shape == (4, 2, 0)


def test_reject_wrong_rate_and_wrong_model(tmp_path):
    path = tmp_path / "wrong.onnx"
    path.write_bytes(b"wrong model")
    with pytest.raises(ValueError, match="44100"):
        StreamingSeparator(path, sample_rate=48000)
    with pytest.raises(ValueError, match="SHA-256"):
        StreamingSeparator(path)
