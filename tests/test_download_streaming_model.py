"""Failed downloads must not publish weights or overwrite local files."""

from io import BytesIO

import pytest

from scripts import download_streaming_model as model_download


def test_bad_checksum_leaves_no_model_or_temporary_file(tmp_path, monkeypatch):
    payload = b"different model"
    monkeypatch.setattr(model_download, "MODEL_BYTES", len(payload))
    monkeypatch.setattr(model_download, "urlopen", lambda *args, **kwargs: BytesIO(payload))
    with pytest.raises(ValueError, match="SHA-256"):
        model_download.download(tmp_path / "model.onnx")
    assert list(tmp_path.iterdir()) == []


def test_existing_different_file_is_preserved(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"existing user file")
    with pytest.raises(FileExistsError):
        model_download.download(path)
    assert path.read_bytes() == b"existing user file"


def test_interrupted_download_leaves_no_partial_model(tmp_path, monkeypatch):
    def interrupted(*args, **kwargs):
        raise TimeoutError("connection interrupted")

    monkeypatch.setattr(model_download, "urlopen", interrupted)
    with pytest.raises(TimeoutError):
        model_download.download(tmp_path / "model.onnx")
    assert list(tmp_path.iterdir()) == []
