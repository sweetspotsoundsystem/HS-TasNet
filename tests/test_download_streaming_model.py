"""Failed downloads must not publish weights or overwrite local files."""

from io import BytesIO

import pytest

from scripts import download_streaming_model as model_download


def test_bad_checksum_leaves_no_model_or_temporary_file(tmp_path, monkeypatch):
    payload = b"different model"
    monkeypatch.setitem(model_download.MODELS["current"], "bytes", len(payload))
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


@pytest.mark.parametrize("variant", ["current", "trainable"])
def test_selected_variant_downloads_and_reuses_its_own_verified_file(tmp_path, monkeypatch, variant):
    import hashlib

    payload = (variant + " model bytes").encode()
    selected = {**model_download.MODELS[variant], "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest()}
    monkeypatch.setitem(model_download.MODELS, variant, selected)
    urls = []

    def response(url, **kwargs):
        urls.append(url)
        return BytesIO(payload)

    monkeypatch.setattr(model_download, "urlopen", response)
    path = tmp_path / selected["filename"]
    assert model_download.download(path, variant=variant) == path
    assert path.read_bytes() == payload
    assert model_download.download(path, variant=variant) == path
    assert urls == [selected["url"]]
    assert list(tmp_path.iterdir()) == [path]
