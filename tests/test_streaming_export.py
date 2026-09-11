"""Export the real weights and verify the graph through the public inference API."""
from pathlib import Path

import numpy as np
import pytest
import torch

from hs_tasnet.streaming import StreamingSeparator
from hs_tasnet.streaming_checkpoint import import_released_onnx, state_sha256
from hs_tasnet.streaming_export import export_streaming_model

ROOT = Path(__file__).resolve().parents[1]


def test_export_preserves_model_and_verifies_recurrent_states(tmp_path):
    torch.set_num_threads(1)
    model = import_released_onnx(ROOT / "models/hop128.onnx")
    before = state_sha256(model.state_dict())
    model.train()
    model.fusion_branch.eval()
    flags = [module.training for module in model.modules()]
    path = tmp_path / "model.onnx"
    report = export_streaming_model(model, path, verify_hops=8)
    try:
        assert report["status"] == "pass" and report["verification"]["passed"]
        assert state_sha256(model.state_dict()) == before
        assert [module.training for module in model.modules()] == flags
        separator = StreamingSeparator(path, expected_sha256=report["onnx_sha256"])
        audio = np.random.default_rng(49).normal(0, .02, (2, 769)).astype(np.float32)
        model.eval()
        reference = model.separate(torch.from_numpy(audio)[None])[0].numpy()
        np.testing.assert_allclose(separator.separate(audio), reference, rtol=0, atol=1e-5)
        with pytest.raises(FileExistsError):
            export_streaming_model(model, path, verify_hops=8)
    finally:
        path.unlink()
        path.with_suffix(".verification.json").unlink()
