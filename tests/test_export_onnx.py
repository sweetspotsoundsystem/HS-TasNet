from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_tiny_checkpoint(tmp_dir: Path, *, use_gru: bool) -> Path:
    from hs_tasnet import HSTasNet

    # Keep this intentionally small so ONNX export stays fast.
    model = HSTasNet(
        dim=32,
        small=True,
        stereo=False,
        num_basis=16,
        segment_len=64,
        overlap_len=32,
        n_fft=64,
        sample_rate=8000,
        num_sources=4,
        use_gru=use_gru,
        spec_branch_use_phase=True,
    ).eval()

    ckpt_path = tmp_dir / ("tiny-gru.pt" if use_gru else "tiny-lstm.pt")
    model.save(ckpt_path)
    return ckpt_path


class TestNewExportONNX(unittest.TestCase):
    def _export_and_compare(self) -> None:
        import hs_tasnet.hs_tasnet as hs_mod
        import onnx
        import onnxruntime as ort

        from export_onnx import (
            HSTasNetONNXWrapper,
            export_onnx,
            load_model,
            onnx_export_patches,
            patch_model_for_onnx,
        )

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            ckpt_path = _make_tiny_checkpoint(td_path, use_gru=False)
            onnx_path = td_path / "tiny.onnx"

            # Capture identities to ensure temporary patching is restored.
            orig_torch_view_as_real = torch.view_as_real
            orig_torch_view_as_complex = torch.view_as_complex
            orig_torch_polar = torch.polar
            orig_hs_multiply = hs_mod.multiply
            orig_hs_repeat = hs_mod.repeat
            orig_hs_rearrange = hs_mod.rearrange
            orig_hs_divide = hs_mod.divide

            export_onnx(ckpt_path, onnx_path, opset_version=18, external_data=False)

            m = onnx.load_model(str(onnx_path), load_external_data=False)
            meta = {p.key: p.value for p in m.metadata_props}
            self.assertEqual(meta.get("hs_tasnet.checkpoint_name"), ckpt_path.name)
            self.assertEqual(meta.get("hs_tasnet.checkpoint_sha256"), _sha256_file(ckpt_path))
            self.assertEqual(meta.get("hs_tasnet.opset_version"), "18")
            self.assertEqual(meta.get("hs_tasnet.external_data"), "false")
            self.assertIn("hs_tasnet.exported_at_utc", meta)

            self.assertIs(torch.view_as_real, orig_torch_view_as_real)
            self.assertIs(torch.view_as_complex, orig_torch_view_as_complex)
            self.assertIs(torch.polar, orig_torch_polar)
            self.assertIs(hs_mod.multiply, orig_hs_multiply)
            self.assertIs(hs_mod.repeat, orig_hs_repeat)
            self.assertIs(hs_mod.rearrange, orig_hs_rearrange)
            self.assertIs(hs_mod.divide, orig_hs_divide)

            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

            # Input length must be a multiple of segment_len.
            audio = np.random.randn(1, 1, 128).astype(np.float32)
            onnx_out = sess.run(None, {"audio": audio})[0]
            self.assertEqual(onnx_out.shape, (1, 4, 1, 128))

            # Compare against PyTorch running the same patched model + wrapper.
            model = load_model(ckpt_path, device=torch.device("cpu")).eval()
            model = patch_model_for_onnx(model)
            wrapped = HSTasNetONNXWrapper(model).eval()

            with torch.no_grad(), onnx_export_patches():
                pt_out = wrapped(torch.from_numpy(audio)).cpu().numpy()

            np.testing.assert_allclose(onnx_out, pt_out, rtol=1e-3, atol=1e-3)

    def test_export_onnx_lstm(self) -> None:
        self._export_and_compare()


if __name__ == "__main__":
    unittest.main()
