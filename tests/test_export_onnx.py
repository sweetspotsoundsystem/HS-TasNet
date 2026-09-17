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


def _make_tiny_checkpoint(
    tmp_dir: Path,
    *,
    use_gru: bool,
    use_branch_rnns: bool = True,
    baked: bool = False,
) -> Path:
    from hs_tasnet import HSTasNet

    model = HSTasNet(
        dim=16,
        small=False,
        stereo=True,
        num_basis=16,
        segment_len=64,
        overlap_len=32,
        n_fft=64,
        sample_rate=8000,
        num_sources=4,
        use_gru=use_gru,
        use_branch_rnns=use_branch_rnns,
        residual_source_softmax=baked,
        spec_branch_use_phase=True,
    ).eval()
    if baked:
        model.set_output_source_gains(torch.tensor([1.0, 1.0, 0.8, 1.12]))
        model.bake_decoder_hann_window_()

    suffix = "c91-baked" if baked else ("gru" if use_gru else "lstm")
    ckpt_path = tmp_dir / f"tiny-{suffix}.pt"
    model.save(ckpt_path)
    return ckpt_path


class TestExportONNX(unittest.TestCase):
    def test_offline_fixed_export_matches_original_lstm(self) -> None:
        import hs_tasnet.hs_tasnet as hs_mod
        import onnx
        import onnxruntime as ort

        from export_onnx import export_onnx, load_model

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            ckpt_path = _make_tiny_checkpoint(td_path, use_gru=False)
            onnx_path = td_path / "tiny-offline.onnx"

            original_functions = (
                torch.view_as_real,
                torch.view_as_complex,
                torch.polar,
                hs_mod.multiply,
                hs_mod.repeat,
                hs_mod.rearrange,
                hs_mod.divide,
            )
            export_onnx(
                ckpt_path,
                onnx_path,
                opset_version=18,
                external_data=False,
                mode="offline",
                residual_source_index=None,
            )
            self.assertEqual(
                original_functions,
                (
                    torch.view_as_real,
                    torch.view_as_complex,
                    torch.polar,
                    hs_mod.multiply,
                    hs_mod.repeat,
                    hs_mod.rearrange,
                    hs_mod.divide,
                ),
            )

            graph = onnx.load_model(str(onnx_path), load_external_data=False)
            metadata = {item.key: item.value for item in graph.metadata_props}
            self.assertEqual(metadata["hs_tasnet.checkpoint_sha256"], _sha256_file(ckpt_path))
            self.assertEqual(metadata["hs_tasnet.export.mode"], "offline")
            self.assertEqual(metadata["hs_tasnet.export.dynamo"], "false")
            self.assertEqual(metadata["hs_tasnet.export.mixture_consistency"], "disabled")

            session = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
            rng = np.random.default_rng(7)
            audio = rng.standard_normal((1, 2, 64), dtype=np.float32)
            onnx_output = session.run(None, {"audio": audio})[0]

            original = load_model(ckpt_path, torch.device("cpu")).eval()
            with torch.inference_mode():
                expected, _ = original(
                    torch.from_numpy(audio),
                    hiddens=None,
                    auto_causal_pad=True,
                    auto_curtail_length_to_multiple=False,
                )
            np.testing.assert_allclose(
                onnx_output,
                expected.numpy(),
                rtol=2e-3,
                atol=2e-3,
            )

    def test_streaming_c91_baked_export_is_stateful_and_mixture_consistent(self) -> None:
        import onnx
        import onnxruntime as ort

        from export_onnx import export_onnx, load_model, route_mixture_residual

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            ckpt_path = _make_tiny_checkpoint(
                td_path,
                use_gru=True,
                use_branch_rnns=False,
                baked=True,
            )
            onnx_path = td_path / "tiny-streaming.onnx"
            export_onnx(
                ckpt_path,
                onnx_path,
                opset_version=18,
                external_data=False,
                mode="streaming",
                residual_source_index=3,
            )

            graph = onnx.load_model(str(onnx_path), load_external_data=False)
            metadata = {item.key: item.value for item in graph.metadata_props}
            self.assertEqual(metadata["hs_tasnet.export.mode"], "streaming")
            self.assertEqual(metadata["hs_tasnet.export.dynamo"], "true")
            self.assertEqual(metadata["hs_tasnet.model.decoder_hann_baked"], "true")
            self.assertEqual(metadata["hs_tasnet.export.mixture_consistency"], "route_residual")
            self.assertEqual(metadata["hs_tasnet.export.residual_source_index"], "3")
            self.assertEqual(metadata["hs_tasnet.streaming.output_delay_hops"], "1")

            session = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
            self.assertEqual(
                [value.name for value in session.get_inputs()],
                ["audio_chunk", "past_audio", "overlap_add_buffer", "fusion_hidden"],
            )
            self.assertEqual(
                [value.name for value in session.get_outputs()],
                [
                    "separated_chunk",
                    "next_past_audio",
                    "next_overlap_add_buffer",
                    "next_fusion_hidden",
                ],
            )

            original = load_model(ckpt_path, torch.device("cpu")).eval()
            native_stream = original.init_stateful_transform_fn(device="cpu")
            past = np.zeros((1, 2, 32), dtype=np.float32)
            overlap = np.zeros((1, 4, 2, 64), dtype=np.float32)
            hidden = np.zeros((2, 1, 32), dtype=np.float32)

            rng = np.random.default_rng(11)
            chunks = [np.zeros((1, 2, 32), dtype=np.float32)]
            impulse = np.zeros((1, 2, 32), dtype=np.float32)
            impulse[:, :, -1] = 0.5
            chunks.append(impulse)
            chunks.extend(
                rng.standard_normal((1, 2, 32), dtype=np.float32) * 0.05
                for _ in range(4)
            )
            # Flush the output aligned with the final real chunk.
            chunks.append(np.zeros((1, 2, 32), dtype=np.float32))

            for chunk in chunks:
                native_output = native_stream(torch.from_numpy(chunk[0])).unsqueeze(0)
                corrected = route_mixture_residual(
                    native_output,
                    torch.from_numpy(past),
                    residual_source_index=3,
                ).numpy()
                outputs = session.run(
                    None,
                    {
                        "audio_chunk": chunk,
                        "past_audio": past,
                        "overlap_add_buffer": overlap,
                        "fusion_hidden": hidden,
                    },
                )
                separated, next_past, next_overlap, next_hidden = outputs
                np.testing.assert_allclose(separated, corrected, rtol=2e-4, atol=2e-5)
                np.testing.assert_allclose(separated[:, :3], native_output[:, :3], rtol=2e-4, atol=2e-5)
                np.testing.assert_allclose(
                    separated.sum(axis=1),
                    past,
                    rtol=0.0,
                    atol=2e-7,
                )
                np.testing.assert_array_equal(next_past, chunk)
                past, overlap, hidden = next_past, next_overlap, next_hidden


if __name__ == "__main__":
    unittest.main()
