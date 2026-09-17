"""Export direct C126/C191 callbacks and verify independent CPU trajectories.

Example:
    python -m research.direct.latency_onnx --kind c191 \
      --checkpoint research/direct/runs/latency11/baselines/c191-native.pt \
      --output research/direct/runs/latency11/baselines/c191-native.onnx

The graph emits the current 512 samples. Its host scheduling allowance is
512 samples; no graph output delay, flush, or future callback is introduced.
ONNX DFT operates on real/imaginary tensors and preserves the native FFT path.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch
from torch import Tensor, nn

import export_onnx as existing_export
from research.direct.latency11 import CORE_SOURCE, file_sha256, load_model, long_analysis_metadata


HOP = 512
FFT_SIZE = 1024
LONG_ANALYSIS_FFT_SIZE = 2048
SAMPLE_RATE = 44100
STATE_NAMES = {
    "c126": ("past_audio", "fusion_hidden"),
    "c191": ("past_audio", "fusion_hidden", "c130_history", "previous_hidden",
             "adapter_valid", "raw_parent_history", "emitted_db_history"),
}


def state_descriptor(model):
    """Return the batch-one state ABI selected by the actual loaded model."""
    names = STATE_NAMES[model.kind]
    analysis = long_analysis_metadata(model) if model.kind == "c191" else None
    if analysis is not None:
        if model.kind != "c191":
            raise ValueError("Long analysis is supported only for C191")
        names = (*names, "older_audio")
    states = model.initial_state(1, device="cpu")
    if len(states) != len(names) or len(set(names)) != len(names):
        raise ValueError("Model state count differs from its declared variant")
    if analysis is not None and tuple(states[-1].shape) != (1, 2, 1024):
        raise ValueError("Long analysis requires older_audio state [1,2,1024]")
    return tuple((name, tuple(value.shape)) for name, value in zip(names, states))


class _RFFT(torch.autograd.Function):
    """Native eager RFFT lowered to real-valued ONNX DFT, opset 17."""

    @staticmethod
    def forward(ctx, audio):
        spectrum = torch.fft.rfft(audio, n=FFT_SIZE, dim=-1)
        return torch.stack((spectrum.real, spectrum.imag), dim=-1)

    @staticmethod
    def symbolic(graph, audio):
        axis = graph.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
        values = graph.op("Unsqueeze", audio, axis)
        length = graph.op("Constant", value_t=torch.tensor(FFT_SIZE, dtype=torch.int64))
        result = graph.op("DFT", values, length, axis_i=1, inverse_i=0, onesided_i=1)
        sizes = audio.type().sizes()
        return result.setType(audio.type().with_sizes([sizes[0], FFT_SIZE // 2 + 1, 2]))


class _LongAnalysisRFFT(torch.autograd.Function):
    """Separate orthonormal 2048-point forward transform for analysis only."""

    @staticmethod
    def forward(ctx, audio):
        spectrum = torch.fft.rfft(audio, n=LONG_ANALYSIS_FFT_SIZE, dim=-1, norm="ortho")
        return torch.stack((spectrum.real, spectrum.imag), dim=-1)

    @staticmethod
    def symbolic(graph, audio):
        axis = graph.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
        values = graph.op("Unsqueeze", audio, axis)
        length = graph.op("Constant", value_t=torch.tensor(LONG_ANALYSIS_FFT_SIZE, dtype=torch.int64))
        transformed = graph.op("DFT", values, length, axis_i=1, inverse_i=0, onesided_i=1)
        normalization = graph.op("Constant", value_t=torch.tensor(LONG_ANALYSIS_FFT_SIZE**0.5, dtype=torch.float32))
        result = graph.op("Div", transformed, normalization)
        sizes = audio.type().sizes()
        return result.setType(audio.type().with_sizes([sizes[0], LONG_ANALYSIS_FFT_SIZE // 2 + 1, 2]))


class _IRFFT(torch.autograd.Function):
    """Explicit Hermitian completion avoids one-sided inverse DFT ambiguity."""

    @staticmethod
    def forward(ctx, spectrum):
        value = torch.complex(spectrum[..., 0], spectrum[..., 1])
        return torch.fft.irfft(value, n=FFT_SIZE, dim=1)

    @staticmethod
    def symbolic(graph, spectrum):
        reverse = graph.op("Constant", value_t=torch.arange(FFT_SIZE // 2 - 1, 0, -1, dtype=torch.int64))
        reflected = graph.op("Gather", spectrum, reverse, axis_i=1)
        sign = graph.op("Constant", value_t=torch.tensor([1.0, -1.0], dtype=torch.float32))
        reflected = graph.op("Mul", reflected, sign)
        full = graph.op("Concat", spectrum, reflected, axis_i=1)
        length = graph.op("Constant", value_t=torch.tensor(FFT_SIZE, dtype=torch.int64))
        inverse = graph.op("DFT", full, length, axis_i=1, inverse_i=1, onesided_i=0)
        real_index = graph.op("Constant", value_t=torch.tensor(0, dtype=torch.int64))
        result = graph.op("Gather", inverse, real_index, axis_i=2)
        sizes = spectrum.type().sizes()
        return result.setType(spectrum.type().with_sizes([sizes[0], FFT_SIZE]))


class CurrentChunkSTFTForONNX(nn.Module):
    """Two native analysis windows and current-half synthesis, without OLA."""

    def __init__(self, original):
        super().__init__()
        if (original.n_fft, original.win_length, original.hop_length) != (1024, 1024, 512):
            raise ValueError("The latency11 exporter requires FFT/window/hop 1024/1024/512")
        if not original.causal_current_chunk:
            raise ValueError("Expected current-chunk synthesis")
        self.register_buffer("window", original.window.detach().clone())
        self.register_buffer("current_chunk_analysis_window", original.current_chunk_analysis_window.detach().clone())

    @staticmethod
    def _spectrum(audio, window):
        values = _RFFT.apply(audio * window).unsqueeze(2)
        return existing_export.FakeComplexTensor(values, eps=0.0)

    def forward(self, audio):
        value = self._spectrum(audio, self.window)
        return value, value.abs()

    def current_chunk_carrier(self, audio):
        return self._spectrum(audio, self.current_chunk_analysis_window)

    def inverse(self, spectrum, is_streaming=False):
        if not is_streaming:
            raise ValueError("The export STFT accepts streaming callbacks only")
        values = spectrum.view_as_real() if isinstance(spectrum, existing_export.FakeComplexTensor) else spectrum
        return _IRFFT.apply(values.squeeze(2))[:, HOP:]


class LongAnalysisForONNX(nn.Module):
    """Real-valued analysis for one fixed 512-sample export callback."""

    def __init__(self, original):
        super().__init__()
        if tuple(original.weight.shape) != (500, 2050) or tuple(original.window.shape) != (2048,):
            raise ValueError("Expected the mag2048-v1 analysis projection and window")
        self.weight = nn.Parameter(original.weight.detach().clone(), requires_grad=original.weight.requires_grad)
        self.register_buffer("window", original.window.detach().clone())

    def forward(self, joined):
        if joined.ndim != 3 or joined.shape[1:] != (2, LONG_ANALYSIS_FFT_SIZE):
            raise ValueError("Exported long analysis requires one [B,2,2048] frame")
        batch = joined.shape[0]
        with torch.autocast(device_type=joined.device.type, enabled=False):
            spectrum = _LongAnalysisRFFT.apply((joined.float() * self.window.float()).reshape(-1, LONG_ANALYSIS_FFT_SIZE))
            magnitude = torch.sqrt(spectrum[..., 0].square() + spectrum[..., 1].square())
            features = magnitude.reshape(batch, 1, 2050)
        return torch.nn.functional.linear(features, self.weight)


@contextmanager
def core_export_patches(core):
    """Patch the actual external core module and restore every symbol on exit."""
    module = sys.modules[type(core).__module__]
    replacements = {
        "multiply": existing_export._onnx_multiply,
        "divide": existing_export._onnx_divide,
        "repeat": existing_export._onnx_repeat,
        "rearrange": existing_export._onnx_rearrange,
        "view_as_real": existing_export._onnx_view_as_real,
        "view_as_complex": existing_export._onnx_view_as_complex,
    }
    saved = []
    try:
        for name, replacement in replacements.items():
            if hasattr(module, name):
                saved.append((module, name, getattr(module, name)))
                setattr(module, name, replacement)
        for name, replacement in (
            ("view_as_real", existing_export._onnx_view_as_real),
            ("view_as_complex", existing_export._onnx_view_as_complex),
            ("polar", existing_export._onnx_polar),
        ):
            saved.append((torch, name, getattr(torch, name)))
            setattr(torch, name, replacement)
        yield
    finally:
        for owner, name, value in reversed(saved):
            setattr(owner, name, value)


def patch_export_copy(model):
    """Make an export copy; the caller's trainable model is left intact."""
    patched = copy.deepcopy(model).cpu().eval()
    core = patched.core
    if core.use_branch_rnns or not core.spec_branch_use_phase:
        raise ValueError("Expected retained phase-aware core without branch RNNs")
    core.stft = CurrentChunkSTFTForONNX(core.stft)
    core.conv_decode = existing_export.ConvTranspose1DWithHannWindowForONNX(core.conv_decode)
    existing_export.replace_rmsnorm_layers(core)
    existing_export.replace_rearrange_layers(core)
    core.fusion_branch = existing_export.OneFrameGRUForONNX(core.fusion_branch)
    if long_analysis_metadata(patched) is not None:
        patched.engine.long_analysis = LongAnalysisForONNX(patched.engine.long_analysis)
    return patched


class StreamingWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio_chunk, *states):
        output, next_states = self.model.forward_chunk(audio_chunk, states)
        return (output, *next_states)


def _verification_inputs(hops, audio_paths):
    rng = np.random.default_rng(20260905)
    samples = hops * HOP
    time_axis = np.arange(samples, dtype=np.float64) / SAMPLE_RATE
    music_like = np.stack([
        0.08 * np.sin(2 * np.pi * 55 * time_axis + phase)
        + 0.04 * np.sin(2 * np.pi * 173.31 * time_axis + phase)
        + 0.02 * np.sin(2 * np.pi * 997.3 * time_axis + phase)
        for phase in (0.23, 0.71)
    ])
    music_like += rng.normal(0.0, 0.005, music_like.shape)
    yield "multitone_noise_then_silence", np.pad(music_like.astype(np.float32), ((0, 0), (0, 8 * HOP)))
    yield "silence_from_reset", np.zeros((2, 8 * HOP), dtype=np.float32)
    if audio_paths:
        import soundfile as sf

        for path in audio_paths:
            audio, rate = sf.read(path, frames=samples, always_2d=True, dtype="float32")
            if rate != SAMPLE_RATE or audio.shape[1] != 2:
                raise ValueError(f"Verification audio must be stereo 44.1kHz: {path}")
            audio = audio.T
            audio = np.pad(audio, ((0, 0), (0, (-audio.shape[-1]) % HOP + 8 * HOP)))
            yield str(Path(path).resolve()), np.ascontiguousarray(audio)


def verify_onnx(model, path, *, hops=48, audio_paths=(), threads=2,
                waveform_atol=1e-4, waveform_rms_atol=1e-5, state_atol=5e-4):
    """Compare independent native/ORT states through music and silence."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    descriptor = state_descriptor(model)
    state_names = [name for name, _ in descriptor]
    input_names = ["audio_chunk", *state_names]
    output_names = ["separated_chunk", *("next_" + name for name in state_names)]
    if [v.name for v in session.get_inputs()] != input_names or [v.name for v in session.get_outputs()] != output_names:
        raise ValueError("Exported state input/output names changed")
    input_shapes = {value.name: list(value.shape) for value in session.get_inputs()}
    expected_inputs = {"audio_chunk": [1, 2, HOP], **{name: list(shape) for name, shape in descriptor}}
    if input_shapes != expected_inputs:
        raise ValueError(f"Exported inputs differ from the model state ABI: {input_shapes}")
    output_shapes = {value.name: list(value.shape) for value in session.get_outputs()}
    if any(not isinstance(size, int) or size <= 0 for shape in output_shapes.values() for size in shape):
        raise ValueError(f"Expected static output dimensions for the callback ABI: {output_shapes}")
    expected_outputs = {"separated_chunk": [1, 4, 2, HOP], **{"next_" + name: list(shape) for name, shape in descriptor}}
    if output_shapes != expected_outputs:
        raise ValueError(f"Exported outputs differ from the model state ABI: {output_shapes}")
    rows = []
    with torch.inference_mode():
        for name, audio in _verification_inputs(hops, audio_paths):
            native_state = model.initial_state(1, device="cpu")
            ort_state = [value.numpy().copy() for value in native_state]
            maximum, rms, closure = 0.0, 0.0, 0.0
            per_state = np.zeros(len(native_state))
            first_output = None
            for index in range(audio.shape[-1] // HOP):
                chunk = np.ascontiguousarray(audio[:, index * HOP:(index + 1) * HOP][None])
                expected, native_state = model.forward_chunk(torch.from_numpy(chunk), native_state)
                outputs = session.run(output_names, dict(zip(input_names, [chunk, *ort_state])))
                if len(native_state) != len(state_names) or len(outputs) != len(output_names):
                    raise ValueError("A callback changed the declared state count")
                for output_name, observed, reference in zip(output_names, outputs, (expected, *native_state)):
                    if list(observed.shape) != output_shapes[output_name] or observed.shape != tuple(reference.shape):
                        raise ValueError(f"Output shape differs from the declared/native ABI: {output_name}")
                if not all(np.isfinite(value).all() for value in outputs):
                    raise FloatingPointError(f"Nonfinite ORT output/state in {name} callback {index}")
                if not torch.isfinite(expected).all() or not all(torch.isfinite(value).all() for value in native_state):
                    raise FloatingPointError(f"Nonfinite native output/state in {name} callback {index}")
                difference = outputs[0].astype(np.float64) - expected.numpy().astype(np.float64)
                maximum = max(maximum, float(np.abs(difference).max()))
                rms = max(rms, float(np.sqrt(np.mean(difference**2, axis=(2, 3))).max()))
                closure = max(closure, float(np.abs(outputs[0].sum(axis=1) - chunk).max()))
                for state_index, (observed, reference) in enumerate(zip(outputs[1:], native_state)):
                    physical_scale = 2.0**18 if model.kind == "c191" and state_index == 1 else 1.0
                    state_error = float(np.abs(observed.astype(np.float64) - reference.numpy()).max()) * physical_scale
                    per_state[state_index] = max(per_state[state_index], state_error)
                if first_output is None:
                    first_output = [value.copy() for value in outputs]
                ort_state = outputs[1:]
            reset = [value.numpy() for value in model.initial_state(1, device="cpu")]
            first_chunk = np.ascontiguousarray(audio[:, :HOP][None])
            replay = session.run(output_names, dict(zip(input_names, [first_chunk, *reset])))
            reset_exact = all(np.array_equal(left, right) for left, right in zip(first_output, replay))
            passed = maximum <= waveform_atol and rms <= waveform_rms_atol and closure <= 1e-6 and bool(np.all(per_state <= state_atol)) and reset_exact
            rows.append({
                "input": name, "callbacks": audio.shape[-1] // HOP,
                "waveform_max_abs": maximum, "maximum_stem_callback_rms_error": rms,
                "reconstruction_max_abs": closure,
                "state_max_abs_decoded_units": dict(zip(state_names, per_state.tolist())),
                "reset_replay_bit_exact": reset_exact, "passed": passed,
            })
    return {
        "passed": all(row["passed"] for row in rows), "device": "cpu",
        "onnxruntime_version": ort.__version__, "torch_version": torch.__version__,
        "output_shapes": output_shapes, "output_shapes_checked_every_callback": True,
        "independent_recurrent_trajectories": True,
        "gru_state_comparison": "decoded internal units; C191 public values multiplied by 2**18",
        "tolerances": {"waveform_max_abs": waveform_atol, "stem_callback_rms": waveform_rms_atol,
                       "state_max_abs_decoded_units": state_atol, "reconstruction_max_abs": 1e-6},
        "cases": rows,
    }


def export_model(kind, checkpoint, output, *, threads=2, verify_hops=48, verify_audio=()):
    import onnx

    if verify_hops < 1:
        raise ValueError("verify_hops must be positive")
    torch.set_num_threads(threads)
    started = time.monotonic()
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    model = load_model(kind, checkpoint, device="cpu")
    exported = patch_export_copy(model)
    wrapper = StreamingWrapper(exported).eval()
    descriptor = state_descriptor(model)
    state_names = [name for name, _ in descriptor]
    states = exported.initial_state(1, device="cpu")
    if tuple(tuple(value.shape) for value in states) != tuple(shape for _, shape in descriptor):
        raise ValueError("Export copy changed the model state ABI")
    args = (torch.zeros(1, 2, HOP), *states)
    input_names = ["audio_chunk", *state_names]
    output_names = ["separated_chunk", *("next_" + name for name in state_names)]
    handle, temporary_name = tempfile.mkstemp(prefix=output.stem + ".", suffix=".onnx", dir=output.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with torch.inference_mode(), core_export_patches(exported.core):
            example_outputs = wrapper(*args)
            torch.onnx.export(wrapper, args, str(temporary), export_params=True,
                              opset_version=17, do_constant_folding=True,
                              input_names=input_names, output_names=output_names,
                              dynamo=False, external_data=False)
        graph = onnx.load(str(temporary), load_external_data=False)
        if [value.name for value in graph.graph.output] != output_names or len(example_outputs) != len(output_names):
            raise ValueError("Exported outputs differ from the batch-one example ABI")
        # Slice inference can leave symbolic lengths even for this fixed callback.
        # Seal only output type shapes; computation nodes and weights stay intact.
        for value, example in zip(graph.graph.output, example_outputs):
            shape = value.type.tensor_type.shape
            shape.ClearField("dim")
            for size in example.shape:
                shape.dim.add().dim_value = int(size)
        metadata = {
            "hs_tasnet.kind": kind, "hs_tasnet.mode": "streaming",
            "hs_tasnet.sample_rate": "44100", "hs_tasnet.hop_samples": "512",
            "hs_tasnet.algorithmic_latency_samples": "512",
            "hs_tasnet.graph_output_delay_samples": "0",
            "hs_tasnet.host_scheduling_delay_samples": "512",
            "hs_tasnet.alignment_samples": "0", "hs_tasnet.future_context_samples": "0",
            "hs_tasnet.flush_required": "false", "hs_tasnet.output_policy": "current mixture minus DBV in Other",
            "hs_tasnet.source_order": "drums,bass,vocals,other",
            "hs_tasnet.state_names": json.dumps(state_names),
            "hs_tasnet.state_shapes": json.dumps([list(shape) for _, shape in descriptor]),
            "hs_tasnet.public_fusion_state_scale": str(2.0**-18 if kind == "c191" else 1.0),
            "hs_tasnet.effective_source_gains": json.dumps((2 * model.output_source_scales).tolist()),
            "hs_tasnet.checkpoint_sha256": file_sha256(checkpoint),
            "hs_tasnet.core_source_sha256": file_sha256(CORE_SOURCE),
            "hs_tasnet.exporter_sha256": file_sha256(__file__),
            "hs_tasnet.export_helpers_sha256": file_sha256(existing_export.__file__),
            "hs_tasnet.fft_implementation": "ONNX DFT; explicit Hermitian completion for inverse",
        }
        analysis = long_analysis_metadata(model)
        if analysis is not None:
            from research.direct import latency_long_analysis

            metadata.update({
                "hs_tasnet.c191_long_analysis": json.dumps(analysis, sort_keys=True),
                "hs_tasnet.analysis_fft_samples": str(LONG_ANALYSIS_FFT_SIZE),
                "hs_tasnet.analysis_window_samples": str(LONG_ANALYSIS_FFT_SIZE),
                "hs_tasnet.analysis_history_samples": "1536",
                "hs_tasnet.older_audio_state_samples": "1024",
                "hs_tasnet.synthesis_fft_samples": str(FFT_SIZE),
                "hs_tasnet.long_analysis_source_sha256": file_sha256(latency_long_analysis.__file__),
                "hs_tasnet.long_analysis_fft_implementation": "ONNX forward DFT2048 / sqrt(2048); FP32 magnitude; LR-frequency projection",
            })
        onnx.helper.set_model_props(graph, metadata)
        onnx.save(graph, str(temporary))
        onnx.checker.check_model(str(temporary), full_check=True)
        validation = verify_onnx(model, temporary, hops=verify_hops, audio_paths=verify_audio, threads=threads)
        receipt = {"schema_version": 1, "kind": kind, "checkpoint": str(checkpoint),
                   "checkpoint_sha256": file_sha256(checkpoint), "output": str(output),
                   "onnx_sha256": file_sha256(temporary), "metadata": metadata,
                   "verification": validation, "elapsed_seconds": time.monotonic() - started,
                   "native_host_timing_qualified": False}
        if not validation["passed"]:
            failure_path = output.with_suffix(".failed-verification.json")
            failure_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
            raise RuntimeError(f"CPU ONNX parity failed; see {failure_path}")
        temporary.replace(output)
        output.with_suffix(".verification.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        output.with_suffix(output.suffix + ".sha256").write_text(f"{receipt['onnx_sha256']}  {output.name}\n")
        return receipt
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("c126", "c191"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--verify-hops", type=int, default=48)
    parser.add_argument("--verify-audio", type=Path, action="append", default=[])
    args = parser.parse_args()
    receipt = export_model(args.kind, args.checkpoint, args.output, threads=args.threads,
                           verify_hops=args.verify_hops, verify_audio=args.verify_audio)
    print(json.dumps({"output": receipt["output"], "onnx_sha256": receipt["onnx_sha256"],
                      "verification": receipt["verification"], "elapsed_seconds": receipt["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
