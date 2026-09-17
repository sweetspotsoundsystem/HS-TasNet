"""Isolated CPU export and three-way numerical verification of trained OLA512.

The static graph receives 256 samples and emits the previous physical hop.
Its four explicit states include both synthesis tails. No host queue is
implemented here; adding a separately qualified 256-sample queue would make
the intended total latency 512 samples. No checkpoint/model/export workload
is created on import; Torch and the existing helper modules are imported.

Run in a fresh process with CUDA_VISIBLE_DEVICES='' and an explicitly
authenticated ola512-inference-v1 snapshot. Existing files are never replaced.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import export_onnx as export_helpers
from research.direct import latency_ola512 as ola


STATE_NAMES = tuple(ola.OLA512State._fields)
STATE_SHAPES = ((1, 2, 768), (2, 1, 1000), (1, 4, 2, 256), (1, 4, 2, 256))
INPUT_NAMES = ("audio_chunk", *STATE_NAMES)
OUTPUT_NAMES = ("separated_chunk", *("next_" + name for name in STATE_NAMES))
INPUT_SHAPES = ((1, 2, 256), *STATE_SHAPES)
OUTPUT_SHAPES = ((1, 4, 2, 256), *STATE_SHAPES)
NATIVE_SOURCE_SCALES = (0.5, 0.5, 0.45, 0.56)
TOLERANCES = {"waveform_max_abs": 1e-4, "stem_callback_rms": 1e-5,
              "state_max_abs_decoded_units": 5e-4, "reconstruction_max_abs": 1e-6}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(model):
    """Same sorted CPU tensor identity used by the OLA checkpoint evaluator."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def require_sha256(value, label):
    require(isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value),
            f"{label} must be an explicit lowercase SHA-256")


def load_checkpoint(checkpoint, *, expected_sha256, expected_model_state_sha256,
                    expected_training_plan_sha256, expected_step):
    """Authenticate bytes before weights-only CPU loading; never initialize C91."""
    for label, value in (("checkpoint", expected_sha256),
                         ("model state", expected_model_state_sha256),
                         ("training plan", expected_training_plan_sha256)):
        require_sha256(value, label)
    require(type(expected_step) is int and expected_step > 0, "Expected step must be positive")
    checkpoint = Path(checkpoint).resolve(strict=True)
    require(sha256(checkpoint) == expected_sha256, "Checkpoint SHA-256 differs before loading")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(isinstance(payload, dict) and set(payload) == {
        "schema", "step", "model", "model_state_sha256", "provenance", "architecture", "plan_sha256"
    }, "Unexpected OLA inference snapshot schema")
    require(payload["schema"] == "ola512-inference-v1"
            and type(payload["step"]) is int and payload["step"] == expected_step
            and payload["plan_sha256"] == expected_training_plan_sha256
            and payload["model_state_sha256"] == expected_model_state_sha256,
            "Snapshot step, plan or model-state identity differs")
    provenance = payload["provenance"]
    require(isinstance(provenance, dict)
            and provenance.get("version") == ola.VERSION
            and provenance.get("training_updates") == expected_step
            and provenance.get("training_plan_sha256") == expected_training_plan_sha256
            and provenance.get("initialization") == "authenticated_c91_right_baked512_then_new_hann"
            and provenance.get("reference_model_modified") is False
            and provenance.get("equivalence_claimed") is False,
            "Snapshot training/initializer provenance differs")
    lineage = provenance.get("base_lineage")
    override = provenance.get("applied_decoder_override")
    require(isinstance(lineage, dict) and lineage.get("version") == ola.VERSION
            and lineage.get("reference_sha256") == ola.REFINED_C91_SHA256
            and lineage.get("initialization") == "authenticated_c91_folded512_then_new_hann"
            and isinstance(override, dict)
            and override.get("only_changed_tensor") == "waveform_decoder_weight"
            and override.get("rule") == "old_baked[...,512:1024]"
            and override.get("all_other_tensors_bit_exact") is True
            and override.get("output_source_scales_unchanged") is True
            and override.get("old_window_division_applied") is False,
            "Snapshot C91 lineage or baked-right decoder override differs")
    # Construction deliberately occurs outside inference_mode, matching the
    # established FP32 evaluator and avoiding inference-tensor kernel changes.
    with torch.device("cpu"):
        model = ola.OLA512Model()
    fixed = {name: value.clone() for name, value in model.named_buffers()
             if name != "output_source_scales"}
    values = payload["model"]
    require(isinstance(values, dict) and set(values) == set(model.state_dict()),
            "Snapshot tensor inventory differs")
    require(all(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                and value.dtype == torch.float32 and bool(torch.isfinite(value).all())
                for value in values.values()), "Snapshot tensors must be finite CPU FP32")
    model.load_state_dict(values, strict=True)
    model.provenance = copy.deepcopy(provenance)
    model.eval().requires_grad_(False)
    require(model.architecture_metadata == payload["architecture"], "OLA architecture metadata differs")
    require(len(list(model.parameters())) == 21 and len(model.state_dict()) == 26,
            "OLA parameter/buffer inventory differs")
    require(all(torch.equal(value, fixed[name]) for name, value in model.named_buffers()
                if name in fixed), "Fixed analysis/synthesis buffers changed")
    expected_scales = torch.tensor(NATIVE_SOURCE_SCALES, dtype=torch.float32, device="cpu")
    require(torch.equal(model.output_source_scales, expected_scales), "Native C91-derived gains changed")
    require(state_sha256(model) == expected_model_state_sha256, "Loaded tensor-state SHA-256 differs")
    require(sha256(checkpoint) == expected_sha256, "Checkpoint changed during loading")
    return model, {"checkpoint": str(checkpoint), "checkpoint_sha256": expected_sha256,
                   "checkpoint_bytes": checkpoint.stat().st_size, "step": expected_step,
                   "model_state_sha256": expected_model_state_sha256,
                   "training_plan_sha256": expected_training_plan_sha256,
                   "architecture": payload["architecture"], "provenance": provenance}


class _RFFT1024(torch.autograd.Function):
    @staticmethod
    def forward(ctx, audio):
        spectrum = torch.fft.rfft(audio, n=1024, dim=-1)
        return torch.stack((spectrum.real, spectrum.imag), dim=-1)

    @staticmethod
    def symbolic(graph, audio):
        axis = graph.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
        values = graph.op("Unsqueeze", audio, axis)
        length = graph.op("Constant", value_t=torch.tensor(1024, dtype=torch.int64))
        result = graph.op("DFT", values, length, axis_i=1, inverse_i=0, onesided_i=1)
        return result.setType(audio.type().with_sizes([audio.type().sizes()[0], 513, 2]))


class _RFFT512(torch.autograd.Function):
    @staticmethod
    def forward(ctx, audio):
        spectrum = torch.fft.rfft(audio, n=512, dim=-1)
        return torch.stack((spectrum.real, spectrum.imag), dim=-1)

    @staticmethod
    def symbolic(graph, audio):
        axis = graph.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
        values = graph.op("Unsqueeze", audio, axis)
        length = graph.op("Constant", value_t=torch.tensor(512, dtype=torch.int64))
        result = graph.op("DFT", values, length, axis_i=1, inverse_i=0, onesided_i=1)
        return result.setType(audio.type().with_sizes([audio.type().sizes()[0], 257, 2]))


class _IRFFT512(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spectrum):
        value = torch.complex(spectrum[..., 0], spectrum[..., 1])
        return torch.fft.irfft(value, n=512, dim=1)

    @staticmethod
    def symbolic(graph, spectrum):
        # Torch irfft ignores imaginary DC/Nyquist values. Do so explicitly;
        # retain each endpoint once and reflect only interior bins 255..1.
        endpoint_mask = torch.ones(257, 2, dtype=torch.float32)
        endpoint_mask[0, 1] = 0.0
        endpoint_mask[-1, 1] = 0.0
        real_endpoints = graph.op("Mul", spectrum, graph.op("Constant", value_t=endpoint_mask))
        indices = graph.op("Constant", value_t=torch.arange(255, 0, -1, dtype=torch.int64))
        reflected = graph.op("Gather", real_endpoints, indices, axis_i=1)
        sign = graph.op("Constant", value_t=torch.tensor([1.0, -1.0], dtype=torch.float32))
        reflected = graph.op("Mul", reflected, sign)
        full = graph.op("Concat", real_endpoints, reflected, axis_i=1)
        length = graph.op("Constant", value_t=torch.tensor(512, dtype=torch.int64))
        # ONNX inverse DFT includes 1/N, matching Torch's default backward norm.
        inverse = graph.op("DFT", full, length, axis_i=1, inverse_i=1, onesided_i=0)
        real_index = graph.op("Constant", value_t=torch.tensor(0, dtype=torch.int64))
        result = graph.op("Gather", inverse, real_index, axis_i=2)
        return result.setType(spectrum.type().with_sizes([spectrum.type().sizes()[0], 512]))


class OLA512StreamingWrapper(nn.Module):
    """One literal hop of the original math, with explicit real/imag transforms."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio_chunk, audio_history, fusion_hidden,
                spectral_numerator_tail, waveform_tail):
        model = self.model
        joined = torch.cat((audio_history, audio_chunk), dim=-1)
        feature_ri = _RFFT1024.apply((joined * model.analysis_window).reshape(2, 1024))
        feature_ri = feature_ri.reshape(1, 2, 1, 513, 2)
        packed = feature_ri.permute(0, 2, 1, 3, 4).reshape(1, 1, 2052)
        spec = model.spec_encode(packed)
        to_relu, to_sigmoid = model.conv_encode(joined).chunk(2, dim=1)
        basis = to_relu.relu() * to_sigmoid.sigmoid()
        waveform = model.basis_to_embed(basis).transpose(1, 2)
        fusion_input = torch.cat((spec, waveform), dim=-1)
        recurrent, next_physical_hidden = model.fusion_branch(
            fusion_input, fusion_hidden / ola.PUBLIC_FUSION_SCALE)
        fused = fusion_input + recurrent
        fused_spec, fused_waveform = fused.chunk(2, dim=-1)
        spec = fused_spec + spec
        waveform = fused_waveform + waveform

        spec_logits = model.to_spec_masks(model.spec_norm(spec))
        spec_logits = spec_logits.reshape(1, 1, 2, 257, 2, 4)
        spec_masks = model._residual_source_softmax(spec_logits).permute(0, 2, 1, 3, 4, 5)
        carrier = _RFFT512.apply((joined[..., 512:] * model.synthesis.window).reshape(2, 512))
        carrier = carrier.reshape(1, 2, 1, 257, 2)
        masked_ri = carrier.unsqueeze(-1) * spec_masks
        source_ri = masked_ri.permute(0, 5, 1, 2, 3, 4).contiguous().reshape(8, 257, 2)
        spectral_frames = _IRFFT512.apply(source_ri).reshape(1, 4, 2, 512)
        spectral_frames = spectral_frames * model.synthesis.window
        spectral = (spectral_frames[..., :256] + spectral_numerator_tail)
        spectral = spectral / model.synthesis.spectral_denominator
        next_spectral_tail = spectral_frames[..., 256:].clone()

        waveform_logits = model.to_waveform_masks(model.waveform_norm(waveform))
        waveform_logits = waveform_logits.reshape(1, 1, 4, 1500).transpose(-1, -2)
        waveform_masks = model._residual_source_softmax(waveform_logits)
        source_basis = basis.transpose(1, 2).unsqueeze(-1) * waveform_masks
        source_basis = source_basis.permute(0, 3, 1, 2)
        decoded = F.linear(source_basis, model.waveform_decoder_weight.flatten(1).t(), bias=None)
        decoded = decoded.reshape(1, 4, 1, 2, 512).permute(0, 1, 3, 2, 4)
        windowed = (decoded * model.synthesis.window).reshape(1, 4, 2, 512)
        waveform_audio = windowed[..., :256] + waveform_tail
        next_waveform_tail = windowed[..., 256:].clone()

        raw = (spectral + waveform_audio) * model.output_source_scales[None, :, None, None]
        delayed_mixture = audio_history[..., -256:]
        retained = raw[:, :3]
        deployed = torch.cat((retained, delayed_mixture.unsqueeze(1)
                              - retained.sum(dim=1, keepdim=True)), dim=1)
        return (deployed, joined[..., -768:].clone(),
                next_physical_hidden * ola.PUBLIC_FUSION_SCALE,
                next_spectral_tail, next_waveform_tail)


def make_export_copy(model):
    patched = copy.deepcopy(model).cpu().eval()
    patched.fusion_branch = export_helpers.OneFrameGRUForONNX(patched.fusion_branch)
    export_helpers.replace_rmsnorm_layers(patched)
    require(state_sha256(patched) == state_sha256(model), "Export copy changed checkpoint tensor bytes")
    return OLA512StreamingWrapper(patched).eval()


def _initial_states(nonzero=False):
    states = [np.zeros(shape, dtype=np.float32) for shape in STATE_SHAPES]
    if nonzero:
        rng = np.random.default_rng(2026090617)
        scales = (0.04, 0.1 * ola.PUBLIC_FUSION_SCALE, 0.001, 0.001)
        states = [rng.normal(0.0, scale, shape).astype(np.float32)
                  for shape, scale in zip(STATE_SHAPES, scales, strict=True)]
    return states


def verification_cases(hops, audio_paths):
    """Physical input lengths exclude padding and the single final flush."""
    require(type(hops) is int and hops >= 8, "Verification requires at least eight signal hops")
    rng = np.random.default_rng(2026090618)
    count = hops * ola.HOP + 37
    t = np.arange(count, dtype=np.float64) / ola.SAMPLE_RATE
    signal = np.stack([sum(amplitude * np.sin(2 * np.pi * frequency * t + phase)
                           for amplitude, frequency in ((.08, 55), (.04, 173.31), (.02, 997.3)))
                       for phase in (.23, .71)])
    signal += rng.normal(0.0, .005, signal.shape)
    signal = signal.astype(np.float32)
    yield {"name": "multitone_noise_then_silence_partial", "audio": np.pad(
        signal, ((0, 0), (0, 8 * ola.HOP))), "initial_states": _initial_states()}
    yield {"name": "silence_from_reset", "audio": np.zeros((2, 8 * ola.HOP + 17), np.float32),
           "initial_states": _initial_states()}
    endpoint = np.stack((.03 + .07 * (-1.0) ** np.arange(count) + .02 * np.sin(2 * np.pi * 55.3 * t),
                         -.04 - .05 * (-1.0) ** np.arange(count) + .03 * np.cos(2 * np.pi * 997.7 * t)))
    yield {"name": "dc_nyquist_offgrid_partial", "audio": endpoint.astype(np.float32),
           "initial_states": _initial_states()}
    impulses = np.zeros((2, 7 * ola.HOP + 37), np.float32)
    for index, position in enumerate((0, 1, 255, 256, 511, 512, impulses.shape[-1] - 1)):
        impulses[:, position] = ((index + 1) / 16.0, -(index + 1) / 19.0)
    yield {"name": "boundary_and_last_sample_impulses", "audio": impulses,
           "initial_states": _initial_states()}
    yield {"name": "nonzero_history_hidden_and_both_tails", "audio": signal[:, :8 * ola.HOP + 31],
           "initial_states": _initial_states(nonzero=True)}
    if audio_paths:
        import soundfile as sf
        for path in audio_paths:
            path = Path(path).resolve(strict=True)
            before = sha256(path)
            audio, rate = sf.read(path, frames=count, dtype="float32", always_2d=True)
            require(rate == ola.SAMPLE_RATE and audio.shape[1] == 2 and audio.shape[0] > 0,
                    f"Verification music must be nonempty stereo 44.1 kHz: {path}")
            require(sha256(path) == before, "Music file changed while reading")
            yield {"name": f"music:{path}", "audio": np.ascontiguousarray(audio.T),
                   "initial_states": _initial_states(), "source": {"path": str(path), "sha256": before}}


def _validate_values(values, context):
    require(len(values) == len(OUTPUT_NAMES), f"Wrong output count: {context}")
    for name, value, shape in zip(OUTPUT_NAMES, values, OUTPUT_SHAPES, strict=True):
        require(value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all(),
                f"Nonfinite or incorrect output ABI: {context}/{name}")


def _run_case(model, wrapper, session, case):
    audio = np.ascontiguousarray(case["audio"], dtype=np.float32)
    require(audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 0
            and np.isfinite(audio).all(), "Invalid verification fixture audio")
    initial = case["initial_states"]
    native_state = ola.OLA512State(*(torch.from_numpy(value.copy()) for value in initial))
    copy_state = tuple(torch.from_numpy(value.copy()) for value in initial)
    ort_state = [value.copy() for value in initial]
    real_samples = audio.shape[-1]
    padding = (-real_samples) % ola.HOP
    padded = np.pad(audio, ((0, 0), (0, padding)))
    received = np.pad(padded, ((0, 0), (0, ola.HOP)))
    # This oracle is indexed from actual input plus the incoming physical history;
    # it does not use native/ORT next-state values to establish alignment.
    aligned = np.concatenate((initial[0][0, :, -ola.HOP:], padded), axis=-1)
    expected_history = initial[0].copy()
    pairs = (("native", "export_copy"), ("native", "ort"), ("export_copy", "ort"))
    errors = {f"{left}_vs_{right}": {"waveform_max_abs": 0.0, "stem_callback_rms": 0.0,
               "waveform_max_abs_by_stem": {name: 0.0 for name in ola.SOURCE_ORDER},
               "maximum_callback_rms_error_by_stem": {name: 0.0 for name in ola.SOURCE_ORDER},
               "squared_error_sum_by_stem": np.zeros(4, dtype=np.float64),
               "state_max_abs_decoded_units": {name: 0.0 for name in STATE_NAMES}}
              for left, right in pairs}
    backends = ("native", "export_copy", "ort")
    digests = {name: hashlib.sha256() for name in backends}
    closure = {name: 0.0 for name in backends}
    flush_closure = {}
    last_sample_error = {}
    recovered_samples = 0
    data_hops = padded.shape[-1] // ola.HOP
    with torch.inference_mode():
        for index in range(data_hops + 1):
            chunk = np.ascontiguousarray(received[:, index * ola.HOP:(index + 1) * ola.HOP][None])
            if index == data_hops:
                native_output, native_state = model.flush(native_state)
            else:
                native_output, native_state = model.forward_chunk(torch.from_numpy(chunk), native_state)
            copied = wrapper(torch.from_numpy(chunk), *copy_state)
            copy_state = copied[1:]
            actual = session.run(list(OUTPUT_NAMES), dict(zip(INPUT_NAMES, [chunk, *ort_state], strict=True)))
            ort_state = actual[1:]
            outputs = {"native": [value.detach().numpy() for value in (native_output, *native_state)],
                       "export_copy": [value.detach().numpy() for value in copied], "ort": actual}
            expected_history = np.concatenate((expected_history[..., ola.HOP:], chunk), axis=-1)
            expected_mixture = aligned[:, index * ola.HOP:(index + 1) * ola.HOP][None]
            for backend, values in outputs.items():
                _validate_values(values, f"{case['name']}/{index}/{backend}")
                require(np.array_equal(values[1], expected_history),
                        f"Physical history differs: {case['name']}/{index}/{backend}")
                for value in values:
                    digests[backend].update(np.ascontiguousarray(value).tobytes())
                mixture = values[0].sum(axis=1)
                error = float(np.abs(mixture.astype(np.float64) - expected_mixture).max())
                closure[backend] = max(closure[backend], error)
                if index == data_hops:
                    flush_closure[backend] = error
                    offset = (real_samples - 1) % ola.HOP
                    last_sample_error[backend] = float(np.abs(
                        mixture[0, :, offset].astype(np.float64) - audio[:, -1]).max())
            for left, right in pairs:
                row = errors[f"{left}_vs_{right}"]
                difference = outputs[left][0].astype(np.float64) - outputs[right][0].astype(np.float64)
                row["waveform_max_abs"] = max(row["waveform_max_abs"], float(np.abs(difference).max()))
                rms = float(np.sqrt(np.mean(difference ** 2, axis=(2, 3))).max())
                row["stem_callback_rms"] = max(row["stem_callback_rms"], rms)
                per_stem_max = np.abs(difference).max(axis=(0, 2, 3))
                per_stem_rms = np.sqrt(np.mean(difference ** 2, axis=(0, 2, 3)))
                row["squared_error_sum_by_stem"] += np.sum(difference ** 2, axis=(0, 2, 3))
                for stem_index, name in enumerate(ola.SOURCE_ORDER):
                    row["waveform_max_abs_by_stem"][name] = max(
                        row["waveform_max_abs_by_stem"][name], float(per_stem_max[stem_index]))
                    row["maximum_callback_rms_error_by_stem"][name] = max(
                        row["maximum_callback_rms_error_by_stem"][name], float(per_stem_rms[stem_index]))
                for state_index, name in enumerate(STATE_NAMES):
                    factor = 1.0 / ola.PUBLIC_FUSION_SCALE if name == "fusion_hidden" else 1.0
                    difference = (outputs[left][state_index + 1].astype(np.float64)
                                  - outputs[right][state_index + 1].astype(np.float64))
                    value = float(np.abs(difference).max()) * factor
                    row["state_max_abs_decoded_units"][name] = max(
                        row["state_max_abs_decoded_units"][name], value)
            if index > 0:
                recovered_samples += min(ola.HOP, max(0, real_samples - (index - 1) * ola.HOP))
    require(recovered_samples == real_samples, "Single-cut/flush physical sample accounting differs")
    for row in errors.values():
        per_stem_rms = np.sqrt(row.pop("squared_error_sum_by_stem") / ((data_hops + 1) * 2 * ola.HOP))
        row["trajectory_rms_error_by_stem"] = dict(zip(ola.SOURCE_ORDER, per_stem_rms.tolist(), strict=True))
        row["passed"] = (row["waveform_max_abs"] <= TOLERANCES["waveform_max_abs"]
                         and row["stem_callback_rms"] <= TOLERANCES["stem_callback_rms"]
                         and max(row["state_max_abs_decoded_units"].values())
                         <= TOLERANCES["state_max_abs_decoded_units"])
    return {"input": case["name"], "source": case.get("source"), "physical_samples": real_samples,
            "decoded_input_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
            "initial_state_sha256": [hashlib.sha256(value.tobytes()).hexdigest() for value in initial],
            "partial_hop_padding": padding, "data_hops": data_hops, "flush_hops": 1,
            "callbacks": data_hops + 1, "physical_samples_recovered": recovered_samples,
            "all_output_shapes_dtypes_finite_checked_every_callback": True,
            "history_shift_bit_exact_every_callback": True, "comparisons": errors,
            "reconstruction_max_abs": closure, "flush_reconstruction_max_abs": flush_closure,
            "last_real_sample_reconstruction_max_abs": last_sample_error,
            "all_output_and_state_trajectory_sha256": {name: value.hexdigest() for name, value in digests.items()},
            "passed": all(row["passed"] for row in errors.values())
                      and max(closure.values()) <= TOLERANCES["reconstruction_max_abs"]}


def verify_onnx(model, wrapper, path, *, hops=48, audio_paths=(), threads=1):
    """Independent native/export-copy/ORT recurrence; reset repeats whole traces."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    require(session.get_providers() == ["CPUExecutionProvider"], "Expected CPU provider only")
    for actual, names, shapes in ((session.get_inputs(), INPUT_NAMES, INPUT_SHAPES),
                                  (session.get_outputs(), OUTPUT_NAMES, OUTPUT_SHAPES)):
        require(tuple(value.name for value in actual) == names, "Graph ABI name/order differs")
        require(tuple(tuple(value.shape) for value in actual) == shapes
                and all(value.type == "tensor(float)" for value in actual), "Graph FP32/static shapes differ")
    rows = []
    for case in verification_cases(hops, audio_paths):
        first = _run_case(model, wrapper, session, case)
        replay = _run_case(model, wrapper, session, case)
        if case.get("source") is not None:
            require(sha256(case["source"]["path"]) == case["source"]["sha256"],
                    "Music source changed during native/export-copy/ORT verification and replay")
        exact = first["all_output_and_state_trajectory_sha256"] == replay["all_output_and_state_trajectory_sha256"]
        first["reset_replay_all_outputs_and_states_bit_exact"] = exact
        first["reset_replay_passed"] = replay["passed"]
        first["passed"] = first["passed"] and replay["passed"] and exact
        rows.append(first)
    return {"passed": all(row["passed"] for row in rows), "device": "cpu",
            "torch_version": torch.__version__, "onnxruntime_version": ort.__version__,
            "intra_op_threads": threads, "inter_op_threads": 1,
            "independent_recurrent_trajectories": ["native", "export_copy", "ort"],
            "gru_state_comparison": "Public values decoded by multiplying by 2**18",
            "closure_reference": "Independent incoming physical history/input timeline, delayed 256 samples",
            "tolerances": TOLERANCES, "cases": rows,
            "native_host_timing_qualified": False}


def _write_new_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def export_model(checkpoint, output, *, expected_sha256, expected_model_state_sha256,
                 expected_training_plan_sha256, expected_step, threads=1,
                 verify_hops=48, verify_audio=()):
    import onnx

    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Set CUDA_VISIBLE_DEVICES='' in a fresh CPU process")
    require(not torch.cuda.is_initialized(), "CUDA was already initialized; use a fresh CPU-only process")
    require(type(threads) is int and 1 <= threads <= 16, "Threads must be in [1,16]")
    require(type(verify_hops) is int and verify_hops >= 8, "Verification requires at least eight signal hops")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    output = Path(output).resolve()
    require(output.suffix == ".onnx", "Output must have an .onnx suffix")
    receipt_path = output.with_suffix(".verification.json")
    failed_receipt_path = output.with_suffix(".failed-verification.json")
    failed_graph_path = output.with_suffix(".failed.onnx")
    hash_path = output.with_suffix(".onnx.sha256")
    for path in (output, receipt_path, failed_receipt_path, failed_graph_path, hash_path):
        require(not path.exists(), f"Refusing to replace an existing artifact: {path}")
    started = time.monotonic()
    source_paths = {"exporter": Path(__file__).resolve(), "ola_model": Path(ola.__file__).resolve(),
                    "export_helpers": Path(export_helpers.__file__).resolve()}
    source_bindings = {name: {"path": str(path), "sha256": sha256(path)}
                       for name, path in source_paths.items()}
    model, identity = load_checkpoint(
        checkpoint, expected_sha256=expected_sha256, expected_model_state_sha256=expected_model_state_sha256,
        expected_training_plan_sha256=expected_training_plan_sha256, expected_step=expected_step)
    wrapper = make_export_copy(model)
    states = model.initial_state(1, device="cpu")
    require(tuple(tuple(value.shape) for value in states) == STATE_SHAPES, "Model state ABI differs")
    args = (torch.zeros(INPUT_SHAPES[0], dtype=torch.float32, device="cpu"), *states)
    metadata = {
        "hs_tasnet.kind": "ola512", "hs_tasnet.mode": "streaming",
        "hs_tasnet.architecture_version": ola.VERSION, "hs_tasnet.sample_rate": "44100",
        "hs_tasnet.hop_samples": "256", "hs_tasnet.analysis_fft_samples": "1024",
        "hs_tasnet.synthesis_fft_samples": "512", "hs_tasnet.analysis_history_samples": "768",
        "hs_tasnet.graph_output_delay_samples": "256", "hs_tasnet.alignment_samples": "256",
        "hs_tasnet.future_context_samples": "256", "hs_tasnet.future_callbacks_beyond_received_input": "0",
        "hs_tasnet.flush_required": "true", "hs_tasnet.flush_hops": "1",
        "hs_tasnet.initial_state": "all_zeros", "hs_tasnet.preroll": "discard_first_output_hop_after_reset",
        "hs_tasnet.external_host_queue_implemented": "false",
        "hs_tasnet.intended_external_host_queue_samples": "256",
        "hs_tasnet.intended_total_latency_samples": "512", "hs_tasnet.native_host_timing_qualified": "false",
        "hs_tasnet.output_policy": "previous physical mixture minus unchanged DBV in Other",
        "hs_tasnet.source_order": ",".join(ola.SOURCE_ORDER),
        "hs_tasnet.state_names": json.dumps(STATE_NAMES), "hs_tasnet.state_shapes": json.dumps(STATE_SHAPES),
        "hs_tasnet.public_fusion_state_scale": str(ola.PUBLIC_FUSION_SCALE),
        "hs_tasnet.output_source_scales": json.dumps(model.output_source_scales.tolist()),
        "hs_tasnet.checkpoint_sha256": expected_sha256, "hs_tasnet.model_state_sha256": expected_model_state_sha256,
        "hs_tasnet.training_plan_sha256": expected_training_plan_sha256, "hs_tasnet.training_updates": str(expected_step),
        "hs_tasnet.exporter_sha256": source_bindings["exporter"]["sha256"],
        "hs_tasnet.ola_source_sha256": source_bindings["ola_model"]["sha256"],
        "hs_tasnet.export_helpers_sha256": source_bindings["export_helpers"]["sha256"],
        "hs_tasnet.fft_implementation": "ONNX DFT1024/DFT512; inverse DFT512 with explicit Hermitian endpoints/interior",
        "hs_tasnet.external_data": "false",
    }
    receipt = {"schema": "ola512-cpu-onnx-verification-v1", "status": "in_progress", "identity": identity,
               "output": str(output), "source_bindings": source_bindings, "metadata": metadata,
               "native_host_timing_qualified": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=output.stem + ".", suffix=".onnx", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with torch.inference_mode():
            examples = wrapper(*args)
            torch.onnx.export(wrapper, args, str(temporary), export_params=True, opset_version=17,
                              do_constant_folding=True, input_names=list(INPUT_NAMES),
                              output_names=list(OUTPUT_NAMES), dynamo=False, external_data=False)
        graph = onnx.load(str(temporary), load_external_data=False)
        require(tuple(value.name for value in graph.graph.output) == OUTPUT_NAMES, "Export changed output names")
        require(not any(value.data_location == onnx.TensorProto.EXTERNAL for value in graph.graph.initializer),
                "Graph must be self-contained")
        for value, example, expected_shape in zip(graph.graph.output, examples, OUTPUT_SHAPES, strict=True):
            require(tuple(example.shape) == expected_shape, "Export-copy output shape differs")
            shape = value.type.tensor_type.shape
            shape.ClearField("dim")
            for size in expected_shape:
                shape.dim.add().dim_value = size
        onnx.helper.set_model_props(graph, metadata)
        onnx.save(graph, str(temporary))
        onnx.checker.check_model(str(temporary), full_check=True)
        receipt["onnx_sha256"] = sha256(temporary)
        receipt["onnx_bytes"] = temporary.stat().st_size
        receipt["verification"] = verify_onnx(model, wrapper, temporary, hops=verify_hops,
                                               audio_paths=verify_audio, threads=threads)
        require(sha256(checkpoint) == expected_sha256, "Checkpoint bytes changed during export/verification")
        require(state_sha256(model) == expected_model_state_sha256
                and state_sha256(wrapper.model) == expected_model_state_sha256,
                "Native or export-copy tensor bytes changed")
        require(all(sha256(value["path"]) == value["sha256"] for value in source_bindings.values()),
                "A bound source changed during export/verification")
        require(receipt["verification"]["passed"], "Three-way CPU numerical verification failed")
        receipt["status"] = "passed_cpu_numerical_verification_only"
        receipt["elapsed_seconds"] = time.monotonic() - started
        # Same-directory hard linking creates the final name without overwriting
        # any file that appeared since preflight; the temporary name is removed.
        os.link(temporary, output)
        _write_new_json(receipt_path, receipt)
        with hash_path.open("x") as stream:
            stream.write(f"{receipt['onnx_sha256']}  {output.name}\n")
        return receipt
    except Exception as error:
        receipt["status"] = "failed_not_qualified"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        receipt["elapsed_seconds"] = time.monotonic() - started
        if temporary.is_file() and temporary.stat().st_size:
            os.link(temporary, failed_graph_path)
            receipt["failed_graph"] = {"path": str(failed_graph_path), "sha256": sha256(temporary)}
        _write_new_json(failed_receipt_path, receipt)
        raise
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-model-state-sha256", required=True)
    parser.add_argument("--expected-training-plan-sha256", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--verify-hops", type=int, default=48,
                        help="Continuous signal/music hops before partial tail and flush; 4096+ supports a long run")
    parser.add_argument("--verify-audio", type=Path, action="append", default=[])
    args = parser.parse_args()
    receipt = export_model(args.checkpoint, args.output, expected_sha256=args.expected_sha256,
                           expected_model_state_sha256=args.expected_model_state_sha256,
                           expected_training_plan_sha256=args.expected_training_plan_sha256,
                           expected_step=args.expected_step, threads=args.threads,
                           verify_hops=args.verify_hops, verify_audio=args.verify_audio)
    print(json.dumps({key: receipt[key] for key in ("status", "output", "onnx_sha256", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
