"""Export and numerically verify the trainable hop-128 model on CPU."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .streaming_model import (
    FEATURE_HISTORY, HOP, PUBLIC_FUSION_SCALE, SOURCE_ORDER, SYNTHESIS_SAMPLES,
    StreamingState, StreamingHSTasNet, require, VERSION,
)
from .streaming_checkpoint import state_sha256, file_sha256 as sha256

STATE_NAMES = tuple(StreamingState._fields)
STATE_SHAPES = ((1, 2, FEATURE_HISTORY), (2, 1, 1000), (1, 4, 2, HOP), (1, 4, 2, HOP))
INPUT_NAMES = ("audio_chunk", *STATE_NAMES)
OUTPUT_NAMES = ("separated_chunk", *("next_" + name for name in STATE_NAMES))
INPUT_SHAPES = ((1, 2, HOP), *STATE_SHAPES)
OUTPUT_SHAPES = ((1, 4, 2, HOP), *STATE_SHAPES)
TOLERANCES = {"waveform_max_abs": 1e-4, "stem_callback_rms": 1e-5,
              "state_max_abs_decoded_units": 5e-4, "reconstruction_max_abs": 1e-6}


def model_state_sha256(model):
    return state_sha256(model.state_dict())


def export_streaming_model(model, output, *, verify_hops=48):
    """Export a self-contained graph and check native/export/ORT recurrence.

    The original model, parameters and RNG remain unchanged. Both the graph
    and its JSON verification report are written only after parity succeeds.
    Existing files are preserved. This verifies numerics, not playback timing.
    """
    import onnx

    require(isinstance(model, StreamingHSTasNet), "Expected StreamingHSTasNet")
    require(all(t.device.type == "cpu" and t.dtype == torch.float32 for t in model.state_dict().values()),
            "Export requires a CPU FP32 model")
    output = Path(output)
    report_path = output.with_suffix(".verification.json")
    if any(path.exists() or path.is_symlink() for path in (output, report_path)):
        raise FileExistsError("Export destinations already exist")
    require(output.suffix == ".onnx", "Use an .onnx output filename")
    output.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = model_state_sha256(model)
    rng = torch.get_rng_state().clone()
    wrapper = make_export_copy(model)
    native = copy.deepcopy(model).eval()
    descriptor, name = tempfile.mkstemp(prefix=".streaming-export-", suffix=".onnx", dir=output.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        inputs = (torch.zeros(INPUT_SHAPES[0]), *model.initial_state(1))
        with torch.inference_mode():
            torch.onnx.export(wrapper, inputs, str(temporary), export_params=True, opset_version=17,
                              do_constant_folding=True, input_names=list(INPUT_NAMES), output_names=list(OUTPUT_NAMES),
                              dynamo=False, external_data=False)
        graph = onnx.load(str(temporary), load_external_data=False)
        require(tuple(value.name for value in graph.graph.output) == OUTPUT_NAMES
                and not any(value.data_location == onnx.TensorProto.EXTERNAL for value in graph.graph.initializer),
                "Exported interface or self-contained storage differs")
        for value, shape in zip(graph.graph.output, OUTPUT_SHAPES):
            dims = value.type.tensor_type.shape
            dims.ClearField("dim")
            for size in shape:
                dims.dim.add().dim_value = size
        onnx.helper.set_model_props(graph, {"architecture": VERSION, "sample_rate": "44100",
            "source_order": json.dumps(SOURCE_ORDER), "hop_samples": "128", "graph_alignment_samples": "128",
            "model_state_sha256": fingerprint})
        onnx.save(graph, str(temporary))
        onnx.checker.check_model(str(temporary), full_check=True)
        verification = verify_onnx(native, wrapper, temporary, hops=verify_hops, threads=1)
        require(verification["passed"] and model_state_sha256(model) == model_state_sha256(wrapper.model) == fingerprint
                and torch.equal(rng, torch.get_rng_state()), "Export parity, parameter or RNG check failed")
        report = {"status": "pass", "model_state_sha256": fingerprint, "onnx_sha256": sha256(temporary),
                  "onnx_bytes": temporary.stat().st_size, "verification": verification,
                  "playback_timing_tested": False}
        os.link(temporary, output)
        with report_path.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return report
    finally:
        temporary.unlink(missing_ok=True)


class RMSNormForONNX(nn.Module):
    """Small ONNX-friendly equivalent of ``torch.nn.RMSNorm``."""

    def __init__(self, original: nn.RMSNorm):
        super().__init__()
        if not bool(original.elementwise_affine) or original.weight is None:
            raise ValueError("ONNX RMSNorm replacement requires affine weights")
        eps = original.eps
        if eps is None:
            eps = torch.finfo(original.weight.dtype).eps
        self.eps = float(eps)
        self.register_buffer("weight", original.weight.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class OneFrameGRUForONNX(nn.Module):
    """Exact, exportable GRU recurrence for the one-frame streaming graph."""

    def __init__(self, original: nn.GRU):
        super().__init__()
        if not original.batch_first:
            raise ValueError("Streaming export requires a batch-first GRU")
        if original.bidirectional:
            raise ValueError("Streaming export requires a unidirectional GRU")
        if not original.bias:
            raise ValueError("Streaming export requires GRU bias tensors")
        if float(original.dropout) != 0.0:
            raise ValueError("Streaming export does not support GRU dropout")

        self.input_size = int(original.input_size)
        self.hidden_size = int(original.hidden_size)
        self.num_layers = int(original.num_layers)
        for layer in range(self.num_layers):
            for prefix in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                value = getattr(original, f"{prefix}_l{layer}")
                self.register_buffer(f"{prefix}_l{layer}", value.detach().clone())

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_input = x[:, 0]
        next_hidden: list[torch.Tensor] = []
        for layer in range(self.num_layers):
            previous = hidden[layer]
            input_gates = F.linear(
                layer_input,
                getattr(self, f"weight_ih_l{layer}"),
                getattr(self, f"bias_ih_l{layer}"),
            )
            hidden_gates = F.linear(
                previous,
                getattr(self, f"weight_hh_l{layer}"),
                getattr(self, f"bias_hh_l{layer}"),
            )
            input_reset, input_update, input_new = input_gates.chunk(3, dim=-1)
            hidden_reset, hidden_update, hidden_new = hidden_gates.chunk(3, dim=-1)
            reset = torch.sigmoid(input_reset + hidden_reset)
            update = torch.sigmoid(input_update + hidden_update)
            candidate = torch.tanh(input_new + reset * hidden_new)
            current = candidate + update * (previous - candidate)
            next_hidden.append(current)
            layer_input = current

        stacked_hidden = torch.stack(next_hidden, dim=0)
        return layer_input.unsqueeze(1), stacked_hidden


def replace_rmsnorm_layers(module: nn.Module) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.RMSNorm):
            setattr(module, name, RMSNormForONNX(child))
            continue
        replace_rmsnorm_layers(child)
    return module

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


def make_export_copy(model):
    """Build only a separate export copy; the complete deployed residual is formed once."""
    class IRFFT1024(torch.autograd.Function):
        @staticmethod
        def forward(ctx, spectrum):
            value = torch.complex(spectrum[..., 0], spectrum[..., 1])
            return torch.fft.irfft(value, n=1024, dim=1)

        @staticmethod
        def symbolic(graph, spectrum):
            endpoints = torch.ones(513, 2, dtype=torch.float32)
            endpoints[0, 1] = 0.0
            endpoints[-1, 1] = 0.0
            values = graph.op('Mul', spectrum, graph.op('Constant', value_t=endpoints))
            indices = graph.op('Constant', value_t=torch.arange(511, 0, -1, dtype=torch.int64))
            reflected = graph.op('Gather', values, indices, axis_i=1)
            sign = graph.op('Constant', value_t=torch.tensor([1.0, -1.0], dtype=torch.float32))
            reflected = graph.op('Mul', reflected, sign)
            full = graph.op('Concat', values, reflected, axis_i=1)
            length = graph.op('Constant', value_t=torch.tensor(1024, dtype=torch.int64))
            inverse = graph.op('DFT', full, length, axis_i=1, inverse_i=1, onesided_i=0)
            real_index = graph.op('Constant', value_t=torch.tensor(0, dtype=torch.int64))
            result = graph.op('Gather', inverse, real_index, axis_i=2)
            return result.setType(spectrum.type().with_sizes([spectrum.type().sizes()[0], 1024]))

    class Latency58StreamingWrapper(torch.nn.Module):
        def __init__(self, copied):
            super().__init__()
            self.model = copied

        def forward(self, audio_chunk, audio_history, fusion_hidden, spectral_numerator_tail, waveform_tail):
            copied = self.model
            joined = torch.cat((audio_history, audio_chunk), dim=-1)
            feature_ri = _RFFT1024.apply((joined * copied.analysis_window).reshape(2, 1024))
            feature_ri = feature_ri.reshape(1, 2, 1, 513, 2)
            packed = feature_ri.permute(0, 2, 1, 3, 4).reshape(1, 1, 2052)
            spec = copied.spec_encode(packed)
            to_relu, to_sigmoid = copied.conv_encode(joined).chunk(2, dim=1)
            basis = to_relu.relu() * to_sigmoid.sigmoid()
            waveform = copied.basis_to_embed(basis).transpose(1, 2)
            fusion_input = torch.cat((spec, waveform), dim=-1)
            recurrent, next_physical_hidden = copied.fusion_branch(
                fusion_input, fusion_hidden / PUBLIC_FUSION_SCALE)
            fused = fusion_input + recurrent
            fused_spec, fused_waveform = fused.chunk(2, dim=-1)
            spec = fused_spec + spec
            waveform = fused_waveform + waveform

            logits = copied.to_spec_masks(copied.spec_norm(spec)).reshape(1, 1, 2, 513, 2, 4)
            masks = copied._residual_source_softmax(logits).permute(0, 2, 1, 3, 4, 5)
            masked_ri = feature_ri.unsqueeze(-1) * masks
            source_ri = masked_ri.permute(0, 5, 1, 2, 3, 4).contiguous().reshape(8, 513, 2)
            frames = IRFFT1024.apply(source_ri).reshape(1, 4, 2, 1024)[..., 1024 - SYNTHESIS_SAMPLES:1024]
            frames = frames * copied.synthesis.spectral_window
            spectral = (frames[..., :HOP] + spectral_numerator_tail) / copied.synthesis.spectral_denominator
            next_spectral_tail = frames[..., HOP:].clone()

            waveform_logits = copied.to_waveform_masks(copied.waveform_norm(waveform))
            waveform_logits = waveform_logits.reshape(1, 1, 4, 1500).transpose(-1, -2)
            waveform_masks = copied._residual_source_softmax(waveform_logits)
            source_basis = basis.transpose(1, 2).unsqueeze(-1) * waveform_masks
            source_basis = source_basis.permute(0, 3, 1, 2)
            decoded = torch.nn.functional.linear(source_basis, copied.waveform_decoder_weight.flatten(1).t(), bias=None)
            decoded = decoded.reshape(1, 4, 1, 2, SYNTHESIS_SAMPLES).permute(0, 1, 3, 2, 4)
            windowed = (decoded * copied.synthesis.window).reshape(1, 4, 2, SYNTHESIS_SAMPLES)
            waveform_audio = windowed[..., :HOP] + waveform_tail
            next_waveform_tail = windowed[..., HOP:].clone()

            raw = (spectral + waveform_audio) * copied.output_source_scales[None, :, None, None]
            delayed_mixture = audio_history[..., -HOP:]
            retained = raw[:, :3]
            deployed = torch.cat((retained, delayed_mixture.unsqueeze(1)
                                  - retained.sum(dim=1, keepdim=True)), dim=1)
            return (deployed, joined[..., -FEATURE_HISTORY:].clone(),
                    next_physical_hidden * PUBLIC_FUSION_SCALE,
                    next_spectral_tail, next_waveform_tail)

    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        copied = copy.deepcopy(model).cpu().eval()
        copied.fusion_branch = OneFrameGRUForONNX(copied.fusion_branch)
        replace_rmsnorm_layers(copied)
        wrapper = Latency58StreamingWrapper(copied).eval()
    require(model_state_sha256(copied) == model_state_sha256(model), 'Export copy changed checkpoint tensor bytes')
    return wrapper


def _initial_states(nonzero=False):
    states = [np.zeros(shape, dtype=np.float32) for shape in STATE_SHAPES]
    if nonzero:
        rng = np.random.default_rng(2026090617)
        scales = (0.04, 0.1 * PUBLIC_FUSION_SCALE, 0.001, 0.001)
        states = [rng.normal(0.0, scale, shape).astype(np.float32)
                  for shape, scale in zip(STATE_SHAPES, scales)]
    return states


def verification_cases(hops, audio_paths):
    """Physical input lengths exclude padding and the single final flush."""
    require(type(hops) is int and hops >= 8, "Verification requires at least eight signal hops")
    rng = np.random.default_rng(2026090618)
    count = hops * HOP + 37
    t = np.arange(count, dtype=np.float64) / 44100
    signal = np.stack([sum(amplitude * np.sin(2 * np.pi * frequency * t + phase)
                           for amplitude, frequency in ((.08, 55), (.04, 173.31), (.02, 997.3)))
                       for phase in (.23, .71)])
    signal += rng.normal(0.0, .005, signal.shape)
    signal = signal.astype(np.float32)
    yield {"name": "multitone_noise_then_silence_partial", "audio": np.pad(
        signal, ((0, 0), (0, 8 * HOP))), "initial_states": _initial_states()}
    yield {"name": "silence_from_reset", "audio": np.zeros((2, 8 * HOP + 17), np.float32),
           "initial_states": _initial_states()}
    endpoint = np.stack((.03 + .07 * (-1.0) ** np.arange(count) + .02 * np.sin(2 * np.pi * 55.3 * t),
                         -.04 - .05 * (-1.0) ** np.arange(count) + .03 * np.cos(2 * np.pi * 997.7 * t)))
    yield {"name": "dc_nyquist_offgrid_partial", "audio": endpoint.astype(np.float32),
           "initial_states": _initial_states()}
    impulses = np.zeros((2, 7 * HOP + 37), np.float32)
    for index, position in enumerate((0, 1, 127, 128, 255, 256, impulses.shape[-1] - 1)):
        impulses[:, position] = ((index + 1) / 16.0, -(index + 1) / 19.0)
    yield {"name": "boundary_and_last_sample_impulses", "audio": impulses,
           "initial_states": _initial_states()}
    yield {"name": "nonzero_history_hidden_and_both_tails", "audio": signal[:, :8 * HOP + 31],
           "initial_states": _initial_states(nonzero=True)}
    if audio_paths:
        import soundfile as sf
        for path in audio_paths:
            path = Path(path).resolve(strict=True)
            before = sha256(path)
            audio, rate = sf.read(path, frames=count, dtype="float32", always_2d=True)
            require(rate == 44100 and audio.shape[1] == 2 and audio.shape[0] > 0,
                    f"Verification music must be nonempty stereo 44.1 kHz: {path}")
            require(sha256(path) == before, "Music file changed while reading")
            yield {"name": f"music:{path}", "audio": np.ascontiguousarray(audio.T),
                   "initial_states": _initial_states(), "source": {"path": str(path), "sha256": before}}


def _validate_values(values, context):
    require(len(values) == len(OUTPUT_NAMES), f"Wrong output count: {context}")
    for name, value, shape in zip(OUTPUT_NAMES, values, OUTPUT_SHAPES):
        require(value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all(),
                f"Nonfinite or incorrect output ABI: {context}/{name}")


def _run_case(model, wrapper, session, case):
    audio = np.ascontiguousarray(case["audio"], dtype=np.float32)
    require(audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 0
            and np.isfinite(audio).all(), "Invalid verification fixture audio")
    initial = case["initial_states"]
    native_state = StreamingState(*(torch.from_numpy(value.copy()) for value in initial))
    copy_state = tuple(torch.from_numpy(value.copy()) for value in initial)
    ort_state = [value.copy() for value in initial]
    real_samples = audio.shape[-1]
    padding = (-real_samples) % HOP
    padded = np.pad(audio, ((0, 0), (0, padding)))
    received = np.pad(padded, ((0, 0), (0, HOP)))
    # This oracle is indexed from actual input plus the incoming physical history;
    # it does not use native/ORT next-state values to establish alignment.
    aligned = np.concatenate((initial[0][0, :, -HOP:], padded), axis=-1)
    expected_history = initial[0].copy()
    pairs = (("native", "export_copy"), ("native", "ort"), ("export_copy", "ort"))
    errors = {f"{left}_vs_{right}": {"waveform_max_abs": 0.0, "stem_callback_rms": 0.0,
               "waveform_max_abs_by_stem": {name: 0.0 for name in SOURCE_ORDER},
               "maximum_callback_rms_error_by_stem": {name: 0.0 for name in SOURCE_ORDER},
               "squared_error_sum_by_stem": np.zeros(4, dtype=np.float64),
               "state_max_abs_decoded_units": {name: 0.0 for name in STATE_NAMES}}
              for left, right in pairs}
    backends = ("native", "export_copy", "ort")
    digests = {name: hashlib.sha256() for name in backends}
    closure = {name: 0.0 for name in backends}
    flush_closure = {}
    last_sample_error = {}
    recovered_samples = 0
    data_hops = padded.shape[-1] // HOP
    with torch.inference_mode():
        for index in range(data_hops + 1):
            chunk = np.ascontiguousarray(received[:, index * HOP:(index + 1) * HOP][None])
            if index == data_hops:
                native_output, native_state = model.flush(native_state)
            else:
                native_output, native_state = model.forward_chunk(torch.from_numpy(chunk), native_state)
            copied = wrapper(torch.from_numpy(chunk), *copy_state)
            copy_state = copied[1:]
            actual = session.run(list(OUTPUT_NAMES), dict(zip(INPUT_NAMES, [chunk, *ort_state])))
            ort_state = actual[1:]
            outputs = {"native": [value.detach().numpy() for value in (native_output, *native_state)],
                       "export_copy": [value.detach().numpy() for value in copied], "ort": actual}
            expected_history = np.concatenate((expected_history[..., HOP:], chunk), axis=-1)
            expected_mixture = aligned[:, index * HOP:(index + 1) * HOP][None]
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
                    offset = (real_samples - 1) % HOP
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
                for stem_index, name in enumerate(SOURCE_ORDER):
                    row["waveform_max_abs_by_stem"][name] = max(
                        row["waveform_max_abs_by_stem"][name], float(per_stem_max[stem_index]))
                    row["maximum_callback_rms_error_by_stem"][name] = max(
                        row["maximum_callback_rms_error_by_stem"][name], float(per_stem_rms[stem_index]))
                for state_index, name in enumerate(STATE_NAMES):
                    factor = 1.0 / PUBLIC_FUSION_SCALE if name == "fusion_hidden" else 1.0
                    difference = (outputs[left][state_index + 1].astype(np.float64)
                                  - outputs[right][state_index + 1].astype(np.float64))
                    value = float(np.abs(difference).max()) * factor
                    row["state_max_abs_decoded_units"][name] = max(
                        row["state_max_abs_decoded_units"][name], value)
            if index > 0:
                recovered_samples += min(HOP, max(0, real_samples - (index - 1) * HOP))
    require(recovered_samples == real_samples, "Single-cut/flush physical sample accounting differs")
    for row in errors.values():
        per_stem_rms = np.sqrt(row.pop("squared_error_sum_by_stem") / ((data_hops + 1) * 2 * HOP))
        row["trajectory_rms_error_by_stem"] = dict(zip(SOURCE_ORDER, per_stem_rms.tolist()))
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
            "closure_reference": "Independent incoming physical history/input timeline, delayed 128 samples",
            "tolerances": TOLERANCES, "cases": rows,
            "native_host_timing_qualified": False}
