"""Compare native, export-copy and ORT trajectories for either selected model.

The physical-input alignment oracle and original tolerances are retained from
latency58_asymmetric_onnx. State geometry comes from the native state family,
including both attention caches when present. Every backend advances its own
state; reset replays the whole trajectory.
"""
from pathlib import Path
import hashlib
import numpy as np
import torch

from research.direct.latency58 import HOP, PUBLIC_FUSION_SCALE, SOURCE_ORDER
from research.direct.latency58_checkpoint import require, sha as sha256
from research.direct.latency58_best_onnx import interface, TOLERANCES
from research.direct.latency58_asymmetric_onnx import verification_cases

def _validate_values(values, context, contract):
    OUTPUT_NAMES, OUTPUT_SHAPES = contract["output_names"], contract["output_shapes"]
    require(len(values) == len(OUTPUT_NAMES), f"Wrong output count: {context}")
    for name, value, shape in zip(OUTPUT_NAMES, values, OUTPUT_SHAPES, strict=True):
        require(value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all(),
                f"Nonfinite or incorrect output ABI: {context}/{name}")


def _run_case(model, wrapper, session, case):
    contract = interface(model)
    INPUT_NAMES, OUTPUT_NAMES = contract["input_names"], contract["output_names"]
    STATE_NAMES = contract["state_names"]
    audio = np.ascontiguousarray(case["audio"], dtype=np.float32)
    require(audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 0
            and np.isfinite(audio).all(), "Invalid verification fixture audio")
    initial = case["initial_states"]
    native_state = type(model.initial_state(1))(*(torch.from_numpy(value.copy()) for value in initial))
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
            actual = session.run(list(OUTPUT_NAMES), dict(zip(INPUT_NAMES, [chunk, *ort_state], strict=True)))
            ort_state = actual[1:]
            outputs = {"native": [value.detach().numpy() for value in (native_output, *native_state)],
                       "export_copy": [value.detach().numpy() for value in copied], "ort": actual}
            expected_history = np.concatenate((expected_history[..., HOP:], chunk), axis=-1)
            expected_mixture = aligned[:, index * HOP:(index + 1) * HOP][None]
            for backend, values in outputs.items():
                _validate_values(values, f"{case['name']}/{index}/{backend}", contract)
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
        row["trajectory_rms_error_by_stem"] = dict(zip(SOURCE_ORDER, per_stem_rms.tolist(), strict=True))
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


def cases(model, *, hops, audio_paths):
    contract = interface(model)
    for case in verification_cases(hops, audio_paths):
        extra = contract["state_shapes"][4:]
        nonzero = case["name"].startswith("nonzero_")
        rng = np.random.default_rng(202609150)
        additions = [rng.normal(0, .03, shape).astype(np.float32) if nonzero else np.zeros(shape, np.float32)
                     for shape in extra]
        yield {**case, "initial_states": [*case["initial_states"], *additions]}


def verify_memory(model, wrapper, data, *, hops=48, audio_paths=(), optimization="all"):
    import onnxruntime as ort
    require(isinstance(data, bytes) and data, "Use immutable in-memory graph bytes")
    contract = interface(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = {"all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                                       "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL}[optimization]
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    session = ort.InferenceSession(data, sess_options=options, providers=["CPUExecutionProvider"])
    require(session.get_providers() == ["CPUExecutionProvider"], "Require CPU provider")
    for direction, actual in (("input", session.get_inputs()), ("output", session.get_outputs())):
        require(tuple(value.name for value in actual) == contract[direction + "_names"]
                and tuple(tuple(value.shape) for value in actual) == contract[direction + "_shapes"]
                and all(value.type == "tensor(float)" for value in actual), "Graph ABI changed")
    rows = []
    for case in cases(model, hops=hops, audio_paths=audio_paths):
        first = _run_case(model, wrapper, session, case)
        replay = _run_case(model, wrapper, session, case)
        if case.get("source"):
            require(sha256(case["source"]["path"]) == case["source"]["sha256"], "Fixture music changed")
        exact = first["all_output_and_state_trajectory_sha256"] == replay["all_output_and_state_trajectory_sha256"]
        first.update({"reset_replay_all_outputs_and_states_bit_exact": exact,
                      "reset_replay_passed": replay["passed"], "passed": first["passed"] and replay["passed"] and exact})
        rows.append(first)
        print(__import__("json").dumps({"case": first["input"], "optimization": optimization,
              "passed": first["passed"], "callbacks": first["callbacks"],
              "native_vs_ort": first["comparisons"]["native_vs_ort"]}), flush=True)
    return {"passed": all(row["passed"] for row in rows), "device": "cpu", "optimization": optimization,
            "torch_version": torch.__version__, "onnxruntime_version": ort.__version__,
            "intra_op_threads": 1, "inter_op_threads": 1, "interface": contract,
            "graph_sha256": hashlib.sha256(data).hexdigest(),
            "independent_recurrent_trajectories": ["native", "export_copy", "ort"],
            "gru_state_comparison": "Public values decoded by multiplying by 2**18",
            "closure_reference": "Independent incoming physical history/input timeline, delayed 128 samples",
            "tolerances": TOLERANCES, "cases": rows, "native_host_timing_qualified": False}
