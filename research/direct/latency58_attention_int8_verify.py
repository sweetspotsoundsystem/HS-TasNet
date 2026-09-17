"""Strict carried-state parity for independently reconstructed integer inference."""
import hashlib

import numpy as np
import torch

from research.direct.latency58_best_onnx import TOLERANCES
from research.direct.run_latency58_quality import require


def session_for(data, contract, optimization="all"):
    import onnxruntime as ort
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
        require(tuple(v.name for v in actual) == tuple(contract[direction + "_names"])
                and tuple(tuple(v.shape) for v in actual) == tuple(tuple(s) for s in contract[direction + "_shapes"])
                and all(v.type == "tensor(float)" for v in actual), "Public graph ABI differs")
    return session


def run_case(reference, session, case, contract):
    audio = np.ascontiguousarray(case["audio"], dtype=np.float32)
    count = audio.shape[-1]
    require(audio.shape == (2, count) and count > 0 and np.isfinite(audio).all(), "Invalid fixture")
    padding = (-count) % 128
    padded = np.pad(audio, ((0, 0), (0, padding + 128)))
    states = [v.copy() for v in case["initial_states"]]
    own = [torch.from_numpy(v.copy()) for v in states]
    require(tuple(v.shape for v in states) == tuple(contract["state_shapes"]), "Initial state shape differs")
    history, previous = states[0].copy(), states[0][0, :, -128:].copy()
    maxima = np.zeros(len(contract["output_names"]))
    per_stem_maximum, per_stem_callback_rms = np.zeros(4), np.zeros(4)
    square_sum = np.zeros(4)
    closure = {"reference": 0., "ort": 0.}
    digests = {key: hashlib.sha256() for key in closure}
    last_sample = {}
    recovered = 0
    with torch.inference_mode():
        for hop in range(padded.shape[-1] // 128):
            chunk = np.ascontiguousarray(padded[None, :, hop * 128:(hop + 1) * 128])
            expected = [v.numpy() for v in reference(torch.from_numpy(chunk), *own)]
            actual = session.run(list(contract["output_names"]),
                dict(zip(contract["input_names"], [chunk, *states], strict=True)))
            history = np.concatenate((history[..., 128:], chunk), -1)
            for label, values in (("reference", expected), ("ort", actual)):
                require(len(values) == len(contract["output_names"])
                        and all(v.dtype == np.float32 and v.shape == tuple(shape) and np.isfinite(v).all()
                                for v, shape in zip(values, contract["output_shapes"], strict=True)),
                        "Nonfinite value, wrong shape or wrong public dtype")
                require(np.array_equal(values[1], history), "Physical history shift differs")
                for value in values:
                    digests[label].update(np.ascontiguousarray(value).tobytes())
                mixture = values[0][0].sum(0)
                closure[label] = max(closure[label], float(np.abs(mixture.astype(np.float64) - previous).max()))
                if hop == padded.shape[-1] // 128 - 1:
                    last_sample[label] = float(np.abs(mixture[:, (count - 1) % 128].astype(np.float64)
                                                     - audio[:, -1]).max())
            for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
                error = np.abs(left.astype(np.float64) - right).max() * (2**18 if index == 2 else 1)
                maxima[index] = max(maxima[index], error)
            difference = actual[0].astype(np.float64) - expected[0]
            per_stem_maximum = np.maximum(per_stem_maximum, np.abs(difference).max(axis=(0, 2, 3)))
            per_stem_callback_rms = np.maximum(per_stem_callback_rms,
                                               np.sqrt(np.mean(difference ** 2, axis=(0, 2, 3))))
            square_sum += np.sum(difference ** 2, axis=(0, 2, 3))
            if hop > 0:
                recovered += min(128, max(0, count - (hop - 1) * 128))
            states, own = actual[1:], [torch.from_numpy(v) for v in expected[1:]]
            previous = chunk[0].copy()
    require(recovered == count, "Partial-hop and single-flush sample accounting differs")
    calls = padded.shape[-1] // 128
    return {"case": case["name"], "physical_samples": count, "partial_hop_padding": padding,
            "decoded_input_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
            "source": case.get("source"), "calls": calls, "flush_hops": 1, "physical_samples_recovered": recovered,
            "every_callback_shapes_dtypes_finite_and_exact_history_verified": True,
            "maximum_errors_in_physical_units": dict(zip(contract["output_names"], maxima.tolist(), strict=True)),
            "maximum_audio_error_by_stem": dict(zip(("drums", "bass", "vocals", "other"), per_stem_maximum.tolist())),
            "maximum_stem_callback_rms_by_stem": dict(zip(("drums", "bass", "vocals", "other"), per_stem_callback_rms.tolist())),
            "trajectory_rms_error_by_stem": dict(zip(("drums", "bass", "vocals", "other"),
                                                       np.sqrt(square_sum / (calls * 256)).tolist())),
            "maximum_closure": closure, "last_real_sample_reconstruction_error": last_sample,
            "all_output_and_state_trajectory_sha256": {key: value.hexdigest() for key, value in digests.items()},
            "passed": bool(maxima[0] <= TOLERANCES["waveform_max_abs"]
                           and maxima[1:].max() <= TOLERANCES["state_max_abs_decoded_units"]
                           and per_stem_callback_rms.max() <= TOLERANCES["stem_callback_rms"]
                           and max(closure.values()) <= TOLERANCES["reconstruction_max_abs"])}
