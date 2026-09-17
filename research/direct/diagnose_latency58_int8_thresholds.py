"""Localize integer trajectory differences using shared-state layer controls.

This diagnostic cannot qualify recurrent parity: shared states deliberately
remove accumulated drift. Expected integer arithmetic is always PyTorch.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, write, sha, require


def main():
    import numpy as np
    import onnx
    from onnx import TensorProto, helper
    import onnxruntime as ort
    import soundfile as sf
    import torch
    from research.direct.latency58_asymmetric import Latency58AsymmetricModel
    from research.direct.latency58_residual_model import load_checkpoint, BASE_STATE
    from research.direct.latency58_asymmetric_onnx import INPUT_NAMES, OUTPUT_NAMES, verification_cases, _initial_states
    from research.direct.latency58_int8_reference import make_reference, IntegerLinear
    from research.direct.check_latency58_fused_gru import session_for
    from research.direct.train_latency58 import state_sha256, verify_inputs
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
            and ort.__version__ == "1.26.0", "Require CPU1 and shipping runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    source_path = PHASE / "full-magnitude-fast16-001/plan.json"
    source = read(source_path)
    require_space(source, 375_000_000)
    quantized = read(PHASE / "m4-int8-screen-002/result.json")["quantized"]
    require(sha(quantized["path"]) == quantized["sha256"], "Quantized graph changed")
    parent_binding = read(PHASE / "full-magnitude-001/plan.json")["parent_checkpoint"]
    parent, _ = load_checkpoint(parent_binding["path"], parent_binding["sha256"])
    native = Latency58AsymmetricModel()
    native.load_state_dict({k: v for k, v in parent.state_dict().items() if k != "fixed_residual_share"}, strict=True)
    native.eval().requires_grad_(False)
    require(state_sha256(native.state_dict()) == BASE_STATE, "Original FP32 source differs")
    graph = onnx.load(quantized["path"], load_external_data=False)
    reference, _ = make_reference(native, graph)
    instrumented = copy.deepcopy(graph)
    nodes = list(graph.graph.node)
    producers = {value: node for node in nodes for value in node.output}
    consumers = {}
    for node in nodes:
        for value in node.input:
            consumers.setdefault(value, []).append(node)
    modules = {m.proof["initializer"]: m for m in reference.modules() if isinstance(m, IntegerLinear)}
    projections = {}
    observed_names = []
    for node in nodes:
        if node.op_type != "MatMulInteger":
            continue
        prefix = node.input[1].removesuffix("_quantized")
        require(prefix in modules, "Unknown integer projection")
        quant = producers[node.input[0]]
        cast, = [n for n in consumers[node.output[0]] if n.op_type == "Cast"]
        scaled, = [n for n in consumers[cast.output[0]] if n.op_type == "Mul"]
        addition, = [n for n in consumers[scaled.output[0]] if n.op_type == "Add"]
        names = [quant.input[0], *quant.output, node.output[0], addition.output[0]]
        types = [TensorProto.FLOAT, TensorProto.UINT8, TensorProto.FLOAT, TensorProto.UINT8,
                 TensorProto.INT32, TensorProto.FLOAT]
        for name, dtype in zip(names, types, strict=True):
            if name not in observed_names:
                instrumented.graph.output.append(helper.make_tensor_value_info(name, dtype, None))
                observed_names.append(name)
        projections[prefix] = names
    require(len(projections) == 9, "Expected all nine layers")
    ordinary = session_for(Path(quantized["path"]).read_bytes())
    observed = ort.InferenceSession(instrumented.SerializeToString(),
        sess_options=ordinary.get_session_options(), providers=["CPUExecutionProvider"])
    require(observed.get_providers() == ["CPUExecutionProvider"]
            and [v.name for v in observed.get_inputs()] == list(INPUT_NAMES)
            and [v.name for v in observed.get_outputs()] == [*OUTPUT_NAMES, *observed_names],
            "Unexpected instrumentation interface")
    del graph, instrumented

    def quantize(x):
        low, high = x.min().clamp_max(0), x.max().clamp_min(0)
        scale = (high - low) / 255.
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        zero = torch.round(-low / scale).clamp(0, 255).to(torch.int32)
        return (torch.round(x / scale) + zero).clamp(0, 255).to(torch.int32), scale, zero

    captured = {}
    def hook_for(prefix):
        def capture(module, inputs, output):
            captured[prefix] = (inputs[0].detach().clone(), output.detach().clone())
        return capture
    handles = [module.register_forward_hook(hook_for(prefix)) for prefix, module in modules.items()]
    music_path = Path("/home/axel/HS-TasNet/data/musdb18hq/train/Actions - One Minute Smile/mixture.wav")
    music, rate = sf.read(music_path, start=10 * 44100, frames=1024 * 128 + 37,
                          always_2d=True, dtype="float32")
    require(rate == 44100 and music.shape == (1024 * 128 + 37, 2), "Music excerpt differs")
    cases = [next(case for case in verification_cases(1024, []) if case["name"] == "dc_nyquist_offgrid_partial"),
             {"name": "recorded_training_music_offset_10_seconds", "audio": np.ascontiguousarray(music.T),
              "initial_states": _initial_states()}]
    out = PHASE / "m4-int8-threshold-localization-001"
    require(not out.exists(), "Preserve diagnostics")
    out.mkdir()
    paths = [source_path, Path(__file__).resolve(), ROOT / "research/direct/latency58_int8_reference.py", music_path,
             Path(quantized["path"]), ROOT / "research/direct/latency58_asymmetric_onnx.py"]
    bindings = {**source["source_bindings"], **{str(p): sha(p) for p in paths}}
    write(out / "plan.json", {"source_bindings": bindings, "graph": quantized,
          "shared_state_controls_only": True, "runtime_oracle_used": False,
          "instrumented_graph_must_be_compared_with_original_on_same_inputs": True,
          "cases": [case["name"] for case in cases], "audio_written": False, "ort_version": ort.__version__})
    rows, began = [], time.monotonic()
    with torch.inference_mode():
        for case in cases:
            count = case["audio"].shape[-1]
            padded = np.pad(case["audio"], ((0, 0), (0, (-count) % 128 + 128)))
            states = [value.copy() for value in case["initial_states"]]
            layers = {prefix: {"calls": 0, "input_max_abs": 0., "input_different_calls": 0,
                       "upstream_quantized_input_different_calls": 0, "same_input_quantization_different_calls": 0,
                       "same_input_integer_dot_different_calls": 0, "same_input_projection_max_abs": 0.,
                       "trajectory_projection_max_abs": 0., "first_upstream_quantized_difference": None}
                      for prefix in projections}
            instrumentation_error, reference_error = np.zeros(5), np.zeros(5)
            for hop, offset in enumerate(range(0, padded.shape[-1], 128), start=1):
                chunk = np.ascontiguousarray(padded[None, :, offset:offset + 128])
                feeds = dict(zip(INPUT_NAMES, [chunk, *states], strict=True))
                actual = ordinary.run(list(OUTPUT_NAMES), feeds)
                tapped_values = observed.run([*OUTPUT_NAMES, *observed_names], feeds)
                tapped = dict(zip(observed_names, tapped_values[5:], strict=True))
                predicted = [value.numpy() for value in reference(torch.from_numpy(chunk),
                             *(torch.from_numpy(value) for value in states))]
                require(set(captured) == set(modules), "Missing reference layer")
                for index in range(5):
                    factor = 2**18 if index == 2 else 1
                    instrumentation_error[index] = max(instrumentation_error[index],
                        float(np.abs(actual[index] - tapped_values[index]).max()) * factor)
                    reference_error[index] = max(reference_error[index],
                        float(np.abs(actual[index] - predicted[index]).max()) * factor)
                for prefix, names in projections.items():
                    incoming, q, scale, zero, integer, projection = [tapped[name] for name in names]
                    own_input, own_output = captured[prefix]
                    same_input = torch.from_numpy(incoming.copy()).reshape(own_input.shape)
                    own_q, own_scale, own_zero = quantize(own_input)
                    same_q, same_scale, same_zero = quantize(same_input)
                    same_integer = (same_q.reshape(1, -1) - same_zero) @ modules[prefix].centered_weight
                    same_output = modules[prefix].forward(same_input)
                    row = layers[prefix]
                    row["calls"] += 1
                    input_error = float(np.abs(incoming.reshape(-1) - own_input.numpy().reshape(-1)).max())
                    row["input_max_abs"] = max(row["input_max_abs"], input_error)
                    row["input_different_calls"] += int(input_error != 0)
                    upstream = not (np.array_equal(own_q.numpy().reshape(-1), q.reshape(-1))
                                    and float(own_zero) == float(zero))
                    row["upstream_quantized_input_different_calls"] += int(upstream)
                    same_q_equal = (np.array_equal(same_q.numpy().reshape(-1), q.reshape(-1))
                                    and float(same_scale) == float(scale) and float(same_zero) == float(zero))
                    row["same_input_quantization_different_calls"] += int(not same_q_equal)
                    row["same_input_integer_dot_different_calls"] += int(not np.array_equal(
                        same_integer.numpy().reshape(-1), integer.reshape(-1)))
                    row["same_input_projection_max_abs"] = max(row["same_input_projection_max_abs"],
                        float(np.abs(same_output.numpy().reshape(-1) - projection.reshape(-1)).max()))
                    row["trajectory_projection_max_abs"] = max(row["trajectory_projection_max_abs"],
                        float(np.abs(own_output.numpy().reshape(-1) - projection.reshape(-1)).max()))
                    if upstream and row["first_upstream_quantized_difference"] is None:
                        row["first_upstream_quantized_difference"] = {"hop": hop, "input_max_abs": input_error,
                            "different_integer_entries": int(np.count_nonzero(own_q.numpy().reshape(-1) != q.reshape(-1))),
                            "reference_scale": float(own_scale), "runtime_scale": float(scale),
                            "reference_zero": int(own_zero), "runtime_zero": int(zero)}
                require(all(np.isfinite(v).all() for v in [*actual, *predicted, *tapped_values]), "Nonfinite values")
                states = actual[1:]
                if hop % 256 == 0:
                    print(json.dumps({"case": case["name"], "hops": hop,
                                      "shared_state_waveform_max_abs": reference_error[0]}), flush=True)
            result = {"case": case["name"], "calls": padded.shape[-1] // 128, "layers": layers,
                      "instrumentation_vs_original_physical_max_errors": dict(zip(OUTPUT_NAMES, instrumentation_error.tolist(), strict=True)),
                      "shared_state_reference_vs_original_physical_max_errors": dict(zip(OUTPUT_NAMES, reference_error.tolist(), strict=True))}
            rows.append(result)
            write(out / f"case-{len(rows)}.json", result)
    for handle in handles:
        handle.remove()
    verify_inputs({"source_bindings": bindings})
    require(not torch.cuda.is_initialized() and state_sha256(native.state_dict()) == BASE_STATE,
            "CPU scope or source changed")
    write(out / "result.json", {"status": "diagnostic_complete", "cases": rows,
          "source_bindings_unchanged": True, "native_host_qualified": False,
          "recurrent_parity_qualified": False, "tolerances_changed": False, "plugin_modified": False,
          "elapsed_seconds": time.monotonic() - began, "counted_bytes_after": require_space(source, 370_000_000)})
    print(json.dumps({"status": "diagnostic_complete", "elapsed_seconds": time.monotonic() - began}), flush=True)


if __name__ == "__main__":
    main()
