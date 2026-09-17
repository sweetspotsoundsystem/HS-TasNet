"""Screen one encoder weight approximation with native ORT and independent arithmetic."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile

import numpy as np

from research.direct.run_latency58_quality import ROOT, PHASE, read, write, sha, require
from research.direct.train_latency58 import verify_inputs, state_sha256


def fingerprint(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def graph(nodes, initializers):
    import onnx
    from onnx import helper as h, TensorProto as T
    value = h.make_model(h.make_graph(nodes, "encoder-kernel", [h.make_tensor_value_info("x", T.FLOAT, [1, 2048])],
                        [h.make_tensor_value_info("y", T.FLOAT, [1, 3000])], initializers),
                        opset_imports=[h.make_opsetid("", 17), h.make_opsetid("com.microsoft", 1)], ir_version=10)
    onnx.checker.check_model(value, full_check=True)
    return value


def build(native, saved_graph):
    from onnx import helper as h, numpy_helper as nh, TensorProto as T
    from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, DefaultWeightOnlyQuantizer
    from onnxruntime.quantization.quant_utils import quantize_data
    weight = np.ascontiguousarray(native.conv_encode.weight.detach().numpy().reshape(3000, 2048).T)
    bias = np.ascontiguousarray(native.conv_encode.bias.detach().numpy())
    stored = {v.name: nh.to_array(v) for v in saved_graph.graph.initializer}
    prefix = "model.conv_encode.weight"
    q8, s8, z8 = (stored[prefix + suffix] for suffix in ("_quantized", "_scale", "_zero_point"))
    require(q8.shape == (2048, 3000) and s8.shape == z8.shape == (3000,), "Saved encoder matrix shape changed")
    for column in range(3000):
        zero, scale, values = quantize_data(np.ascontiguousarray(weight[:, column]), T.UINT8,
                                          symmetric=False, reduce_range=False)
        require(np.array_equal(values, q8[:, column]) and np.array_equal(scale, s8[column])
                and np.array_equal(zero, z8[column]), "Saved U8 weights differ from authenticated original")
    config = DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, accuracy_level=1, bits=4)
    packed, scales, unused_zeros = DefaultWeightOnlyQuantizer(config).qbits_block_quant(weight)
    require(packed.shape == (3000, 64, 16) and scales.shape == (3000, 64)
            and np.isfinite(scales).all() and np.all(scales != 0), "Invalid four-bit encoder blocks")
    unpacked = np.stack((packed & 15, packed >> 4), axis=-1).reshape(3000, 64, 32)
    decoded = np.ascontiguousarray(((unpacked.astype(np.float32) - np.float32(8)) *
                                   scales[..., None]).reshape(3000, 2048).T)
    normalized_error = np.abs(weight.T.reshape(3000, 64, 32) - decoded.T.reshape(3000, 64, 32)) / np.abs(scales[..., None])
    # A symmetric signed block has an asymmetric -8..7 code range. Values at
    # the clipped endpoint may differ by up to one step; interior codes must
    # still be nearest to the source value.
    endpoint = (unpacked == 0) | (unpacked == 15)
    require(float(normalized_error.max()) < 1.001
            and float(normalized_error[~endpoint].max()) < .501,
            "Four-bit codes exceed the representable nearest-code error")
    arrays8 = {"w": q8, "s": s8, "z": z8, "bias": bias}
    nodes8 = [h.make_node("DynamicQuantizeLinear", ["x"], ["qx", "sx", "zx"], name="quantize"),
              h.make_node("MatMulInteger", ["qx", "w", "zx", "z"], ["integer"], name="matrix"),
              h.make_node("Cast", ["integer"], ["floating"], to=T.FLOAT, name="cast"),
              h.make_node("Mul", ["sx", "s"], ["scale"], name="scale"),
              h.make_node("Mul", ["floating", "scale"], ["linear"], name="dequantize"),
              h.make_node("Add", ["linear", "bias"], ["y"], name="bias")]
    arrays4 = {"w": packed, "s": scales, "bias": bias}
    nodes4 = [h.make_node("MatMulNBits", ["x", "w", "s"], ["linear"], name="matrix_int4",
                         domain="com.microsoft", K=2048, N=3000, block_size=32, bits=4, accuracy_level=1),
              h.make_node("Add", ["linear", "bias"], ["y"], name="bias")]
    models = {"u8u8": graph(nodes8, [nh.from_array(v, k) for k, v in arrays8.items()]),
              "int4_fp32_activation": graph(nodes4, [nh.from_array(v, k) for k, v in arrays4.items()])}
    proof = {"weight_shape": list(weight.shape), "source_matrix_sha256": fingerprint(weight),
             "saved_u8_weight_matches_original_quantizer": True, "packed_int4_sha256": fingerprint(packed),
             "int4_scales_sha256": fingerprint(scales), "decoded_int4_sha256": fingerprint(decoded),
             "block_size": 32, "accuracy_level": 1, "default_zero_point": 8,
             "maximum_weight_error_in_quantization_steps": float(normalized_error.max()),
             "weight_relative_rms_error": float(np.linalg.norm(weight - decoded) / np.linalg.norm(weight))}
    return models, (weight, bias, q8, s8, z8, decoded), proof


def references(inputs, weights):
    weight, bias, q8, s8, z8, decoded = weights
    integer_reference = []
    for row in inputs:
        minimum = np.minimum(np.float32(0), row.min())
        maximum = np.maximum(np.float32(0), row.max())
        scale = np.float32((maximum - minimum) / np.float32(255)) if maximum != minimum else np.float32(1)
        zero = np.clip(np.rint(-minimum / scale), 0, 255).astype(np.uint8)
        quantized = np.clip(np.rint(row / scale) + zero.astype(np.float32), 0, 255).astype(np.uint8)
        accum = (quantized.astype(np.int64) - zero.astype(np.int64)) @ (q8.astype(np.int64) - z8.astype(np.int64))
        integer_reference.append(accum.astype(np.float32) * (scale * s8) + bias)
    return {"u8u8": np.stack(integer_reference),
            "int4_fp32_activation": inputs.astype(np.float64) @ decoded.astype(np.float64) + bias.astype(np.float64),
            "original_fp32_weights": inputs.astype(np.float64) @ weight.astype(np.float64) + bias.astype(np.float64)}


def main():
    import onnx
    import torch
    from research.direct.latency58_quadrature_checkpoint import load_model
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use CPU1 with CUDA hidden")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    training_path = PHASE / "fusion-refinement-001/plan.json"
    training = read(training_path)
    counted = require_space(training, 380_000_000 + 40_000_000)
    native_path = PHASE / "m4-quadrature-candidates-quiet-native-001/plan.json"
    native_plan = read(native_path)
    profile_path = PHASE / "quadrature-native-operator-profile-001/analysis.json"
    profile = read(profile_path)
    require(profile["status"] == "pass" and profile["actual_root_exit_code"] == 0, "Profile must be reviewed first")
    verify_inputs(profile)
    parent_plan = PHASE / "quadrature-continuation-001/plan.json"
    parent_binding = read(parent_plan)["parent_checkpoint"]
    native, _ = load_model(parent_binding)
    parent_sha = state_sha256(native.state_dict())
    sdk = Path(native_plan["sdk"])
    cpp = Path(__file__).with_name("benchmark_latency58_encoder_kernel.cpp").resolve()
    quantizer = Path(importlib.import_module("onnxruntime.quantization.matmul_nbits_quantizer").__file__)
    paths = (Path(__file__).resolve(), cpp, quantizer, native_path, profile_path, parent_plan,
             Path(parent_binding["path"]), training_path)
    bindings = {**profile["source_bindings"], **{str(p): sha(p) for p in paths}}
    out = PHASE / "encoder-int4-kernel-screen-001"
    require(not out.exists(), "Preserve kernel screens")
    out.mkdir()
    write(out / "plan.json", {"source_bindings": bindings, "source_checkpoint": parent_binding,
          "source_model_state_sha256": parent_sha, "counted_bytes_before": counted,
          "scope": "Isolated encoder matrix; no full-model graph, training change, quality evaluation or M4 claim",
          "recipe": {"bits": 4, "block_size": 32, "symmetric": True, "accuracy_level": 1},
          "preregistered_reference_max_abs_tolerance": 1e-4,
          "timing_design": "Four fresh-process cycles in ABBA order; 64 warmup and 1024 measured calls per process",
          "official_operator_reference": "https://raw.githubusercontent.com/microsoft/onnxruntime/v1.26.0/docs/ContribOperators.md",
          "limitations": "Concurrent GPU training; isolated-kernel timings are a feasibility screen only."})
    saved = onnx.load(PHASE / "m4-quadrature-magint8-saved-001/model.onnx")
    models, weights, proof = build(native, saved)
    del saved
    rng = np.random.default_rng(202609151)
    inputs = (.03 * rng.standard_normal((16, 2048))).astype(np.float32)
    inputs[0] = 0
    inputs[1] = np.linspace(-.1, .1, 2048, dtype=np.float32)
    inputs[2] = 0
    inputs[2, 1023] = .5
    inputs[3] *= 1e-4
    expected = references(inputs, weights)
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    require(compiler is not None, "Native compiler unavailable")
    rows, numerical = [], {}
    # Temporary matrices total under 11 MB; the reservation includes 40 MB.
    with tempfile.TemporaryDirectory(prefix="latency58-encoder-kernel-", dir=PHASE) as temporary:
        temporary = Path(temporary)
        binary = temporary / "kernel"
        command = [compiler, "-std=c++20", "-O3", "-DNDEBUG", "-Wall", "-Wextra", "-Werror", str(cpp),
                   "-I" + str(sdk / "include"), "-L" + str(sdk / "lib"),
                   "-Wl,-rpath," + str(sdk / "lib"), "-lonnxruntime", "-o", str(binary)]
        write(out / "compiler-command.json", {"argv": command})
        subprocess.run(command, check=True, timeout=90)
        inputs.tofile(temporary / "inputs.bin")
        graph_proof = {}
        for label, model in models.items():
            path = temporary / (label + ".onnx")
            onnx.save(model, path)
            graph_proof[label] = {"sha256": sha(path), "bytes": path.stat().st_size}
        require_space(training, 380_000_000)
        for cycle, label in enumerate(("u8u8", "int4_fp32_activation", "int4_fp32_activation", "u8u8"), 1):
            output_path = temporary / (str(cycle) + "-output.bin")
            completed = subprocess.run([str(binary), str(temporary / (label + ".onnx")),
                                        str(temporary / "inputs.bin"), str(output_path)],
                                       check=True, capture_output=True, text=True, timeout=120)
            report = json.loads(completed.stdout)
            require(report["status"] == "pass" and report["runtime"] == "1.26.0", "Native kernel failed")
            actual = np.fromfile(output_path, np.float32).reshape(16, 3000)
            error = float(np.max(np.abs(actual - expected[label])))
            require(error < 1e-4, "Native kernel differs from independent numeric reference")
            if label in numerical:
                require(numerical[label]["native_outputs_sha256"] == fingerprint(actual), "Fresh-process output replay changed")
            original = expected["original_fp32_weights"]
            numerical[label] = {"native_reference_max_abs": error, "native_outputs_sha256": fingerprint(actual),
                                "approximation_max_abs_vs_original": float(np.max(np.abs(actual - original))),
                                "approximation_relative_rms_vs_original": float(np.linalg.norm(actual - original) / np.linalg.norm(original)),
                                "per_case_max_abs_vs_reference": np.max(np.abs(actual - expected[label]), axis=1).tolist()}
            rows.append({"cycle": cycle, "variant": label, **report, "runtime_stderr": completed.stderr})
            write(out / f"cycle-{cycle}.json", rows[-1])
    verify_inputs({"source_bindings": bindings})
    require(state_sha256(native.state_dict()) == parent_sha and not torch.cuda.is_initialized(), "Parent or CPU scope changed")
    medians = {label: statistics.median(r["p50_us"] for r in rows if r["variant"] == label) for label in models}
    write(out / "result.json", {"status": "pass", "source_bindings_unchanged": True,
          "plan_sha256": sha(out / "plan.json"), "weight_proof": proof, "graphs_built_and_removed": graph_proof,
          "fixture_inputs_sha256": fingerprint(inputs), "numerical": numerical, "cycles": rows,
          "median_of_process_p50_us": medians, "int4_over_u8_ratio": medians["int4_fp32_activation"] / medians["u8u8"],
          "parent_model_unchanged": True, "temporary_artifacts_removed": True,
          "counted_bytes_after": require_space(training, 380_000_000),
          "quality_measured": False, "native_host_qualified": False, "full_model_change_selected": False,
          "limitation": "Short isolated-kernel screen during GPU training, using the earlier packaged quadrature weights. Requires full-model accuracy, quality and quiet native timing before adoption."})
    print(json.dumps({"status": "pass", "medians_us": medians, "numerical": numerical}), flush=True)


if __name__ == "__main__":
    main()
