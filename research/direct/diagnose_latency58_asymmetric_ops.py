"""Compare ORT operations against Torch and FP64 using identical inputs.

The retained worst-state witness fixes a real music hop. Instrumentation must
reproduce an uninstrumented ORT_DISABLE_ALL graph exactly before attributing
local differences. No model, graph or deployment artifact is changed.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time

from research.direct.latency58_checkpoint import require, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Diagnostic plan changed")
    plan = json.loads(args.plan.read_text())
    require(plan["schema"] == "latency58-asymmetric-ops-plan-v1"
            and all(sha(p) == h for p, h in plan["source_bindings"].items()), "Diagnostic inputs differ")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CPU1 with CUDA hidden")
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve previous diagnostic")
    import numpy as np
    import torch
    import torch.nn.functional as F
    import onnx
    import onnxruntime as ort
    from research.direct.latency58_asymmetric_checkpoint import load_model_state, make_model
    from research.direct.latency58_asymmetric import AsymmetricState
    from research.direct.latency58_evaluate import model_state_sha256
    from research.direct.latency58_asymmetric_onnx import INPUT_NAMES, OUTPUT_NAMES

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260907)
    torch.use_deterministic_algorithms(True)
    began = time.monotonic()
    witness = dict(np.load(plan["witness"]["path"], allow_pickle=False))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    plain = ort.InferenceSession(plan["graph"]["path"], options, providers=["CPUExecutionProvider"])
    graph = onnx.shape_inference.infer_shapes(onnx.load(plan["graph"]["path"], load_external_data=False))
    constants = {v.name: onnx.numpy_helper.to_array(v).copy() for v in graph.graph.initializer}
    for node in graph.graph.node:
        if node.op_type == "Constant":
            attrs = {v.name: onnx.helper.get_attribute_value(v) for v in node.attribute}
            require(set(attrs) == {"value"}, "Unexpected constant representation")
            constants[node.output[0]] = onnx.numpy_helper.to_array(attrs["value"]).copy()
    info = {v.name: v for v in (*graph.graph.input, *graph.graph.output, *graph.graph.value_info)}
    original_outputs = set(OUTPUT_NAMES)
    for name, value in info.items():
        if name not in original_outputs and value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT:
            graph.graph.output.append(copy.deepcopy(value))
    onnx.checker.check_model(graph)
    traced = ort.InferenceSession(graph.SerializeToString(), options, providers=["CPUExecutionProvider"])
    require(plain.get_providers() == traced.get_providers() == ["CPUExecutionProvider"], "CPU only")
    model = make_model(plan["parent_checkpoint"])
    require(load_model_state(model, plan["checkpoint"]) == plan["step"]
            and model_state_sha256(model) == plan["model_state_sha256"], "Model identity differs")
    rng = torch.get_rng_state().clone()
    replay = {}
    for backend in ("native", "ort_disabled"):
        inputs = [witness["audio_chunk"], *(witness[f"{backend}_prior_{i}"] for i in range(4))]
        feed = dict(zip(INPUT_NAMES, inputs, strict=True))
        values = plain.run(list(OUTPUT_NAMES), feed)
        captured = dict(zip((v.name for v in traced.get_outputs()), traced.run(None, feed), strict=True))
        require(all(np.array_equal(value, captured[name]) for name, value in zip(OUTPUT_NAMES, values, strict=True)),
                "Instrumentation changes original outputs")
        if backend == "native":
            with torch.inference_mode():
                prediction, state = model.forward_chunk(torch.from_numpy(inputs[0]),
                    AsymmetricState(*(torch.from_numpy(v) for v in inputs[1:])))
            actual = [prediction.numpy(), *(v.numpy() for v in state)]
        else:
            actual = values
        require(all(np.array_equal(value, witness[f"{backend}_output_{i}"]) for i, value in enumerate(actual)),
                "Witness replay differs")
        replay[backend] = {"witness_reproduced_exactly": True, "instrumentation_outputs_exact": True}
        if backend == "native":
            tensors = {**constants, **feed, **captured}
            replay[backend]["same_input_output_errors"] = [float(np.max(np.abs(a.astype(np.float64) - b)))
                                                           for a, b in zip(actual, values, strict=True)]

    def operation(node, inputs, dtype):
        x = [torch.from_numpy(np.array(v, copy=True)).to(dtype) for v in inputs]
        a = {v.name: onnx.helper.get_attribute_value(v) for v in node.attribute}
        op = node.op_type
        if op == "DFT":
            value = x[0][..., 0] if x[0].shape[-1] == 1 else torch.complex(x[0][..., 0], x[0][..., 1])
            fn = torch.fft.ifft if a.get("inverse", 0) else torch.fft.rfft if a.get("onesided", 0) else torch.fft.fft
            result = fn(value, n=int(inputs[1]), dim=a.get("axis", 1))
            return torch.view_as_real(result)
        if op == "Conv":
            require(a.get("auto_pad", b"NOTSET") == b"NOTSET" and a.get("pads", [0, 0])[0] == a.get("pads", [0, 0])[1],
                    "Unsupported convolution padding")
            return F.conv1d(x[0], x[1], x[2] if len(x) > 2 else None, stride=a.get("strides", [1]),
                            padding=a.get("pads", [0, 0])[0], dilation=a.get("dilations", [1]), groups=a.get("group", 1))
        if op == "Gemm":
            left = x[0].T if a.get("transA", 0) else x[0]
            right = x[1].T if a.get("transB", 0) else x[1]
            return torch.addmm(x[2], left, right, alpha=a.get("alpha", 1.), beta=a.get("beta", 1.))
        if op == "MatMul": return torch.matmul(*x)
        if op == "Sigmoid": return torch.sigmoid(x[0])
        if op == "Tanh": return torch.tanh(x[0])
        if op == "Relu": return torch.relu(x[0])
        if op == "Softmax": return torch.softmax(x[0], dim=a.get("axis", -1))
        if op == "ReduceMean": return x[0].mean(dim=tuple(a["axes"]), keepdim=bool(a.get("keepdims", 1)))
        if op == "Sqrt": return torch.sqrt(x[0])
        if op == "Mul": return x[0] * x[1]
        if op == "Add": return x[0] + x[1]
        if op == "Sub": return x[0] - x[1]
        if op == "Div": return x[0] / x[1]
        raise ValueError(op)

    selected = {"DFT", "Conv", "Gemm", "MatMul", "Sigmoid", "Tanh", "Relu", "Softmax",
                "ReduceMean", "Sqrt", "Mul", "Add", "Sub", "Div"}
    rows = []
    with torch.inference_mode():
        for node in graph.graph.node:
            if node.op_type not in selected or node.output[0] not in tensors or tensors[node.output[0]].dtype != np.float32:
                continue
            require(len(node.output) == 1 and all(name in tensors for name in node.input), "Missing trace input")
            inputs = [tensors[name] for name in node.input]
            actual = tensors[node.output[0]].astype(np.float64)
            fp32 = operation(node, inputs, torch.float32).numpy().astype(np.float64)
            fp64 = operation(node, inputs, torch.float64).numpy()
            require(actual.shape == fp32.shape == fp64.shape and all(np.isfinite(v).all() for v in (actual, fp32, fp64)),
                    "Invalid local output")
            rows.append({"node": node.name, "op": node.op_type, "shape": list(actual.shape),
                         "identical_inputs": True,
                         "ort_vs_torch_fp32_max_abs": float(np.max(np.abs(actual - fp32))),
                         "ort_vs_fp64_max_abs": float(np.max(np.abs(actual - fp64))),
                         "torch_fp32_vs_fp64_max_abs": float(np.max(np.abs(fp32 - fp64))),
                         "output_rms": float(np.sqrt(np.mean(actual ** 2)))})
    require(torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
            and model_state_sha256(model) == plan["model_state_sha256"]
            and all(sha(p) == h for p, h in plan["source_bindings"].items()), "Inputs/model/runtime scope changed")
    result = {"status": "diagnostic_completed", "quality_or_native_qualified": False,
              "plan_sha256": args.plan_sha256, "source_bindings_unchanged": True,
              "runtime_versions": {"torch": torch.__version__, "onnxruntime": ort.__version__, "onnx": onnx.__version__},
              "witness_input_start": int(witness["input_start"]), "replay": replay, "operations": rows,
              "instrumentation_graph_saved": False, "elapsed_seconds": time.monotonic() - began}
    with (out / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
