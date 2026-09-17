"""Isolate integer projection arithmetic from FFT and recurrent trajectory drift."""
import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, write, sha, require


def main():
    import numpy as np
    import onnx
    from onnx import helper as h, numpy_helper as nh, TensorProto as T
    import onnxruntime as ort
    import torch
    from research.direct.latency58_asymmetric import Latency58AsymmetricModel
    from research.direct.latency58_residual_model import load_checkpoint, BASE_STATE
    from research.direct.latency58_int8_reference import make_reference, IntegerLinear
    from research.direct.train_latency58 import state_sha256, verify_inputs
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and ort.__version__ == "1.26.0", "Require CPU and shipping runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source = read(PHASE / "full-magnitude-001/plan.json")
    quantized = read(PHASE / "m4-int8-screen-002/result.json")["quantized"]
    require(sha(quantized["path"]) == quantized["sha256"], "Graph changed")
    parent, _ = load_checkpoint(source["parent_checkpoint"]["path"], source["parent_checkpoint"]["sha256"])
    native = Latency58AsymmetricModel()
    native.load_state_dict({key: value for key, value in parent.state_dict().items() if key != "fixed_residual_share"}, strict=True)
    native.eval().requires_grad_(False)
    require(state_sha256(native.state_dict()) == BASE_STATE, "Source state differs")
    graph = onnx.load(quantized["path"], load_external_data=False)
    stored = {value.name: nh.to_array(value) for value in graph.graph.initializer}
    reference, _ = make_reference(native, graph)
    out = PHASE / "m4-int8-projection-arithmetic-001"
    require(not out.exists(), "Preserve arithmetic diagnostics")
    out.mkdir()
    bindings = {**source["source_bindings"], str(Path(__file__).resolve()): sha(__file__),
                str(ROOT / "research/direct/latency58_int8_reference.py"): sha(ROOT / "research/direct/latency58_int8_reference.py"),
                quantized["path"]: quantized["sha256"]}
    write(out / "plan.json", {"source_bindings": bindings, "ort_version": ort.__version__,
          "purpose": "Same exact projection input for CPU PyTorch integer and ORT arithmetic; no FFT or recurrence"})
    rows = []
    rng = np.random.default_rng(20260912)
    for name, projection in reference.named_modules():
        if not isinstance(projection, IntegerLinear):
            continue
        prefix = projection.proof["initializer"]
        weight, zero, scale = (stored[prefix + suffix].copy() for suffix in ("_quantized", "_zero_point", "_scale"))
        constants = [nh.from_array(value, key) for key, value in (("W", weight), ("WZ", zero), ("WS", scale))]
        nodes = [h.make_node("DynamicQuantizeLinear", ["X"], ["Q", "S", "Z"]),
                 h.make_node("MatMulInteger", ["Q", "W", "Z", "WZ"], ["I"]),
                 h.make_node("Cast", ["I"], ["F"], to=T.FLOAT), h.make_node("Mul", ["S", "WS"], ["SS"]),
                 h.make_node("Mul", ["F", "SS"], ["Y0"])]
        if projection.bias is not None:
            constants.append(nh.from_array(projection.bias.numpy(), "B"))
            nodes.append(h.make_node("Add", ["Y0", "B"], ["Y"]))
        else:
            nodes.append(h.make_node("Identity", ["Y0"], ["Y"]))
        model = h.make_model(h.make_graph(nodes, "projection", [h.make_tensor_value_info("X", T.FLOAT, [1, weight.shape[0]])],
            [h.make_tensor_value_info("Y", T.FLOAT, [1, weight.shape[1]])], constants), opset_imports=[h.make_opsetid("", 17)], ir_version=10)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        session = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
        maximum = 0.
        for amplitude in (0., .0001, .01, .1, 1., 10.):
            x = (rng.normal(size=(1, weight.shape[0])) * amplitude).astype(np.float32)
            with torch.inference_mode():
                expected = projection(torch.from_numpy(x)).numpy()
            actual = session.run(["Y"], {"X": x})[0]
            maximum = max(maximum, float(np.abs(expected - actual).max()))
        row = {"projection": name, "six_input_levels_maximum_error": maximum, "weight_proof": projection.proof}
        rows.append(row)
        print(json.dumps(row), flush=True)
    verify_inputs({"source_bindings": bindings})
    write(out / "result.json", {"status": "diagnostic_complete", "projections": rows,
          "maximum_error": max(row["six_input_levels_maximum_error"] for row in rows),
          "source_bindings_unchanged": True, "native_host_qualified": False, "plugin_modified": False})


if __name__ == "__main__":
    main()
