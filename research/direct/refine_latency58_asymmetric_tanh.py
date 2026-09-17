"""Test FP64 for both GRU Tanh activations with the public FP32 graph ABI.

This is a measured arithmetic hypothesis, not a quality or runtime promotion.
The learned tensors, both FFTs and other operations are preserved exactly.
"""
from __future__ import annotations

import argparse
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
    require(sha(args.plan) == args.plan_sha256, "Plan changed")
    plan = json.loads(args.plan.read_text())
    require(plan["schema"] == "latency58-asymmetric-tanh-refinement-plan-v1"
            and all(sha(p) == h for p, h in plan["source_bindings"].items()), "Inputs differ")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CPU1 with CUDA hidden")
    out = Path(plan["output_directory"])
    output = out / "asymmetric500-gru-tanh64.onnx"
    require(out.is_dir() and not output.exists() and not (out / "result.json").exists(), "Preserve previous attempt")
    import torch
    import onnx
    import onnxruntime as ort
    from research.direct.latency58_asymmetric_checkpoint import load_model_state, make_model
    from research.direct.latency58_evaluate import model_state_sha256
    from research.direct.latency58_asymmetric_onnx import make_export_copy, verify_onnx

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260907)
    torch.use_deterministic_algorithms(True)
    began = time.monotonic()
    graph = onnx.load(plan["graph"]["path"], load_external_data=False)
    original_tensors = {v.name: v.SerializeToString() for v in graph.graph.initializer}
    nodes = []
    changed = []
    for node in graph.graph.node:
        attrs = {v.name: onnx.helper.get_attribute_value(v) for v in node.attribute}
        if node.op_type == "Tanh" and node.name.startswith("/fusion_branch/"):
            require(node.name in ("/fusion_branch/Tanh", "/fusion_branch/Tanh_1") and not attrs
                    and len(node.input) == 1 and len(node.output) == 1, "Unexpected GRU activation")
            original_output, original_input = node.output[0], node.input[0]
            node.input[0] = original_input + "_tanh64"
            node.output[0] = original_output + "_tanh64"
            nodes.extend((onnx.helper.make_node("Cast", [original_input], [node.input[0]],
                          name=node.name + "ToDouble", to=onnx.TensorProto.DOUBLE), node,
                          onnx.helper.make_node("Cast", [node.output[0]], [original_output],
                          name=node.name + "ToFloat", to=onnx.TensorProto.FLOAT)))
            changed.append(node.name)
        else:
            nodes.append(node)
    require(changed == ["/fusion_branch/Tanh", "/fusion_branch/Tanh_1"], "Change exactly both GRU Tanh nodes")
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    props = {v.key: v.value for v in graph.metadata_props}
    props.update({"hs_tasnet.gru_tanh_precision": "float64_internal_float32_io",
                  "hs_tasnet.arithmetic_parent_onnx_sha256": plan["graph"]["sha256"],
                  "hs_tasnet.arithmetic_refinement_plan_sha256": args.plan_sha256})
    onnx.helper.set_model_props(graph, props)
    require(original_tensors == {v.name: v.SerializeToString() for v in graph.graph.initializer},
            "Graph tensor bytes changed")
    onnx.checker.check_model(graph, full_check=True)
    with output.open("xb") as stream:
        stream.write(graph.SerializeToString())
    model = make_model(plan["parent_checkpoint"])
    require(load_model_state(model, plan["checkpoint"]) == plan["step"]
            and model_state_sha256(model) == plan["model_state_sha256"], "Model differs")
    rng = torch.get_rng_state().clone()
    wrapper = make_export_copy(model)
    verification = verify_onnx(model, wrapper, output, hops=plan["verify_hops"],
                               audio_paths=(plan["mixture"]["path"],), threads=1)
    require(torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
            and model_state_sha256(model) == model_state_sha256(wrapper.model) == plan["model_state_sha256"]
            and all(sha(p) == h for p, h in plan["source_bindings"].items()), "Source/model/runtime changed")
    result = {"status": "passed_short_fixtures_only" if verification["passed"] else "failed_not_qualified",
              "quality_or_native_qualified": False, "plan_sha256": args.plan_sha256,
              "source_bindings_unchanged": True, "original_tensor_bytes_unchanged": True,
              "graph": {"path": str(output), "sha256": sha(output), "bytes": output.stat().st_size},
              "graph_transform": "Only both GRU Tanh input casts FP64 and output casts FP32; both DFTs and all other ops unchanged",
              "checkpoint": plan["checkpoint"], "model_state_sha256": plan["model_state_sha256"],
              "runtime_versions": {"torch": torch.__version__, "onnxruntime": ort.__version__, "onnx": onnx.__version__},
              "verification": verification, "elapsed_seconds": time.monotonic() - began}
    with (out / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k != "verification"}, allow_nan=False), flush=True)
    require(verification["passed"], "Short numerical verification failed")


if __name__ == "__main__":
    main()
