"""Bounded in-memory GRU fusion trial; no training or deployment mutation.

Replace only the authenticated export's expanded two-layer recurrence with
standard ONNX GRU nodes. All non-recurrent nodes remain byte-for-byte equal;
the recurrent weights are permuted r,z,n -> z,r,h without changing values.
The original native Torch and export-copy trajectories remain the oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from research.direct.latency58_checkpoint import require, sha


def fuse(graph):
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    prefix = "/fusion_branch/"
    removed = [node for node in graph.graph.node if node.name.startswith(prefix)]
    require(len(removed) == 103, "Expanded recurrence changed")
    produced = {value for node in removed for value in node.output}
    external_inputs = {value for node in removed for value in node.input if value not in produced}
    weights = {value.name: value for value in graph.graph.initializer
               if value.name.startswith("model.fusion_branch.")}
    require(len(weights) == 8 and external_inputs - weights.keys() == {
        "/Concat_1_output_0", "/Div_1_output_0", "/conv_encode/Constant_output_0"},
        "Recurrence input boundary changed")
    retained = [node for node in graph.graph.node if not node.name.startswith(prefix)]
    observed = {value for node in retained for value in node.input if value in produced}
    require(observed == {prefix + "Concat_output_0", prefix + "Unsqueeze_2_output_0"},
            "Recurrence output boundary changed")
    original_retained = [node.SerializeToString() for node in retained]
    initializers = [value for value in graph.graph.initializer if value.name not in weights]
    original_nonrecurrent = {value.name: value.SerializeToString() for value in initializers}
    nodes, proof = [], []

    def constant(name, value):
        key = "/fused_gru/" + name
        initializers.append(numpy_helper.from_array(np.asarray(value, dtype=np.int64), key))
        return key

    indices = [constant("layer0", [0]), constant("layer1", [1])]
    layer_input = "/Concat_1_output_0"  # [batch=1, time=1, feature=1000]
    for layer in range(2):
        name = f"/fused_gru/layer{layer}/"
        converted = {}
        for kind in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            source = numpy_helper.to_array(weights[f"model.fusion_branch.{kind}_l{layer}"])
            require(source.dtype == np.float32 and source.shape == (
                (3000, 1000) if kind.startswith("weight") else (3000,)), "Weight ABI changed")
            reordered = np.concatenate((source[1000:2000], source[:1000], source[2000:]), axis=0)
            round_trip = np.concatenate((reordered[1000:2000], reordered[:1000], reordered[2000:]), axis=0)
            require(source.tobytes() == round_trip.tobytes(), "GRU weight permutation lost bytes")
            proof.append({"layer": layer, "tensor": kind,
                          "original_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
                          "reordered_sha256": hashlib.sha256(reordered.tobytes()).hexdigest(),
                          "inverse_permutation_bit_exact": True})
            converted[kind] = reordered
        for key, values in (("W", converted["weight_ih"]), ("R", converted["weight_hh"]),
                            ("B", np.concatenate((converted["bias_ih"], converted["bias_hh"])))):
            initializers.append(numpy_helper.from_array(np.ascontiguousarray(values[None]), name + key))
        nodes.append(helper.make_node("Gather", ["/Div_1_output_0", indices[layer]],
                                      [name + "initial_h"], name=name + "gather", axis=0))
        # Both sequence and batch axes are statically one. Y_h therefore also
        # serves directly as the next layer's [seq, batch, feature] input.
        nodes.append(helper.make_node("GRU", [layer_input, name + "W", name + "R", name + "B", "",
                                               name + "initial_h"], ["", name + "next_h"],
                                      name=name + "GRU", hidden_size=1000, linear_before_reset=1,
                                      direction="forward", layout=0))
        layer_input = name + "next_h"
    nodes.append(helper.make_node("Concat", ["/fused_gru/layer0/next_h", "/fused_gru/layer1/next_h"],
                                  [prefix + "Concat_output_0"], name="/fused_gru/stack", axis=0))
    nodes.append(helper.make_node("Identity", [layer_input], [prefix + "Unsqueeze_2_output_0"],
                                  name="/fused_gru/output"))
    rewritten, inserted = [], False
    for node in graph.graph.node:
        if node.name.startswith(prefix):
            if not inserted:
                rewritten.extend(nodes)
                inserted = True
        else:
            rewritten.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(rewritten)
    del graph.graph.initializer[:]
    graph.graph.initializer.extend(initializers)
    require([node.SerializeToString() for node in graph.graph.node
             if not node.name.startswith("/fused_gru/")] == original_retained,
            "Non-recurrent node bytes changed")
    require(all(original_nonrecurrent[value.name] == value.SerializeToString()
                for value in graph.graph.initializer if value.name in original_nonrecurrent),
            "Non-recurrent initializer changed")
    props = {item.key: item.value for item in graph.metadata_props}
    props.update({"hs_tasnet.gru_implementation": "ONNX GRU; two layers; linear_before_reset=1; zrh",
                  "hs_tasnet.gru_transform_source_sha256": sha(__file__)})
    helper.set_model_props(graph, props)
    onnx.checker.check_model(graph, full_check=True)
    return {"removed_nodes": len(removed), "replacement_nodes": len(nodes),
            "nonrecurrent_nodes_and_initializers_byte_exact": True, "weight_permutations": proof}


def session_for(graph_bytes, threads=1):
    import onnxruntime as ort
    from research.direct.latency58_asymmetric_onnx import INPUT_NAMES, INPUT_SHAPES, OUTPUT_NAMES, OUTPUT_SHAPES

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    session = ort.InferenceSession(graph_bytes, sess_options=options, providers=["CPUExecutionProvider"])
    require(session.get_providers() == ["CPUExecutionProvider"], "Unexpected provider")
    for actual, names, shapes in ((session.get_inputs(), INPUT_NAMES, INPUT_SHAPES),
                                  (session.get_outputs(), OUTPUT_NAMES, OUTPUT_SHAPES)):
        require(tuple(value.name for value in actual) == names and
                tuple(tuple(value.shape) for value in actual) == shapes and
                all(value.type == "tensor(float)" for value in actual), "Static FP32 ABI changed")
    return session


def benchmark(session, audio, warmup, measured):
    import numpy as np
    from research.direct.latency58_asymmetric_onnx import INPUT_NAMES, OUTPUT_NAMES, _initial_states

    states, times = _initial_states(), []
    for index in range(warmup + measured):
        chunk = np.ascontiguousarray(audio[:, index * 128:(index + 1) * 128][None])
        inputs = dict(zip(INPUT_NAMES, [chunk, *states], strict=True))
        start = time.perf_counter_ns()
        values = session.run(list(OUTPUT_NAMES), inputs)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        require(all(np.isfinite(value).all() for value in values), "Nonfinite timing output")
        states = values[1:]
        if index >= warmup:
            times.append(elapsed)
    budget = 128 / 44.1
    return {"warmup_hops": warmup, "measured_hops": measured, "times_ms": times,
            "p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)), "maximum_ms": max(times),
            "hop_budget_ms": budget, "deadline_misses": sum(value > budget for value in times),
            "scope": "Python ORT CPU1, local host diagnostic; no plugin queue or target-M4 qualification"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Plan changed")
    plan = json.loads(args.plan.read_text())
    require(plan["schema"] == "latency58-fused-gru-screen-v1", "Wrong schema")
    require(all(sha(path) == digest for path, digest in plan["bindings"].items()), "Input changed")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and all(os.environ.get(key) == "1"
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Expected CPU1")
    output = Path(plan["result"])
    require(not output.exists(), "Preserve outputs")
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    import soundfile as sf
    from research.direct.latency58_asymmetric_onnx import _run_case, verification_cases, make_export_copy, TOLERANCES
    from research.direct.latency58_teacher_checkpoint_v2 import make_model, load_model_state
    from research.direct.latency58_evaluate import model_state_sha256

    versions = {"torch": torch.__version__, "numpy": np.__version__, "onnx": onnx.__version__,
                "onnxruntime": ort.__version__, "soundfile": sf.__version__}
    require(versions == plan["runtime_versions"] and not torch.cuda.is_initialized(), "Runtime changed")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260907)
    torch.use_deterministic_algorithms(True)
    model = make_model(plan["parent_checkpoint"])
    require(load_model_state(model, plan["checkpoint"]) == 250 and
            model_state_sha256(model) == plan["model_state_sha256"], "Wrong native model")
    wrapper = make_export_copy(model)
    graph = onnx.load(plan["baseline_graph"]["path"], load_external_data=False)
    report = {"schema": plan["schema"], "plan_sha256": args.plan_sha256,
              "runtime_versions": versions, "baseline_graph": plan["baseline_graph"],
              "transformation": fuse(graph), "tolerances": TOLERANCES, "cases": [],
              "training_updates_executed": 0, "native_host_timing_qualified": False,
              "graph_written_to_disk": False, "source_bindings": plan["bindings"]}
    graph_bytes = graph.SerializeToString()
    report["fused_graph_sha256"] = hashlib.sha256(graph_bytes).hexdigest()
    report["fused_graph_bytes"] = len(graph_bytes)
    session = session_for(graph_bytes)
    for case in verification_cases(plan["verify_hops"], [plan["audio"]["path"]]):
        first = _run_case(model, wrapper, session, case)
        repeat = _run_case(model, wrapper, session, case)
        exact = first["all_output_and_state_trajectory_sha256"] == repeat["all_output_and_state_trajectory_sha256"]
        first.update(reset_replay_bit_exact=exact, reset_replay_passed=repeat["passed"])
        first["passed"] = first["passed"] and repeat["passed"] and exact
        report["cases"].append(first)
        print(json.dumps({"case": first["input"], "passed": first["passed"]}), flush=True)
    report["short_parity_passed"] = all(row["passed"] for row in report["cases"])
    if report["short_parity_passed"]:
        frames = 128 * (plan["warmup_hops"] + plan["measured_hops"])
        audio, rate = sf.read(plan["audio"]["path"], frames=frames, dtype="float32", always_2d=True)
        require(rate == 44100 and audio.shape == (frames, 2), "Timing audio shape differs")
        report["timing_fused"] = benchmark(session, audio.T, plan["warmup_hops"], plan["measured_hops"])
        del session
        baseline_session = session_for(Path(plan["baseline_graph"]["path"]).read_bytes())
        report["timing_baseline"] = benchmark(baseline_session, audio.T, plan["warmup_hops"], plan["measured_hops"])
        report["p50_ratio_fused_over_baseline"] = report["timing_fused"]["p50_ms"] / report["timing_baseline"]["p50_ms"]
    report["inputs_and_model_unchanged"] = (all(sha(path) == digest for path, digest in plan["bindings"].items())
        and model_state_sha256(model) == model_state_sha256(wrapper.model) == plan["model_state_sha256"]
        and not torch.cuda.is_initialized())
    report["status"] = "short_parity_pass" if report["short_parity_passed"] else "rejected_short_parity"
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    require(report["inputs_and_model_unchanged"] and report["short_parity_passed"], "Fusion screen failed")
    print(json.dumps({"status": report["status"], "ratio": report.get("p50_ratio_fused_over_baseline")}), flush=True)


if __name__ == "__main__":
    main()
