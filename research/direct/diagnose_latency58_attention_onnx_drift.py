"""Compare bounded FP32 execution changes against one cached native trajectory.

The cache lives only in RAM and must match the native trajectory hash already
recorded by the long checker. No model parameters or tolerances are changed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs


def fuse_current(graph):
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
    def axis(node):
        return next((a.i for a in node.attribute if a.name == "axis"), 0)
    inputs = [node.input[0] for node in removed if node.op_type == "Gather"
              and node.input[0] in external_inputs and axis(node) == 1]
    hidden = {node.input[0] for node in removed if node.op_type == "Gather"
              and node.input[0] in external_inputs and axis(node) == 0}
    require(len(inputs) == 1 and len(hidden) == 1 and len(weights) == 8
            and len(external_inputs - weights.keys()) == 3, "Recurrence boundary changed")
    feature_input, hidden_input = inputs[0], next(iter(hidden))
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
    layer_input = feature_input  # [batch=1, time=1, feature=1000]
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
        nodes.append(helper.make_node("Gather", [hidden_input, indices[layer]],
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



def precise_nonlinear(graph):
    from onnx import helper, TensorProto as T
    changed, nodes = [], []
    for original in graph.graph.node:
        if not original.name.startswith("/fusion_branch/") or original.op_type not in ("Sigmoid", "Tanh"):
            nodes.append(original)
            continue
        node = copy.deepcopy(original)
        inp, out = node.input[0], node.output[0]
        node.input[0], node.output[0] = inp + "__nonlinear64", out + "__nonlinear64"
        nodes.extend((helper.make_node("Cast", [inp], [node.input[0]], name=node.name + "/input64", to=T.DOUBLE),
                      node, helper.make_node("Cast", [node.output[0]], [out], name=node.name + "/output32", to=T.FLOAT)))
        changed.append(original.name)
    require(len(changed) == 6, "Expected two GRU layers with two sigmoids and one tanh each")
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    del graph.graph.value_info[:]
    return {"changed_nonlinear_nodes": changed, "parameters_unchanged": True}


def main():
    import numpy as np
    import onnx
    import onnxruntime as ort
    import soundfile as sf
    import torch
    from research.direct.check_latency58_best_onnx_memory import selected_endpoint
    from research.direct.latency58_best_onnx_export import build
    from research.direct.latency58_best_onnx import interface, TOLERANCES
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and ort.__version__ == "1.26.0" and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "Require reviewed CPU runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    budget = read(PHASE / "temporal-attention-001/plan.json")
    require_space(budget, 5_000_000)
    model, payload, training, checkpoint, quality_path, review_path = selected_endpoint()
    fingerprint = state_sha256(model.state_dict())
    _, original = build(model, payload, training, checkpoint, quality_path)
    contract = interface(model)
    short_root = PHASE / "best-model-onnx-memory-001"
    screen = read(short_root / "result.json")
    require(hashlib.sha256(original.SerializeToString()).hexdigest() == screen["graph_sha256"], "Original graph differs")
    long_path = PHASE / "best-model-onnx-long-001/repeat-1.json"
    previous = read(long_path)
    require(not previous["passed"] and previous["comparisons"]["native_vs_export_copy"]["waveform_max_abs"] == 0,
            "Require the observed ORT-only long drift")
    music = Path(previous["source"]["path"])
    audio, rate = sf.read(music, frames=previous["physical_samples"], dtype="float32", always_2d=True)
    require(rate == 44100 and len(audio) == 30 * 44100 + 37, "Music fixture differs")
    audio = np.ascontiguousarray(audio.T)
    require(hashlib.sha256(audio.tobytes()).hexdigest() == previous["decoded_input_sha256"], "Decoded music differs")
    padded = np.pad(audio, ((0, 0), (0, (-audio.shape[-1]) % 128 + 128)))
    calls = padded.shape[-1] // 128
    sizes = [int(np.prod(shape)) for shape in contract["output_shapes"]]
    offsets = np.cumsum([0, *sizes])
    cache = np.empty((calls, sum(sizes)), np.float32)
    paths = [Path(__file__).resolve(), long_path, music, review_path, short_root / "plan.json", short_root / "result.json",
             ROOT / "research/direct/check_latency58_fused_gru.py"]
    bindings = {**read(short_root / "plan.json")["source_bindings"], **{str(p): sha(p) for p in paths}}
    verify_inputs({"source_bindings": bindings})
    out = PHASE / "attention-onnx-drift-diagnostic-001"
    require(not out.exists(), "Preserve diagnostics")
    out.mkdir()
    write(out / "plan.json", {"source_bindings": bindings, "checkpoint": checkpoint,
          "tolerances": TOLERANCES, "samples": audio.shape[-1], "native_cache_bytes": cache.nbytes,
          "variants": ["original_unoptimized", "fused_gru", "precise_gru_nonlinear"],
          "scope": "Bounded numerical diagnostic; no graph saved, no quality or performance claims"})
    # Validate both graph transforms before spending time on the native cache.
    for transformation in (fuse_current, precise_nonlinear):
        probe = copy.deepcopy(original)
        transformation(probe)
        onnx.checker.check_model(probe, full_check=True)
        del probe
    began, digest = time.monotonic(), hashlib.sha256()
    state = model.initial_state(1)
    with torch.inference_mode():
        for hop in range(calls):
            chunk = torch.from_numpy(np.ascontiguousarray(padded[None, :, hop * 128:(hop + 1) * 128]))
            output, state = model.forward_chunk(chunk, state)
            for index, tensor in enumerate((output, *state)):
                value = tensor.detach().numpy()
                cache[hop, offsets[index]:offsets[index + 1]] = value.reshape(-1)
                digest.update(np.ascontiguousarray(value).tobytes())
            if (hop + 1) % 2000 == 0:
                print(json.dumps({"native_cache_hops": hop + 1}), flush=True)
    require(digest.hexdigest() == previous["all_output_and_state_trajectory_sha256"]["native"],
            "Cached native oracle differs from the completed long trajectory")
    summaries = []
    for label in ("original_unoptimized", "fused_gru", "precise_gru_nonlinear"):
        graph = copy.deepcopy(original)
        proof = fuse_current(graph) if label == "fused_gru" else (precise_nonlinear(graph) if label == "precise_gru_nonlinear" else {})
        onnx.checker.check_model(graph, full_check=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_DISABLE_ALL if label == "original_unoptimized"
                                            else ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        data = graph.SerializeToString()
        session = ort.InferenceSession(data, sess_options=options, providers=["CPUExecutionProvider"])
        for direction, actual in (("input", session.get_inputs()), ("output", session.get_outputs())):
            require(tuple(v.name for v in actual) == contract[direction + "_names"]
                    and tuple(tuple(v.shape) for v in actual) == contract[direction + "_shapes"]
                    and all(v.type == "tensor(float)" for v in actual), "Variant ABI changed")
        states = [np.zeros(shape, np.float32) for shape in contract["state_shapes"]]
        maximum = np.zeros(len(sizes))
        rms, closure = 0., 0.
        for hop in range(calls):
            chunk = np.ascontiguousarray(padded[None, :, hop * 128:(hop + 1) * 128])
            actual = session.run(list(contract["output_names"]), dict(zip(contract["input_names"], [chunk, *states], strict=True)))
            require(all(np.isfinite(value).all() for value in actual), "Nonfinite variant output")
            for index, value in enumerate(actual):
                difference = value.reshape(-1).astype(np.float64) - cache[hop, offsets[index]:offsets[index + 1]]
                maximum[index] = max(maximum[index], np.abs(difference).max() * (2**18 if index == 2 else 1))
                if index == 0:
                    rms = max(rms, float(np.sqrt(np.mean(difference.reshape(1, 4, 2, 128) ** 2, axis=(2, 3))).max()))
            expected_mix = np.zeros((1, 2, 128), np.float32) if hop == 0 else padded[None, :, (hop - 1) * 128:hop * 128]
            closure = max(closure, float(np.abs(actual[0].sum(1) - expected_mix).max()))
            states = actual[1:]
        passed = bool(maximum[0] <= TOLERANCES["waveform_max_abs"] and maximum[1:].max() <= TOLERANCES["state_max_abs_decoded_units"]
                      and rms <= TOLERANCES["stem_callback_rms"] and closure <= TOLERANCES["reconstruction_max_abs"])
        row = {"variant": label, "graph_sha256": hashlib.sha256(data).hexdigest(), "proof": proof,
               "maximum_errors_in_physical_units": dict(zip(contract["output_names"], maximum.tolist(), strict=True)),
               "maximum_stem_callback_rms": rms, "maximum_closure": closure, "original_tolerances_passed": passed}
        summaries.append(row)
        write(out / (label + ".json"), row)
        print(json.dumps(row), flush=True)
        del session, data, graph
    verify_inputs({"source_bindings": bindings})
    require(state_sha256(model.state_dict()) == fingerprint and not torch.cuda.is_initialized(), "Source model changed")
    write(out / "result.json", {"status": "diagnostic_complete", "variants": summaries,
          "source_bindings_unchanged": True, "native_cache_trajectory_sha256": digest.hexdigest(),
          "graph_saved": False, "quality_measured": False, "performance_measured": False,
          "elapsed_seconds": time.monotonic() - began, "counted_bytes_after": require_space(budget, 0)})


if __name__ == "__main__":
    main()
