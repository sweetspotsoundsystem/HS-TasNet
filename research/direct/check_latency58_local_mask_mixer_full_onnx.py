"""Full-graph CPU parity and paired timing for an untrained local mixer fixture.

The preparation parent is fixed independently of future training selection.
Neither graph is a trained mixer candidate or a qualified plugin replacement.
"""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs
from research.direct.latency58_sdr_checkpoint import require_space


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Full mixer fixture plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-local-mask-mixer-full-onnx-plan-v1"
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require CUDA-hidden CPU1 fixture")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve existing result")
    require(plan["fixture_allowance_bytes"] == 240_000_000
            and plan["history_reserve_bytes"] == 350_000_000,
            "Preserve the current history completion reserve")
    before = require_space(plan, plan["fixture_allowance_bytes"] + plan["history_reserve_bytes"])
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    from research.direct.latency58_drum_accum_parent import load_parent
    from research.direct.latency58_local_mask_mixer import SpectralHeadWithLocalMixer, VERSION
    from research.direct.latency58_asymmetric_onnx import (
        INPUT_NAMES, INPUT_SHAPES, OUTPUT_NAMES, OUTPUT_SHAPES,
        make_export_copy, verification_cases, verify_onnx,
    )
    versions = {"numpy": np.__version__, "torch": torch.__version__,
                "onnx": onnx.__version__, "onnxruntime": ort.__version__}
    require(versions == plan["runtime_versions"], "Full fixture runtime changed")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(plan["seed"])
    generator = torch.Generator().manual_seed(plan["seed"])
    binding = plan["parent_plan"]
    require(sha(binding["path"]) == binding["sha256"], "Preparation parent changed")
    parent_plan = read(binding["path"])
    verify_inputs(parent_plan)
    model = load_parent(parent_plan).eval().requires_grad_(False)
    parent_fingerprint = state_sha256(model.state_dict())
    require(parent_fingerprint == parent_plan["parent"]["model_state_sha256"], "Wrong preparation parent")
    original_head = model.to_spec_masks
    mixed_head = SpectralHeadWithLocalMixer(original_head).eval().requires_grad_(False)
    require(sum(p.numel() for p in mixed_head.mixer.parameters()) == 536,
            "Correction parameter count changed")
    began = time.monotonic()
    # Both export copies use exactly the same original weights. At initialization,
    # the added module must preserve every output and state even from nonzero state.
    original_copy = make_export_copy(model)
    model.to_spec_masks = mixed_head
    zero_copy = make_export_copy(model)
    zero_rows = []
    with torch.inference_mode():
        for case in verification_cases(8, ()):
            inputs = (torch.from_numpy(case["audio"][:, :128].copy()[None]),
                      *(torch.from_numpy(s.copy()) for s in case["initial_states"]))
            baseline, zero = original_copy(*inputs), zero_copy(*inputs)
            require(all(torch.equal(a, b) for a, b in zip(baseline, zero, strict=True)),
                    "Zero correction changed an export-copy output or state")
            zero_rows.append({"case": case["name"], "all_outputs_and_states_exact": True})
        mixed_head.mixer.out.weight.copy_(
            torch.randn(mixed_head.mixer.out.weight.shape, generator=generator) * .003)
        mixed_head.mixer.out.bias.copy_(
            torch.randn(mixed_head.mixer.out.bias.shape, generator=generator) * .002)
    del original_copy, zero_copy
    fixture_fingerprint = state_sha256(model.state_dict())
    fixture_coefficients = {name: value.detach().tolist()
                            for name, value in mixed_head.mixer.state_dict().items()}
    write(out / "untrained-mixer-coefficients.json", fixture_coefficients)
    rng = torch.get_rng_state().clone()
    graphs, parity, sessions = {}, {}, {}
    for kind in ("parent", "active_fixture"):
        model.to_spec_masks = original_head if kind == "parent" else mixed_head
        fingerprint = parent_fingerprint if kind == "parent" else fixture_fingerprint
        require(state_sha256(model.state_dict()) == fingerprint, "Fixture identity changed")
        wrapper = make_export_copy(model)
        path = out / (kind + "-full-graph.onnx")
        require(not path.exists(), "Preserve prior full graph")
        inputs = (torch.zeros(INPUT_SHAPES[0]), *model.initial_state(1))
        with torch.inference_mode():
            example = wrapper(*inputs)
            torch.onnx.export(wrapper, inputs, str(path), export_params=True,
                              opset_version=17, do_constant_folding=True, dynamo=False,
                              external_data=False, input_names=list(INPUT_NAMES),
                              output_names=list(OUTPUT_NAMES))
        graph = onnx.load(str(path), load_external_data=False)
        require(tuple(value.name for value in graph.graph.output) == OUTPUT_NAMES
                and not any(p.data_location == onnx.TensorProto.EXTERNAL for p in graph.graph.initializer),
                "Full graph ABI or self-contained storage changed")
        for value, tensor, shape in zip(graph.graph.output, example, OUTPUT_SHAPES, strict=True):
            require(tuple(tensor.shape) == shape, "Full export-copy shape differs")
            dims = value.type.tensor_type.shape
            dims.ClearField("dim")
            for size in shape:
                dims.dim.add().dim_value = size
        onnx.helper.set_model_props(graph, {
            "purpose": "untrained_local_mixer_full_graph_fixture_only",
            "fixture": kind, "mixer_version": VERSION, "plan_sha256": args.plan_sha256,
            "preparation_parent_state_sha256": parent_fingerprint,
            "graph_delay_samples": "128", "host_queue_qualified": "false",
            "trained_candidate": "false", "quality_selected": "false",
        })
        onnx.save(graph, str(path))
        onnx.checker.check_model(str(path), full_check=True)
        graph_row = {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size,
                     "state_sha256": fingerprint,
                     "node_types_before_runtime_optimization": dict(Counter(n.op_type for n in graph.graph.node))}
        del graph
        graphs[kind] = graph_row
        require(sum(row["bytes"] for row in graphs.values()) + 1_000_000 < plan["fixture_allowance_bytes"],
                "Full graphs exceed their separate allowance")
        parity[kind] = verify_onnx(model, wrapper, path, hops=plan["verify_hops"], audio_paths=(), threads=1)
        require(parity[kind]["passed"] and state_sha256(model.state_dict())
                == state_sha256(wrapper.model.state_dict()) == fingerprint,
                "Full native/export-copy/ORT parity or weights changed")
        write(out / (kind + "-parity.json"), parity[kind])
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        sessions[kind] = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        require(sessions[kind].get_providers() == ["CPUExecutionProvider"], "Unexpected timing backend")
        print({"event": "full_graph_parity_pass", "fixture": kind, "graph_bytes": path.stat().st_size}, flush=True)
        del wrapper
    require(graphs["active_fixture"]["node_types_before_runtime_optimization"]["Conv"]
            == graphs["parent"]["node_types_before_runtime_optimization"]["Conv"] + 2,
            "Full graph lost the two correction convolutions")
    case = next(verification_cases(8, ()))
    chunk = np.ascontiguousarray(case["audio"][:, :128][None])

    def calls(kind, count):
        state = [s.copy() for s in case["initial_states"]]
        wall_start, cpu_start = time.perf_counter_ns(), time.thread_time_ns()
        for _ in range(count):
            output = sessions[kind].run(list(OUTPUT_NAMES), dict(zip(INPUT_NAMES, [chunk, *state], strict=True)))
            state = output[1:]
        cpu_ns = time.thread_time_ns() - cpu_start
        wall_ns = time.perf_counter_ns() - wall_start
        require(all(np.isfinite(value).all() for value in output), "Nonfinite timed graph output")
        return {"wall_ms_per_call": wall_ns / count / 1e6,
                "calling_thread_cpu_ms_per_call": cpu_ns / count / 1e6,
                "last_output_checksum": float(output[0].sum())}

    for kind in sessions:
        calls(kind, plan["warmup_calls"])
    timings, pairs = [], []
    for pair in range(plan["timed_pairs"]):
        order = ("parent", "active_fixture") if pair % 2 == 0 else ("active_fixture", "parent")
        rows = {}
        for kind in order:
            rows[kind] = calls(kind, plan["calls_per_pair"])
            timings.append({"pair": pair, "order": list(order), "fixture": kind, **rows[kind]})
        pairs.append({"pair": pair, **{field: rows["active_fixture"][field] - rows["parent"][field]
                       for field in ("wall_ms_per_call", "calling_thread_cpu_ms_per_call")}})
    summary = {}
    for kind in (*sessions, "active_minus_parent"):
        rows = pairs if kind == "active_minus_parent" else [r for r in timings if r["fixture"] == kind]
        summary[kind] = {}
        for field in ("wall_ms_per_call", "calling_thread_cpu_ms_per_call"):
            values = [row[field] for row in rows]
            summary[kind][field] = {"minimum": min(values), "median": float(np.median(values)), "maximum": max(values)}
    require(parity["active_fixture"]["cases"][0]["all_output_and_state_trajectory_sha256"]["ort"]
            != parity["parent"]["cases"][0]["all_output_and_state_trajectory_sha256"]["ort"],
            "Nonzero correction had no full-graph effect")
    require(state_sha256(model.state_dict()) == fixture_fingerprint
            and torch.equal(torch.get_rng_state(), rng) and not torch.cuda.is_initialized(),
            "Fixture weights, RNG or CPU scope changed")
    model.to_spec_masks = original_head
    require(state_sha256(model.state_dict()) == parent_fingerprint, "Original learned tensors changed")
    verify_inputs(plan)
    require(all(sha(row["path"]) == row["sha256"] for row in graphs.values()), "Saved graph changed")
    after = require_space(plan, plan["history_reserve_bytes"])
    result = {"schema": "latency58-local-mask-mixer-full-onnx-result-v1", "status": "pass",
              "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"],
              "source_bindings_unchanged": True, "runtime_versions": versions,
              "preparation_parent_state_sha256": parent_fingerprint, "fixture_state_sha256": fixture_fingerprint,
              "zero_export_copy_cases": zero_rows, "graphs": graphs, "parity": parity,
              "timings": timings, "paired_differences": pairs, "timing_summary": summary,
              "elapsed_seconds": time.monotonic() - began, "counted_bytes_before": before,
              "counted_bytes_after": after, "history_reserve_bytes": plan["history_reserve_bytes"],
              "cuda_initialized": False, "training_updates_executed": 0,
              "validation_material_used": False, "quality_selected": False, "training_parent_selected": False,
              "limitations": ["Fixed nonzero synthetic correction; no learned mixer or quality evidence.",
                              "Zero identity is checked in export copies; the full zero-mixer ONNX graph is not exported.",
                              "Full graph timing includes Python session calls on a concurrent Linux host, not M4 callbacks.",
                              "Timing extrema are block means, not individual callback minima or maxima.",
                              "Calling-thread CPU time omits descheduling; no callback deadline guarantee.",
                              "Graph parity does not qualify plugin queue behavior or a future trained model."]}
    write(out / "result.json", result)
    print({"status": "pass", "timing_summary": summary, "history_reserve_bytes": plan["history_reserve_bytes"]}, flush=True)


if __name__ == "__main__":
    main()
