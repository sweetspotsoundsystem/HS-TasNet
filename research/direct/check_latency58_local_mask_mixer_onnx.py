"""Standalone synthetic mixer export, CPU parity and bounded backend timing.

This does not export a separation model or qualify an audio callback. Saved
graphs contain fixture coefficients, never trained or selected weights.
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
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Mixer backend plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-local-mask-mixer-onnx-plan-v1"
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require a bounded CUDA-hidden CPU1 fixture")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve existing proof")
    counted_before = require_space(plan, 5_000_000)
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    from research.direct.latency58_local_mask_mixer import LocalSpectralMaskMixer, VERSION
    versions = {"numpy": np.__version__, "torch": torch.__version__,
                "onnx": onnx.__version__, "onnxruntime": ort.__version__}
    require(versions == plan["runtime_versions"], "Backend runtime versions changed")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(plan["seed"])
    generator = torch.Generator().manual_seed(plan["seed"])
    began = time.monotonic()
    module = LocalSpectralMaskMixer().eval().requires_grad_(False)
    shape = (1, 1, 2 * 513 * 2 * 4)
    require(sum(p.numel() for p in module.parameters()) == 536, "Mixer size changed")
    cases = {"zeros": torch.zeros(shape)}
    for scale in (.2, 1., 8., 64.):
        cases[f"gaussian_scale_{scale:g}"] = torch.randn(shape, generator=generator) * scale
    for frequency in (0, 1, 257, 511, 512):
        value = torch.zeros(1, 1, 2, 513, 2, 4)
        value[0, 0, 0, frequency, 0, 1] = 1.
        cases[f"left_bass_real_bin_{frequency}"] = value.reshape(shape)
    alternating = torch.ones(1, 1, 2, 513, 2, 4)
    alternating[:, :, 0, ::2] *= -1
    alternating[:, :, 1, 1::2] *= -1
    cases["alternating_frequency_stereo"] = alternating.reshape(shape)
    rng = torch.get_rng_state().clone()
    graphs, parity, sessions, fingerprints = {}, [], {}, {}
    with torch.inference_mode():
        for kind in ("zero", "active"):
            if kind == "active":
                module.out.weight.copy_(torch.randn(module.out.weight.shape, generator=generator) * .003)
                module.out.bias.copy_(torch.randn(module.out.bias.shape, generator=generator) * .002)
            fingerprints[kind] = state_sha256(module.state_dict())
            path = out / (kind + "-synthetic-mixer.onnx")
            require(not path.exists(), "Preserve existing synthetic graph")
            torch.onnx.export(module, (cases["zeros"],), str(path), export_params=True,
                              opset_version=17, do_constant_folding=True, dynamo=False,
                              external_data=False, input_names=["logits"], output_names=["corrected_logits"])
            graph = onnx.load(str(path), load_external_data=False)
            require(not any(p.data_location == onnx.TensorProto.EXTERNAL for p in graph.graph.initializer),
                    "Require a self-contained fixture graph")
            onnx.helper.set_model_props(graph, {"purpose": "untrained_synthetic_mixer_fixture_only",
                                               "mixer_version": VERSION, "fixture": kind,
                                               "plan_sha256": args.plan_sha256})
            onnx.save(graph, str(path))
            onnx.checker.check_model(str(path), full_check=True)
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            options.add_session_config_entry("session.inter_op.allow_spinning", "0")
            session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
            require(session.get_providers() == ["CPUExecutionProvider"]
                    and len(session.get_inputs()) == len(session.get_outputs()) == 1
                    and tuple(session.get_inputs()[0].shape) == tuple(session.get_outputs()[0].shape) == shape
                    and session.get_inputs()[0].type == session.get_outputs()[0].type == "tensor(float)",
                    "CPU backend ABI differs")
            for name, value in cases.items():
                expected = module(value).numpy()
                actual = session.run(None, {"logits": value.numpy()})[0]
                error = np.abs(actual - expected)
                require(np.isfinite(actual).all()
                        and np.all(error <= plan["absolute_tolerance"] + plan["relative_tolerance"] * np.abs(expected)),
                        "Mixer backend numerical mismatch: " + kind + "/" + name)
                effect = float(np.abs(actual - value.numpy()).max())
                require(kind != "zero" or np.array_equal(actual, value.numpy()), "Zero fixture changed logits")
                replay = session.run(None, {"logits": value.numpy()})[0]
                require(np.array_equal(replay, actual), "Stateless backend replay differs")
                parity.append({"fixture": kind, "case": name, "max_absolute_error": float(error.max()),
                               "max_absolute_effect": effect, "replay_exact": True})
            require(kind != "active" or any(r["max_absolute_effect"] > 0 for r in parity if r["fixture"] == kind),
                    "Active fixture was optimized away")
            require(state_sha256(module.state_dict()) == fingerprints[kind], "Export mutated fixture weights")
            graphs[kind] = {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size,
                            "node_types_before_runtime_optimization": dict(Counter(n.op_type for n in graph.graph.node))}
            sessions[kind] = session
        session = sessions["active"]
        value = cases["gaussian_scale_1"]
        backend_input = {"logits": value.numpy()}
        calls = {"pytorch": lambda: module(value), "onnxruntime": lambda: session.run(None, backend_input)[0]}
        for call in calls.values():
            for _ in range(plan["warmup_calls"]):
                call()
        timings = []
        for pair in range(plan["timed_pairs"]):
            order = ("pytorch", "onnxruntime") if pair % 2 == 0 else ("onnxruntime", "pytorch")
            for backend in order:
                call = calls[backend]
                wall_start, cpu_start = time.perf_counter_ns(), time.thread_time_ns()
                for _ in range(plan["calls_per_pair"]):
                    last = call()
                cpu_ns, wall_ns = time.thread_time_ns() - cpu_start, time.perf_counter_ns() - wall_start
                checksum = float(last.sum())
                timings.append({"pair": pair, "order": list(order), "backend": backend,
                                "wall_ms_per_call": wall_ns / plan["calls_per_pair"] / 1e6,
                                "calling_thread_cpu_ms_per_call": cpu_ns / plan["calls_per_pair"] / 1e6,
                                "checksum": checksum})
    require(torch.equal(torch.get_rng_state(), rng) and not torch.cuda.is_initialized()
            and state_sha256(module.state_dict()) == fingerprints["active"], "Fixture RNG, weights or CPU scope changed")
    verify_inputs(plan)
    require(all(sha(row["path"]) == row["sha256"] for row in graphs.values()), "Synthetic graph changed")
    summary = {}
    for backend in calls:
        summary[backend] = {}
        for field in ("wall_ms_per_call", "calling_thread_cpu_ms_per_call"):
            values = [row[field] for row in timings if row["backend"] == backend]
            summary[backend][field] = {"minimum": min(values), "median": float(np.median(values)), "maximum": max(values)}
    result = {"schema": "latency58-local-mask-mixer-onnx-result-v1", "status": "pass",
              "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"],
              "source_bindings_unchanged": True, "runtime_versions": versions, "version": VERSION,
              "fixture_state_sha256": fingerprints, "graphs": graphs, "parity": parity,
              "timings": timings, "timing_summary": summary, "elapsed_seconds": time.monotonic() - began,
              "counted_bytes_before": counted_before, "counted_bytes_after": require_space(plan, 0),
              "cuda_initialized": False, "training_updates_executed": 0, "validation_material_used": False,
              "separation_checkpoint_loaded": False, "quality_selected": False,
              "limitations": ["Standalone untrained mixer graphs and synthetic logits; no separation quality evidence.",
                              "CPUExecutionProvider and Python session call boundaries; full separation graph not measured.",
                              "Timing is concurrent-host Linux diagnostics, not M4 runtime or callback deadline evidence.",
                              "CPU time measures the calling thread and omits descheduled wall time; neither is a real-time guarantee.",
                              "Synthetic fixture coefficients do not qualify a future trained checkpoint."]}
    write(out / "result.json", result)
    print({"status": "pass", "parity_cases": len(parity), "timing_summary": summary,
           "maximum_absolute_parity_error": max(r["max_absolute_error"] for r in parity)}, flush=True)


if __name__ == "__main__":
    main()
