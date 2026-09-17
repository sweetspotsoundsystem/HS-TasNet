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


def main():
    import numpy as np
    import onnx
    import onnxruntime as ort
    import soundfile as sf
    import torch
    from research.direct.check_latency58_best_onnx_memory import selected_endpoint
    from research.direct.latency58_best_onnx_export import build
    from research.direct.latency58_attention_onnx_spectral_precision import transform
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
             ROOT / "research/direct/check_latency58_fused_gru.py",
             ROOT / "research/direct/diagnose_latency58_attention_onnx_drift.py",
             PHASE / "attention-onnx-drift-diagnostic-stage-001/execution.json",
             ROOT / "research/direct/diagnose_latency58_attention_onnx_drift_v2.py",
             PHASE / "attention-onnx-drift-diagnostic-stage-002/execution.json"]
    paths.extend((ROOT / "research/direct/latency58_attention_onnx_spectral_precision.py",
                  ROOT / "research/direct/diagnose_latency58_attention_onnx_drift_v3.py",
                  PHASE / "attention-onnx-drift-diagnostic-stage-003/execution.json"))
    paths.extend((ROOT / "research/direct/diagnose_latency58_attention_onnx_drift_v4.py",
                  ROOT / "research/direct/latency58_attention_onnx_arithmetic.py",
                  PHASE / "attention-onnx-drift-diagnostic-stage-004/execution.json"))
    bindings = {**read(short_root / "plan.json")["source_bindings"], **{str(p): sha(p) for p in paths}}
    verify_inputs({"source_bindings": bindings})
    out = PHASE / "attention-onnx-drift-diagnostic-005"
    require(not out.exists(), "Preserve diagnostics")
    out.mkdir()
    write(out / "plan.json", {"source_bindings": bindings, "checkpoint": checkpoint,
          "tolerances": TOLERANCES, "samples": audio.shape[-1], "native_cache_bytes": cache.nbytes,
          "variants": ["precise_analysis_dft", "precise_magnitude_log1p", "precise_dft_and_log1p"],
          "scope": "Bounded numerical diagnostic; no graph saved, no quality or performance claims"})
    # Validate both graph transforms before spending time on the native cache.
    for label in ("precise_analysis_dft", "precise_magnitude_log1p", "precise_dft_and_log1p"):
        probe = copy.deepcopy(original)
        transform(probe, label)
        onnx.checker.check_model(probe, full_check=True)
        preflight_options = ort.SessionOptions()
        preflight_options.intra_op_num_threads = preflight_options.inter_op_num_threads = 1
        preflight_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        preflight_session = ort.InferenceSession(probe.SerializeToString(), sess_options=preflight_options, providers=["CPUExecutionProvider"])
        del preflight_session, preflight_options, probe
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
    for label in ("precise_analysis_dft", "precise_magnitude_log1p", "precise_dft_and_log1p"):
        graph = copy.deepcopy(original)
        proof = transform(graph, label)
        onnx.checker.check_model(graph, full_check=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
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
        maximum_hops = [-1] * len(sizes)
        injected_one_step_maximum = np.zeros(len(sizes))
        rms, closure = 0., 0.
        for hop in range(calls):
            chunk = np.ascontiguousarray(padded[None, :, hop * 128:(hop + 1) * 128])
            actual = session.run(list(contract["output_names"]), dict(zip(contract["input_names"], [chunk, *states], strict=True)))
            require(all(np.isfinite(value).all() for value in actual), "Nonfinite variant output")
            for index, value in enumerate(actual):
                difference = value.reshape(-1).astype(np.float64) - cache[hop, offsets[index]:offsets[index + 1]]
                error = np.abs(difference).max() * (2**18 if index == 2 else 1)
                if error > maximum[index]:
                    maximum[index], maximum_hops[index] = error, hop
                if index == 0:
                    rms = max(rms, float(np.sqrt(np.mean(difference.reshape(1, 4, 2, 128) ** 2, axis=(2, 3))).max()))
            expected_mix = np.zeros((1, 2, 128), np.float32) if hop == 0 else padded[None, :, (hop - 1) * 128:hop * 128]
            closure = max(closure, float(np.abs(actual[0].sum(1) - expected_mix).max()))
            states = actual[1:]
            if hop % 128 == 0:
                injected = ([np.zeros(shape, np.float32) for shape in contract["state_shapes"]] if hop == 0 else
                            [cache[hop - 1, offsets[i]:offsets[i + 1]].reshape(contract["output_shapes"][i]).copy()
                             for i in range(1, len(sizes))])
                local = session.run(list(contract["output_names"]), dict(zip(contract["input_names"], [chunk, *injected], strict=True)))
                for i, value in enumerate(local):
                    error = np.abs(value.reshape(-1).astype(np.float64) - cache[hop, offsets[i]:offsets[i + 1]]).max()
                    injected_one_step_maximum[i] = max(injected_one_step_maximum[i], error * (2**18 if i == 2 else 1))
        passed = bool(maximum[0] <= TOLERANCES["waveform_max_abs"] and maximum[1:].max() <= TOLERANCES["state_max_abs_decoded_units"]
                      and rms <= TOLERANCES["stem_callback_rms"] and closure <= TOLERANCES["reconstruction_max_abs"])
        row = {"variant": label, "graph_sha256": hashlib.sha256(data).hexdigest(), "proof": proof,
               "maximum_errors_in_physical_units": dict(zip(contract["output_names"], maximum.tolist(), strict=True)),
               "maximum_error_hops": dict(zip(contract["output_names"], maximum_hops, strict=True)),
               "native_state_injected_one_step_maximum_every_128_hops": dict(zip(contract["output_names"], injected_one_step_maximum.tolist(), strict=True)),
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
