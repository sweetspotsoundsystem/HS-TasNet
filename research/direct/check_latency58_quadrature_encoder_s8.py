"""Check the reduced signed encoder inside the complete saved streaming graph."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, write, sha, require


def main():
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from research.direct.latency58_quadrature_checkpoint import load_model
    from research.direct.latency58_asymmetric_onnx import INPUT_NAMES, OUTPUT_NAMES, TOLERANCES, verification_cases
    from research.direct.latency58_quadrature_encoder_s8 import build, make_reference, BASELINE_SHA
    from research.direct.check_latency58_fused_gru import session_for
    from research.direct.train_latency58 import state_sha256, verify_inputs
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
            and ort.__version__ == "1.26.0", "Require CPU1 and shipping runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    training_root = PHASE / "fusion-refinement-001"
    source_path = training_root / "plan.json"
    source = read(source_path)
    production_execution = read(training_root / "production-stage/execution.json")
    audit = read(training_root / "checkpoint-audit.json")
    require(production_execution["actual_exit_code"] == 0 and production_execution["source_bindings_unchanged"]
            and audit["status"] == "pass" and audit["saved_optimizer_tensor_count"] == 26,
            "The pending checkpoint must be saved and audited before releasing its storage reservation")
    require_space(source, 5_000_000)
    kernel_root = PHASE / "encoder-s8-reduced-kernel-screen-001"
    kernel = read(kernel_root / "result.json")
    kernel_stage = PHASE / "encoder-s8-reduced-kernel-screen-stage-001/execution.json"
    execution = read(kernel_stage)
    require(kernel["status"] == "pass" and kernel["s8_over_u8_ratio"] < 1
            and kernel["numerical"]["u8s8_symmetric_reduced"]["native_reference_max_abs"] == 0
            and execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"],
            "Require the completed exact and faster isolated-kernel screen")
    kernel_plan_path = kernel_root / "plan.json"
    kernel_plan = read(kernel_plan_path)
    verify_inputs(kernel_plan)
    checkpoint = kernel_plan["source_checkpoint"]
    native, _ = load_model(checkpoint)
    fingerprint = state_sha256(native.state_dict())
    original_path = PHASE / "m4-quadrature-magint8-saved-001/model.onnx"
    require(sha(original_path) == BASELINE_SHA, "The saved original graph changed")
    original = onnx.load(original_path)
    original_proof_path = PHASE / "m4-quadrature-magint8-screen-001/graph.json"
    original_proof = read(original_proof_path)["graph_conversion"]
    music_path = Path("/home/axel/HS-TasNet/data/musdb18hq/train/Actions - One Minute Smile/mixture.wav")
    paths = [Path(__file__).resolve(), ROOT / "research/direct/latency58_quadrature_encoder_s8.py", source_path,
             training_root / "production-stage/execution.json", training_root / "checkpoint-audit.json",
             kernel_root / "result.json", kernel_stage, kernel_plan_path, original_path, original_proof_path, music_path]
    paths.extend(ROOT / "research/direct" / name for name in ("latency58_quadrature_magint8.py",
        "latency58_quadrature_int8.py", "latency58_int8_reference.py", "latency58_int8_precise_float.py",
        "latency58_asymmetric_onnx.py", "check_latency58_fused_gru.py"))
    bindings = {**kernel_plan["source_bindings"], **{str(p): sha(p) for p in paths}}
    verify_inputs({"source_bindings": bindings})
    out = PHASE / "quadrature-encoder-s8-screen-001"
    require(not out.exists(), "Preserve streaming checks")
    out.mkdir()
    write(out / "plan.json", {"source_bindings": bindings, "checkpoint": checkpoint,
          "baseline_graph_sha256": BASELINE_SHA, "graph_saved": False, "tolerances": TOLERANCES,
          "oracle": "Independent reduced signed encoder and original nine U8U8 CPU integer projections",
          "ort_version": ort.__version__, "verification_hops": 1024, "repetitions": 2,
          "native_host_qualified": False, "timing_performed": False,
          "scope": "Complete streaming reference parity only; quality and quiet native timing remain required"})
    graph, conversion = build(native, original)
    require(conversion["s8_weights_sha256"] == kernel["weight_proof"]["s8_weights_sha256"]
            and conversion["s8_scales_sha256"] == kernel["weight_proof"]["s8_scales_sha256"],
            "Whole-graph encoder differs from the checked native kernel")
    candidate_bytes = graph.SerializeToString()
    candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()
    reference, independent = make_reference(native, original, graph, original_proof)
    write(out / "graph.json", {"graph_sha256": candidate_sha, "graph_bytes": len(candidate_bytes),
          "conversion": conversion, "independent_integer_projections": independent})
    require(len(candidate_bytes) < 35_000_000, "Prospective graph exceeded save allowance")
    del graph, original
    session = session_for(candidate_bytes)
    cases, began = [], time.monotonic()
    with torch.inference_mode():
        for case in verification_cases(1024, [music_path]):
            count = case["audio"].shape[-1]
            padded = np.pad(case["audio"], ((0, 0), (0, (-count) % 128 + 128)))
            repetitions = []
            for repeat in range(2):
                states = [value.copy() for value in case["initial_states"]]
                own_states = [torch.from_numpy(value.copy()) for value in states]
                previous = states[0][0, :, -128:].copy()
                maximum = np.zeros(5)
                rms, closure = 0., 0.
                digests = [hashlib.sha256(), hashlib.sha256()]
                for offset in range(0, padded.shape[-1], 128):
                    chunk = np.ascontiguousarray(padded[None, :, offset:offset + 128])
                    expected = [v.numpy() for v in reference(torch.from_numpy(chunk), *own_states)]
                    actual = session.run(list(OUTPUT_NAMES), dict(zip(INPUT_NAMES, [chunk, *states], strict=True)))
                    require(all(np.isfinite(v).all() for v in [*actual, *expected]), "Nonfinite result")
                    for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
                        factor = 2**18 if index == 2 else 1
                        maximum[index] = max(maximum[index], float(np.abs(left - right).max()) * factor)
                        digests[0].update(left.tobytes())
                        digests[1].update(right.tobytes())
                    difference = actual[0].astype(np.float64) - expected[0]
                    rms = max(rms, float(np.sqrt(np.mean(difference**2, axis=-1)).max()))
                    closure = max(closure, *(float(np.abs(v[0].sum(axis=0) - previous).max())
                                              for v in (actual[0], expected[0])))
                    states, own_states = actual[1:], [torch.from_numpy(v) for v in expected[1:]]
                    previous = chunk[0].copy()
                repetitions.append({"maximum_errors_in_physical_units": dict(zip(OUTPUT_NAMES, maximum.tolist(), strict=True)),
                      "maximum_stem_callback_rms": rms, "maximum_closure": closure,
                      "reference_trajectory_sha256": digests[0].hexdigest(), "runtime_trajectory_sha256": digests[1].hexdigest(),
                      "passed": bool(maximum[0] <= TOLERANCES["waveform_max_abs"] and
                         maximum[1:].max() <= TOLERANCES["state_max_abs_decoded_units"] and
                         rms <= TOLERANCES["stem_callback_rms"] and closure <= TOLERANCES["reconstruction_max_abs"])})
            replay = all(repetitions[0][k] == repetitions[1][k] for k in
                         ("reference_trajectory_sha256", "runtime_trajectory_sha256"))
            row = {"case": case["name"], "calls": padded.shape[-1] // 128, "repetitions": repetitions,
                   "reset_replay_bit_exact": replay, "passed": replay and all(v["passed"] for v in repetitions)}
            cases.append(row)
            write(out / f"case-{len(cases)}.json", row)
            print(json.dumps({"case": case["name"], "passed": row["passed"], **repetitions[0]}), flush=True)
    verify_inputs({"source_bindings": bindings})
    require(not torch.cuda.is_initialized() and state_sha256(native.state_dict()) == fingerprint,
            "Source model or CPU scope changed")
    result = {"status": "diagnostic_complete", "strict_parity_passed": all(row["passed"] for row in cases),
          "cases": cases, "source_bindings_unchanged": True, "graph_sha256": candidate_sha,
          "graph_saved": False, "checkpoint": checkpoint, "graph_bytes": len(candidate_bytes),
          "native_host_qualified": False, "plugin_modified": False, "tolerances_changed": False,
          "timing_performed": False, "quality_measured": False,
          "elapsed_seconds": time.monotonic() - began, "counted_bytes_after": require_space(source, 0)}
    write(out / "result.json", result)
    require(result["strict_parity_passed"], "Signed encoder streaming reference parity failed")
    print(json.dumps({"status": result["status"], "strict_parity_passed": result["strict_parity_passed"]}), flush=True)


if __name__ == "__main__":
    main()
