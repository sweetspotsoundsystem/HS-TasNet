"""Screen a tenth integer projection against the checked nine-projection quadrature graph."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import statistics

from research.direct.run_latency58_quality import ROOT, PHASE, read, write, sha, require


def main():
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from research.direct.latency58_quadrature_checkpoint import load_model
    from research.direct.latency58_asymmetric_onnx import INPUT_NAMES, OUTPUT_NAMES, TOLERANCES, verification_cases
    from research.direct.latency58_quadrature_magint8 import build, make_reference
    from research.direct.check_latency58_fused_gru import session_for, benchmark
    from research.direct.train_latency58 import state_sha256, verify_inputs
    from research.direct.latency58_pending_training_space import require_cpu_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
            and ort.__version__ == "1.26.0", "Require CPU1 and shipping runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    source_path = PHASE / "quadrature-continuation-001/plan.json"
    source = read(source_path)
    verify_inputs(source)
    require_cpu_space(source, 5_000_000)
    checkpoint = read(PHASE / "quadrature-001/result.json")["checkpoint"]
    native, _ = load_model(checkpoint)
    fingerprint = state_sha256(native.state_dict())
    require(checkpoint == source["parent_checkpoint"] and fingerprint == source["parent_model_state_sha256"],
            "Integer screen must use the reviewed 4.2277 dB saved quadrature parent")
    music_path = Path("/home/axel/HS-TasNet/data/musdb18hq/train/Actions - One Minute Smile/mixture.wav")
    baseline_root = PHASE / "m4-quadrature-int8-screen-001"
    baseline_screen = read(baseline_root / "result.json")
    baseline_long = read(PHASE / "m4-quadrature-int8-long-001/result.json")
    prerequisite_paths = [baseline_root / name for name in ("plan.json", "result.json", "graph.json", "float-export-parity.json")]
    prerequisite_paths.extend(PHASE / name / "execution.json" for name in
                              ("m4-quadrature-int8-screen-stage-001", "m4-quadrature-int8-long-stage-001"))
    prerequisite_paths.extend(PHASE / "m4-quadrature-int8-long-001" / name for name in ("plan.json", "result.json"))
    require(baseline_screen["strict_parity_passed"] and baseline_screen["float_export_parity_passed"]
            and baseline_screen["checkpoint"] == checkpoint and baseline_long["source_bindings_unchanged"]
            and all(row["existing_strict_tolerances_passed"] for row in baseline_long["cases"]),
            "The nine-projection baseline must pass both independent checks")
    for path in prerequisite_paths:
        if path.name == "execution.json":
            execution = read(path)
            require(execution["actual_exit_code"] == 0 and not execution["timed_out"]
                    and execution["source_bindings_unchanged"], "Baseline check did not close successfully")
        if path.name == "plan.json":
            verify_inputs(read(path))
    out = PHASE / "m4-quadrature-magint8-screen-001"
    require(not out.exists(), "Preserve earlier screens")
    out.mkdir()
    module = Path(ort.__file__).resolve()
    paths = [ROOT / "research/direct/latency58_pending_training_space.py", source_path, Path(__file__).resolve(), ROOT / "research/direct/latency58_quadrature_int8.py",
             ROOT / "research/direct/latency58_int8_precise_core.py", ROOT / "research/direct/latency58_int8_precise_float.py",
             ROOT / "research/direct/latency58_int8_reference.py", ROOT / "research/direct/latency58_quadrature_onnx.py",
             ROOT / "research/direct/latency58_conv_gemm.py", ROOT / "research/direct/check_latency58_fused_gru.py",
             Path(checkpoint["path"]), music_path, module,
             *sorted((module.parent / "quantization").glob("*.py")), *sorted((module.parent / "capi").glob("*.so*"))]
    paths.extend([*prerequisite_paths, ROOT / "research/direct/latency58_quadrature_magint8.py",
                  ROOT / "research/direct/latency58_int8_precise_ten.py"])
    bindings = {**source["source_bindings"], **{str(p): sha(p) for p in paths}}
    write(out / "plan.json", {"source_bindings": bindings, "checkpoint": checkpoint,
          "graph_saved": False, "tolerances": TOLERANCES, "oracle": "Independent CPU PyTorch integer arithmetic",
          "ort_version": ort.__version__, "pending_training_reservation_bytes": 380000000,
          "magnitude_projection_quantized": True, "phase_factors_kept_fp32": True,
          "native_host_qualified": False, "timing_performed": True,
          "exploratory_speed_gate": {"maximum_ten_over_nine_median_ratio": 0.97, "minimum_faster_cycles": 3,
             "cycles": 4, "orders": [["nine", "ten"], ["ten", "nine"], ["ten", "nine"], ["nine", "ten"]],
             "warmup_hops": 64, "measured_hops": 512},
          "timing_scope": "Linux Python ORT CPU1 under concurrent GPU training; exploratory, not native or M4 qualification"})
    import shutil
    for name in ("latency58_quadrature_int8.py", "latency58_quadrature_magint8.py",
                 "latency58_int8_precise_ten.py", Path(__file__).name):
        shutil.copyfile(ROOT / "research/direct" / name, out / name)
    float_graph, integer_graph, graph, graph_proof, baseline_bytes = build(native, return_baseline=True)
    candidate_bytes = graph.SerializeToString()
    original_bytes = float_graph.SerializeToString()
    candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()
    reference, proof = make_reference(native, integer_graph, graph_proof)
    write(out / "graph.json", {"graph_sha256": candidate_sha, "graph_bytes": len(candidate_bytes),
          "float_graph_sha256": hashlib.sha256(original_bytes).hexdigest(),
          "integer_weight_proofs": proof, "graph_conversion": graph_proof})
    require(len(candidate_bytes) < 35000000, "Integer graph exceeded prospective save reservation")
    require(hashlib.sha256(baseline_bytes).hexdigest() == baseline_screen["graph_sha256"]
            == graph_proof["baseline_nine_projection_graph_sha256"], "Nine-projection comparison graph changed")
    require(hashlib.sha256(original_bytes).hexdigest() == read(baseline_root / "graph.json")["float_graph_sha256"],
            "Floating graph differs from the completed native/export/ORT check")
    prior_float = read(baseline_root / "float-export-parity.json")
    require(prior_float["passed"], "Existing identical floating graph has not passed")
    write(out / "float-export-parity.json", {"passed": True,
          "reused_identical_float_graph": True, "float_graph_sha256": hashlib.sha256(original_bytes).hexdigest(),
          "source": {"path": str(baseline_root / "float-export-parity.json"), "sha256": sha(baseline_root / "float-export-parity.json")}})
    del float_graph, integer_graph, graph
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
    measurements = []
    audio = next(verification_cases(600, []))["audio"]
    for cycle, order in enumerate((("nine", "ten"), ("ten", "nine"), ("ten", "nine"), ("nine", "ten"))):
        for name in order:
            timing = benchmark(session_for(baseline_bytes if name == "nine" else candidate_bytes), audio, 64, 512)
            measurements.append({"cycle": cycle, "variant": name, "timing": timing})
    medians = {name: statistics.median(row["timing"]["p50_ms"] for row in measurements if row["variant"] == name)
               for name in ("nine", "ten")}
    faster = sum(next(row["timing"]["p50_ms"] for row in measurements if row["cycle"] == cycle and row["variant"] == "ten")
                 < next(row["timing"]["p50_ms"] for row in measurements if row["cycle"] == cycle and row["variant"] == "nine")
                 for cycle in range(4))
    ratio = medians["ten"] / medians["nine"]
    speed_gate = ratio <= .97 and faster >= 3
    verify_inputs({"source_bindings": bindings})
    require(not torch.cuda.is_initialized() and state_sha256(native.state_dict()) == fingerprint, "Source or CPU scope changed")
    result = {"status": "diagnostic_complete", "strict_parity_passed": all(row["passed"] for row in cases),
          "cases": cases, "timings": measurements, "timing_performed": True,
          "median_of_cycle_p50_ms": medians, "ten_over_nine_median_ratio": ratio, "faster_cycles": faster,
          "exploratory_speed_gate_passed": speed_gate,
          "advance_to_long_check": speed_gate and all(row["passed"] for row in cases), "source_bindings_unchanged": True, "graph_sha256": candidate_sha,
          "graph_saved": False, "checkpoint": checkpoint, "graph_bytes": len(candidate_bytes),
          "float_export_parity_passed": True, "native_host_qualified": False, "plugin_modified": False, "tolerances_changed": False,
          "elapsed_seconds": time.monotonic() - began, "counted_bytes_after": require_cpu_space(source, 0)}
    write(out / "result.json", result)
    print(json.dumps({"status": result["status"], "strict_parity_passed": result["strict_parity_passed"]}), flush=True)


if __name__ == "__main__":
    main()
