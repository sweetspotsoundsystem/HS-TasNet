"""Check a combined rank-128/integer graph before any validation scoring."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, write, sha, require


def check_case(reference, session, case):
    import numpy as np
    import torch
    from research.direct.latency58_asymmetric_onnx import INPUT_NAMES, OUTPUT_NAMES, TOLERANCES, _validate_values
    audio = case["audio"]
    padded = np.pad(audio, ((0, 0), (0, (-audio.shape[-1]) % 128 + 128)))
    repeats = []
    with torch.inference_mode():
        for _ in range(2):
            states = [v.copy() for v in case["initial_states"]]
            own_states = [torch.from_numpy(v.copy()) for v in states]
            history = states[0].copy()
            maximum, rms, closure = np.zeros(5), 0., 0.
            digests = [hashlib.sha256(), hashlib.sha256()]
            for offset in range(0, padded.shape[-1], 128):
                chunk = np.ascontiguousarray(padded[None, :, offset:offset + 128])
                expected_mixture = history[..., -128:].copy()
                history = np.concatenate((history[..., 128:], chunk), axis=-1)
                expected = [v.numpy() for v in reference(torch.from_numpy(chunk), *own_states)]
                actual = session.run(list(OUTPUT_NAMES), dict(zip(INPUT_NAMES, [chunk, *states], strict=True)))
                for digest, values in zip(digests, (expected, actual), strict=True):
                    _validate_values(values, case["name"])
                    require(np.array_equal(values[1], history), "Physical history shifted incorrectly")
                    for value in values:
                        digest.update(value.tobytes())
                    closure = max(closure, float(np.abs(values[0].sum(1).astype(np.float64) - expected_mixture).max()))
                for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
                    factor = 2**18 if index == 2 else 1
                    maximum[index] = max(maximum[index], float(np.abs(left.astype(np.float64) - right).max()) * factor)
                difference = actual[0].astype(np.float64) - expected[0]
                rms = max(rms, float(np.sqrt(np.mean(difference**2, axis=(2, 3))).max()))
                states, own_states = actual[1:], [torch.from_numpy(v) for v in expected[1:]]
            repeats.append({"maximum_errors_in_physical_units": dict(zip(OUTPUT_NAMES, maximum.tolist(), strict=True)),
                "maximum_stem_callback_rms": rms, "maximum_closure": closure,
                "reference_trajectory_sha256": digests[0].hexdigest(), "runtime_trajectory_sha256": digests[1].hexdigest(),
                "passed": bool(maximum[0] <= TOLERANCES["waveform_max_abs"] and
                    maximum[1:].max() <= TOLERANCES["state_max_abs_decoded_units"] and
                    rms <= TOLERANCES["stem_callback_rms"] and closure <= TOLERANCES["reconstruction_max_abs"])})
    replay = all(repeats[0][key] == repeats[1][key] for key in
                 ("reference_trajectory_sha256", "runtime_trajectory_sha256"))
    return {"case": case["name"], "calls": padded.shape[-1] // 128, "repetitions": repeats,
            "physical_history_bit_exact": True, "reset_replay_bit_exact": replay,
            "passed": replay and all(v["passed"] for v in repeats)}


def main():
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from research.direct.latency58_asymmetric import Latency58AsymmetricModel
    from research.direct.latency58_residual_model import load_checkpoint, BASE_STATE
    from research.direct.latency58_asymmetric_onnx import TOLERANCES, verification_cases
    from research.direct.latency58_rank_int8 import build
    from research.direct.check_latency58_fused_gru import session_for, benchmark
    from research.direct.train_latency58 import state_sha256, verify_inputs
    from research.direct.latency58_sdr_checkpoint import require_space

    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
            and ort.__version__ == "1.26.0", "Require CPU1 and shipping runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    source_path = PHASE / "remix-magnitude-001/plan.json"
    source = read(source_path)
    counted = require_space(source, 410_000_000)
    float_path = Path("/home/axel/autoresearch/codex/stemgen-rt-hop128-5ms/model/model.onnx")
    integer_binding = read(PHASE / "m4-int8-screen-002/result.json")["quantized"]
    baseline = PHASE / "m4-int8-precise-core-full14-001/model.onnx"
    require(sha(float_path) == "b8574ac2e67bcd1df533e3fc7464c0659d4bfa6744cfbb4389c0967271594fe3"
            and sha(integer_binding["path"]) == integer_binding["sha256"]
            and sha(baseline) == "a550c904ef501fe98f3afa010eca5a53abd5bf66d63906e018511a01c5076d61",
            "Source graphs changed")
    parent_binding = read(PHASE / "full-magnitude-001/plan.json")["parent_checkpoint"]
    parent, _ = load_checkpoint(parent_binding["path"], parent_binding["sha256"])
    native = Latency58AsymmetricModel()
    native.load_state_dict({k: v for k, v in parent.state_dict().items() if k != "fixed_residual_share"}, strict=True)
    native.eval().requires_grad_(False)
    require(state_sha256(native.state_dict()) == BASE_STATE, "Wrong C204 source")
    del parent
    out = PHASE / "m4-rank128-int8-001"
    require(not out.exists(), "Preserve previous diagnostics")
    music = Path("/home/axel/HS-TasNet/data/musdb18hq/train/Actions - One Minute Smile/mixture.wav")
    paths = [source_path, float_path, Path(integer_binding["path"]), baseline, music,
             Path(parent_binding["path"]), Path(__file__).resolve()]
    paths.extend(ROOT / "research/direct" / name for name in (
        "latency58_rank_int8.py", "latency58_low_rank_mask.py", "latency58_conv_gemm.py",
        "latency58_int8_precise_core.py", "latency58_int8_precise_float.py", "latency58_int8_reference.py",
        "check_latency58_fused_gru.py", "latency58_asymmetric_onnx.py"))
    bindings = {**source["source_bindings"], **{str(p): sha(p) for p in paths}}
    plan = {"source_bindings": bindings, "source_checkpoint": parent_binding, "tolerances": TOLERANCES,
            "counted_bytes_before": counted, "pending_training_save_reserved_bytes": 370_000_000,
            "temporary_and_candidate_reservation_bytes": 40_000_000, "ort_version": ort.__version__,
            "arithmetic": "Rank128 reduction FP64; nine U8U8 blocks; FP64 quantizer ancestors; FP32 decoding and public states",
            "oracle": "Independent PyTorch integer arithmetic with independently reconstructed quantized weights",
            "validation_used": False, "screen_hops": 1024, "reset_repetitions": 2,
            "timing_rule": "Advance only after strict parity and pooled CPU1 median at least 3 percent below precise C204 integer; local exploratory timing only",
            "graph_delay_samples": 128, "host_queue_samples": 128, "native_host_qualified": False}
    out.mkdir()
    write(out / "plan.json", plan)
    began = time.monotonic()
    graph, reference, proof = build(native, onnx.load(float_path, load_external_data=False),
                                    onnx.load(integer_binding["path"], load_external_data=False))
    graph_bytes = graph.SerializeToString()
    require(len(graph_bytes) < 32_000_000, "Graph exceeds its reservation")
    write(out / "construction.json", {"proof": proof, "graph_sha256": hashlib.sha256(graph_bytes).hexdigest(),
                                     "graph_bytes": len(graph_bytes)})
    runtime = session_for(graph_bytes)
    cases = []
    for case in verification_cases(1024, [music]):
        row = check_case(reference, runtime, case)
        cases.append(row)
        write(out / f"parity-{len(cases)}.json", row)
        print(json.dumps({"event": "parity", "case": row["case"], "passed": row["passed"],
                          **row["repetitions"][0]}), flush=True)
    passed = all(row["passed"] for row in cases)
    measurements, summary, ratio = [], {}, None
    if passed:
        audio = next(verification_cases(600, []))["audio"]
        data = {"precise_int8": baseline.read_bytes(), "rank128_int8": graph_bytes}
        for cycle, order in enumerate((("precise_int8", "rank128_int8"), ("rank128_int8", "precise_int8"),
                                       ("rank128_int8", "precise_int8"), ("precise_int8", "rank128_int8"))):
            for label in order:
                row = {"cycle": cycle, "variant": label, **benchmark(session_for(data[label]), audio, 64, 512)}
                measurements.append(row)
                print(json.dumps({k: row[k] for k in ("cycle", "variant", "p50_ms", "p95_ms")}), flush=True)
        for label in data:
            values = [t for row in measurements if row["variant"] == label for t in row["times_ms"]]
            summary[label] = {f"p{p}_ms": float(np.percentile(values, p)) for p in (50, 95, 99)}
        ratio = summary["rank128_int8"]["p50_ms"] / summary["precise_int8"]["p50_ms"]
    verify_inputs({"source_bindings": bindings})
    require(state_sha256(native.state_dict()) == BASE_STATE and not torch.cuda.is_initialized(), "CPU scope or source changed")
    advance = passed and ratio <= .97
    if advance:
        with (out / "model.onnx").open("xb") as stream:
            stream.write(graph_bytes)
    result = {"status": "screen_complete", "plan_sha256": sha(out / "plan.json"), "strict_parity_passed": passed,
              "cases": cases, "timings": measurements, "timing_summary": summary, "pooled_p50_ratio": ratio,
              "advance_to_long_parity_and_quality": advance, "graph_saved": advance,
              "graph_sha256": hashlib.sha256(graph_bytes).hexdigest(), "graph_bytes": len(graph_bytes),
              "source_bindings_unchanged": True, "native_host_qualified": False, "plugin_modified": False,
              "validation_used": False, "quality_measured": False, "elapsed_seconds": time.monotonic() - began,
              "counted_bytes_after": require_space(source, 370_000_000),
              "limitations": "Linux Python ORT timing under GPU training does not establish native M4 speed. Combined quality and longer integer recurrence still need verification."}
    write(out / "result.json", result)
    print(json.dumps({k: result[k] for k in ("status", "strict_parity_passed", "timing_summary", "pooled_p50_ratio", "advance_to_long_parity_and_quality")}), flush=True)


if __name__ == "__main__":
    main()
