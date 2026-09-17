"""Screen fixed hidden-state quantization without changing parity thresholds."""
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
    from research.direct.latency58_asymmetric import Latency58AsymmetricModel
    from research.direct.latency58_residual_model import load_checkpoint, BASE_STATE
    from research.direct.latency58_asymmetric_onnx import INPUT_NAMES, OUTPUT_NAMES, TOLERANCES, verification_cases
    from research.direct.latency58_int8_fixed_hidden import convert, make_reference, PREFIXES, SCALE, ZERO
    from research.direct.check_latency58_fused_gru import session_for, benchmark
    from research.direct.train_latency58 import state_sha256, verify_inputs
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
            and ort.__version__ == "1.26.0", "Require CPU1 and shipping runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    source_path = PHASE / "full-magnitude-fast16-001/plan.json"
    source = read(source_path)
    require_space(source, 375_000_000)
    binding = read(PHASE / "m4-int8-screen-002/result.json")["quantized"]
    require(sha(binding["path"]) == binding["sha256"], "Original integer graph changed")
    graph = convert(onnx.load(binding["path"], load_external_data=False))
    onnx.checker.check_model(graph, full_check=True)
    candidate_bytes = graph.SerializeToString()
    candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()
    parent_binding = read(PHASE / "full-magnitude-001/plan.json")["parent_checkpoint"]
    parent, _ = load_checkpoint(parent_binding["path"], parent_binding["sha256"])
    native = Latency58AsymmetricModel()
    native.load_state_dict({k: v for k, v in parent.state_dict().items() if k != "fixed_residual_share"}, strict=True)
    native.eval().requires_grad_(False)
    require(state_sha256(native.state_dict()) == BASE_STATE, "Wrong FP32 source")
    reference, _ = make_reference(native, graph)
    session = session_for(candidate_bytes)
    music_path = Path("/home/axel/HS-TasNet/data/musdb18hq/train/Actions - One Minute Smile/mixture.wav")
    out = PHASE / "m4-int8-fixed-hidden-001"
    require(not out.exists(), "Preserve earlier screens")
    out.mkdir()
    paths = [source_path, Path(__file__).resolve(), ROOT / "research/direct/latency58_int8_fixed_hidden.py",
             ROOT / "research/direct/latency58_int8_reference.py", Path(binding["path"]), music_path]
    bindings = {**source["source_bindings"], **{str(p): sha(p) for p in paths}}
    write(out / "plan.json", {"source_bindings": bindings, "parent_graph": binding, "graph_sha256": candidate_sha,
          "graph_saved": False, "bounded_projections": PREFIXES, "scale": float(SCALE), "zero": int(ZERO),
          "hidden_bound_basis": "From reset: sigmoid gates in [0,1], tanh candidates in [-1,1], convex hidden updates",
          "tolerances": TOLERANCES, "oracle": "Independent CPU PyTorch integer arithmetic", "ort_version": ort.__version__})
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
    # These are exploratory Linux timings under training load, not M4 qualification.
    audio = next(verification_cases(600, []))["audio"]
    measurements = []
    original_bytes = Path(binding["path"]).read_bytes()
    for cycle, order in enumerate((("dynamic", "fixed"), ("fixed", "dynamic"),
                                   ("fixed", "dynamic"), ("dynamic", "fixed"))):
        for name in order:
            timing = benchmark(session_for(original_bytes if name == "dynamic" else candidate_bytes), audio, 64, 512)
            measurements.append({"cycle": cycle, "variant": name, "timing": timing})
    verify_inputs({"source_bindings": bindings})
    require(not torch.cuda.is_initialized() and state_sha256(native.state_dict()) == BASE_STATE, "Source or CPU scope changed")
    result = {"status": "diagnostic_complete", "strict_parity_passed": all(row["passed"] for row in cases),
          "cases": cases, "timings": measurements, "source_bindings_unchanged": True, "graph_sha256": candidate_sha,
          "graph_saved": False, "native_host_qualified": False, "plugin_modified": False, "tolerances_changed": False,
          "elapsed_seconds": time.monotonic() - began, "counted_bytes_after": require_space(source, 370_000_000)}
    write(out / "result.json", result)
    print(json.dumps({"status": result["status"], "strict_parity_passed": result["strict_parity_passed"]}), flush=True)


if __name__ == "__main__":
    main()
