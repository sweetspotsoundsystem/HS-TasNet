"""Measure long precise-integer full-magnitude trajectories at unchanged qualification tolerances."""
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
    import soundfile as sf
    import torch
    from research.direct.latency58_full_magnitude_checkpoint import load_model
    from research.direct.latency58_asymmetric_onnx import INPUT_NAMES, OUTPUT_NAMES, _initial_states, verification_cases, TOLERANCES
    from research.direct.latency58_magnitude_int8 import build, make_reference
    from research.direct.check_latency58_fused_gru import session_for
    from research.direct.train_latency58 import state_sha256, verify_inputs
    from research.direct.latency58_pending_training_space import require_cpu_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "" and ort.__version__ == "1.26.0",
            "Require CPU and shipping runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    source_path = PHASE / "quadrature-001/plan.json"
    source = read(source_path)
    require_cpu_space(source, 5_000_000)
    screen_path = PHASE / "m4-remix-int8-screen-001/result.json"
    screen = read(screen_path)
    screen_execution_path = PHASE / "m4-remix-int8-screen-stage-001/execution.json"
    screen_execution = read(screen_execution_path)
    require(screen["strict_parity_passed"] and screen["float_export_parity_passed"]
            and screen["source_bindings_unchanged"] and screen_execution["actual_exit_code"] == 0
            and screen_execution["source_bindings_unchanged"] and not screen_execution["timed_out"],
            "Short independent screen incomplete")
    screen_plan = read(screen_path.parent / "plan.json")
    verify_inputs(screen_plan)
    checkpoint = screen["checkpoint"]
    native, _ = load_model(checkpoint)
    fingerprint = state_sha256(native.state_dict())
    float_graph, integer_graph, graph, proof = build(native)
    reference, _ = make_reference(native, integer_graph, proof)
    candidate_bytes = graph.SerializeToString()
    require(hashlib.sha256(candidate_bytes).hexdigest() == screen["graph_sha256"], "Rebuilt graph differs from screen")
    del float_graph, integer_graph, graph
    session = session_for(candidate_bytes)
    music_path = Path("/home/axel/HS-TasNet/data/musdb18hq/train/Actions - One Minute Smile/mixture.wav")
    music, rate = sf.read(music_path, frames=30 * 44100 + 37, always_2d=True, dtype="float32")
    require(rate == 44100 and music.shape == (30 * 44100 + 37, 2), "Music excerpt differs")
    cases = [next(verification_cases(1024, [])),
             {"name": "recorded_music_30_seconds", "audio": np.ascontiguousarray(music.T), "initial_states": _initial_states()}]
    out = PHASE / "m4-remix-int8-long-001"
    require(not out.exists(), "Preserve long diagnostics")
    out.mkdir()
    paths = [ROOT / "research/direct/latency58_pending_training_space.py", source_path, Path(__file__).resolve(), ROOT / "research/direct/latency58_magnitude_int8.py",
             screen_path, screen_path.parent / "plan.json", screen_execution_path, Path(checkpoint["path"]), music_path]
    bindings = {**screen_plan["source_bindings"], **{str(path): sha(path) for path in paths}}
    write(out / "plan.json", {"source_bindings": bindings, "checkpoint": checkpoint,
          "graph_sha256": screen["graph_sha256"], "graph_saved": False, "ort_version": ort.__version__,
          "tolerances": TOLERANCES, "pending_training_reservation_bytes": 380000000,
          "scope": "Independent long trajectory diagnostic; unchanged strict tolerances", "audio_written": False})
    rows, began = [], time.monotonic()
    with torch.inference_mode():
        for case in cases:
            count = case["audio"].shape[-1]
            padded = np.pad(case["audio"], ((0, 0), (0, (-count) % 128 + 128)))
            states = [value.copy() for value in case["initial_states"]]
            torch_states = [torch.from_numpy(value.copy()) for value in states]
            previous = states[0][0, :, -128:].copy()
            maximum = np.zeros(5)
            maximum_rms, closure = 0., 0.
            energy, error = np.zeros(4), np.zeros(4)
            digest = hashlib.sha256()
            for hop, offset in enumerate(range(0, padded.shape[-1], 128), start=1):
                chunk = np.ascontiguousarray(padded[None, :, offset:offset + 128])
                expected = [value.numpy() for value in reference(torch.from_numpy(chunk), *torch_states)]
                actual = session.run(list(OUTPUT_NAMES), dict(zip(INPUT_NAMES, [chunk, *states], strict=True)))
                require(all(np.isfinite(value).all() for value in [*actual, *expected]), "Nonfinite output/state")
                for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
                    maximum[index] = max(maximum[index], float(np.abs(left - right).max()) * (2**18 if index == 2 else 1))
                difference = actual[0].astype(np.float64) - expected[0]
                maximum_rms = max(maximum_rms, float(np.sqrt(np.mean(difference ** 2, axis=-1)).max()))
                energy += np.square(expected[0].astype(np.float64)).sum(axis=(0, 2, 3))
                error += np.square(difference).sum(axis=(0, 2, 3))
                closure = max(closure, *(float(np.abs(value[0].sum(axis=0) - previous).max()) for value in (expected[0], actual[0])))
                digest.update(actual[0].tobytes())
                states, torch_states = actual[1:], [torch.from_numpy(value) for value in expected[1:]]
                previous = chunk[0].copy()
                if hop % 1000 == 0:
                    print(json.dumps({"case": case["name"], "hops": hop, "maximum_waveform_error": maximum[0],
                                      "maximum_physical_hidden_error": maximum[2]}), flush=True)
            row = {"case": case["name"], "samples": count, "calls": padded.shape[-1] // 128,
                   "maximum_errors_in_physical_units": dict(zip(OUTPUT_NAMES, maximum.tolist(), strict=True)),
                   "maximum_callback_stem_rms": maximum_rms, "maximum_closure": closure,
                   "per_stem_reference_to_difference_snr_db": (10 * np.log10(np.maximum(energy, 1e-30) / np.maximum(error, 1e-30))).tolist(),
                   "runtime_audio_trajectory_sha256": digest.hexdigest(),
                   "existing_strict_tolerances_passed": bool(maximum[0] <= TOLERANCES["waveform_max_abs"] and
                     maximum[1:].max() <= TOLERANCES["state_max_abs_decoded_units"] and maximum_rms <= TOLERANCES["stem_callback_rms"]
                     and closure <= TOLERANCES["reconstruction_max_abs"])}
            rows.append(row)
            write(out / f"case-{len(rows)}.json", row)
            print(json.dumps(row), flush=True)
    require(state_sha256(native.state_dict()) == fingerprint and not torch.cuda.is_initialized(), "Source or CPU scope changed")
    verify_inputs({"source_bindings": bindings})
    write(out / "result.json", {"status": "diagnostic_complete", "cases": rows,
          "source_bindings_unchanged": True, "native_host_qualified": False, "plugin_modified": False,
          "elapsed_seconds": time.monotonic() - began, "counted_bytes_after": require_cpu_space(source, 0)})


if __name__ == "__main__":
    main()
