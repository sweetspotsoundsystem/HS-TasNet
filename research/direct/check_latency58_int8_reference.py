"""Compare integer ONNX trajectories with an independent CPU Torch oracle."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, write, sha, require
from research.direct.train_latency58 import state_sha256, verify_inputs


def main():
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from research.direct.latency58 import PUBLIC_FUSION_SCALE
    from research.direct.latency58_asymmetric import Latency58AsymmetricModel
    from research.direct.latency58_residual_model import load_checkpoint, BASE_STATE
    from research.direct.latency58_asymmetric_onnx import verification_cases, INPUT_NAMES, OUTPUT_NAMES, TOLERANCES
    from research.direct.latency58_int8_reference import make_reference
    from research.direct.check_latency58_fused_gru import session_for
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "" and
            all(os.environ.get(key) == "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use CPU1 with CUDA hidden")
    require(ort.__version__ == "1.26.0", "Use the shipping ORT version")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    source_path = PHASE / "full-magnitude-001/plan.json"
    source = read(source_path)
    verify_inputs(source)
    require_space(source, 375_000_000)
    quantized = read(PHASE / "m4-int8-screen-002/result.json")["quantized"]
    require(sha(quantized["path"]) == quantized["sha256"], "Quantized graph changed")
    audio = Path("/home/axel/HS-TasNet/data/musdb18hq/train/Actions - One Minute Smile/mixture.wav")
    paths = [source_path, Path(__file__).resolve(), ROOT / "research/direct/latency58_int8_reference.py", audio,
             Path(quantized["path"]), ROOT / "research/direct/latency58_asymmetric_onnx.py"]
    bindings = {**source["source_bindings"], **{str(path): sha(path) for path in paths}}
    out = PHASE / "m4-int8-torch-reference-003"
    require(not out.exists(), "Preserve oracle checks")
    out.mkdir()
    write(out / "plan.json", {"source_bindings": bindings, "quantized": quantized, "ort_version": ort.__version__,
          "tolerances": TOLERANCES, "hops": 64, "oracle": "CPU PyTorch integer dot products, native FFT and synthesis",
          "runtime_is_not_an_oracle": True, "plugin_modified": False})
    began = time.monotonic()
    parent, _ = load_checkpoint(source["parent_checkpoint"]["path"], source["parent_checkpoint"]["sha256"])
    native = Latency58AsymmetricModel()
    native.load_state_dict({key: value for key, value in parent.state_dict().items() if key != "fixed_residual_share"}, strict=True)
    native.eval().requires_grad_(False)
    require(state_sha256(native.state_dict()) == BASE_STATE, "C204 source state differs")
    reference, weight_proof = make_reference(native, onnx.load(quantized["path"], load_external_data=False))
    write(out / "weight-proof.json", weight_proof)
    session = session_for(Path(quantized["path"]).read_bytes())
    cases = []
    with torch.inference_mode():
        for case in verification_cases(64, [audio]):
            count = case["audio"].shape[-1]
            padded = np.pad(case["audio"], ((0, 0), (0, (-count) % 128 + 128)))
            repetitions = []
            for repeat in range(2):
                states = [value.copy() for value in case["initial_states"]]
                torch_states = [torch.from_numpy(value.copy()) for value in states]
                previous = states[0][0, :, -128:].copy()
                maximum, rms, closure = np.zeros(5), 0., 0.
                runtime_digest, reference_digest = hashlib.sha256(), hashlib.sha256()
                for offset in range(0, padded.shape[-1], 128):
                    chunk = np.ascontiguousarray(padded[None, :, offset:offset + 128])
                    predicted = [value.numpy() for value in reference(torch.from_numpy(chunk), *torch_states)]
                    actual = session.run(list(OUTPUT_NAMES), dict(zip(INPUT_NAMES, [chunk, *states], strict=True)))
                    require(all(value.dtype == np.float32 and np.isfinite(value).all() for value in [*actual, *predicted]),
                            "Nonfinite output or state")
                    for index, (left, right) in enumerate(zip(predicted, actual, strict=True)):
                        scale = 1 / PUBLIC_FUSION_SCALE if index == 2 else 1.
                        maximum[index] = max(maximum[index], float(np.abs(left - right).max()) * scale)
                        runtime_digest.update(right.tobytes())
                        reference_digest.update(left.tobytes())
                    rms = max(rms, float(np.sqrt(np.mean((predicted[0].astype(np.float64) - actual[0]) ** 2, axis=-1)).max()))
                    closure = max(closure, *(float(np.abs(value[0].sum(axis=0) - previous).max())
                                              for value in (predicted[0], actual[0])))
                    states, torch_states = actual[1:], [torch.from_numpy(value) for value in predicted[1:]]
                    previous = chunk[0].copy()
                repetitions.append({"maximum_errors_in_physical_units": dict(zip(OUTPUT_NAMES, maximum.tolist(), strict=True)),
                      "callback_stem_rms": rms, "closure": closure,
                      "runtime_trajectory_sha256": runtime_digest.hexdigest(), "reference_trajectory_sha256": reference_digest.hexdigest(),
                      "passed": bool(maximum[0] <= TOLERANCES["waveform_max_abs"] and
                        maximum[1:].max() <= TOLERANCES["state_max_abs_decoded_units"] and
                        rms <= TOLERANCES["stem_callback_rms"] and closure <= TOLERANCES["reconstruction_max_abs"])})
            replay = all(repetitions[0][key] == repetitions[1][key] for key in
                         ("runtime_trajectory_sha256", "reference_trajectory_sha256"))
            row = {"case": case["name"], "callbacks": padded.shape[-1] // 128, "repetitions": repetitions,
                   "reset_replay_bit_exact": replay, "passed": replay and all(value["passed"] for value in repetitions)}
            cases.append(row)
            write(out / f"case-{len(cases)}.json", row)
            print(json.dumps({"case": case["name"], "passed": row["passed"], **repetitions[0]}), flush=True)
    verify_inputs({"source_bindings": bindings})
    require(state_sha256(native.state_dict()) == BASE_STATE and not torch.cuda.is_initialized(), "Source or CPU scope changed")
    passed = all(case["passed"] for case in cases)
    write(out / "result.json", {"status": "pass" if passed else "rejected_strict_parity", "cases": cases,
          "weight_proof": weight_proof, "source_bindings_unchanged": True, "plugin_modified": False,
          "native_host_qualified": False, "elapsed_seconds": time.monotonic() - began,
          "counted_bytes_after": require_space(source, 370_000_000)})
    require(passed, "Integer oracle did not meet existing strict parity tolerances")


if __name__ == "__main__":
    main()
