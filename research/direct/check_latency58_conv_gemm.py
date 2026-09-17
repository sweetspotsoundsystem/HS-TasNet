"""Check one-frame Gemm rewrites against native Torch, then compare CPU timing."""
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
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CPU1")
    import numpy as np
    import onnx
    import onnxruntime as ort
    import soundfile as sf
    import torch
    from research.direct.latency58_asymmetric import Latency58AsymmetricModel
    from research.direct.latency58_asymmetric_onnx import _run_case, verification_cases, make_export_copy, TOLERANCES
    from research.direct.latency58_residual_model import load_checkpoint, BASE_STATE
    from research.direct.latency58_conv_gemm import convert
    from research.direct.check_latency58_fused_gru import session_for, benchmark
    from research.direct.latency58_sdr_checkpoint import require_space
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model_path = Path("/home/axel/autoresearch/codex/stemgen-rt-hop128-5ms/model/model.onnx")
    require(sha(model_path) == "b8574ac2e67bcd1df533e3fc7464c0659d4bfa6744cfbb4389c0967271594fe3", "Native plugin graph changed")
    source_path = PHASE / "full-magnitude-001/plan.json"
    source = read(source_path)
    verify_inputs(source)
    require_space(source, 370_000_000)
    parent_binding = source["parent_checkpoint"]
    parent, _ = load_checkpoint(parent_binding["path"], parent_binding["sha256"])
    model = Latency58AsymmetricModel()
    model.load_state_dict({k: v for k, v in parent.state_dict().items() if k != "fixed_residual_share"}, strict=True)
    model.eval().requires_grad_(False)
    require(state_sha256(model.state_dict()) == BASE_STATE, "Native C204 weights differ")
    initial_parent_sha = state_sha256(parent.state_dict())
    wrapper = make_export_copy(model)
    graph = onnx.load(model_path, load_external_data=False)
    metadata = {p.key: p.value for p in graph.metadata_props}
    require(metadata["hs_tasnet.model_state_sha256"] == BASE_STATE, "ONNX and Torch weights differ")
    audio_path = Path("/home/axel/HS-TasNet/data/musdb18hq/train/Actions - One Minute Smile/mixture.wav")
    out = PHASE / "m4-conv-gemm-001"
    require(not out.exists(), "Preserve completed screens")
    bindings = {**source["source_bindings"], str(source_path): sha(source_path), str(model_path): sha(model_path),
                str(audio_path): sha(audio_path), str(Path(__file__).resolve()): sha(__file__),
                str(ROOT / "research/direct/latency58_conv_gemm.py"): sha(ROOT / "research/direct/latency58_conv_gemm.py"),
                str(ROOT / "research/direct/check_latency58_fused_gru.py"): sha(ROOT / "research/direct/check_latency58_fused_gru.py")}
    out.mkdir()
    write(out / "plan.json", {"source_bindings": bindings, "reference_model_state_sha256": BASE_STATE,
          "tolerances": TOLERANCES, "cpu_only": True, "native_plugin_unchanged": True,
          "variants": ["original", "encoder_gemm", "both_gemm"], "graph_saved": False})
    began, data, transforms, parity = time.monotonic(), {}, {}, {}
    for label, names in (("original", []), ("encoder_gemm", ["/conv_encode/Conv"]),
                         ("both_gemm", ["/conv_encode/Conv", "/basis_to_embed/Conv"])):
        candidate = copy.deepcopy(graph)
        transforms[label] = convert(candidate, names) if names else []
        onnx.checker.check_model(candidate, full_check=True)
        data[label] = candidate.SerializeToString()
        session = session_for(data[label])
        parity[label] = []
        for case in verification_cases(128, [str(audio_path)]):
            first = _run_case(model, wrapper, session, case)
            repeated = _run_case(model, wrapper, session, case)
            exact = first["all_output_and_state_trajectory_sha256"] == repeated["all_output_and_state_trajectory_sha256"]
            first["reset_replay_bit_exact"] = exact
            first["passed"] = first["passed"] and repeated["passed"] and exact
            parity[label].append(first)
            print(json.dumps({"variant": label, "case": first["input"], "passed": first["passed"]}), flush=True)
        del session, candidate
        write(out / (label + "-parity.json"), parity[label])
        require(all(case["passed"] for case in parity[label]), "Native numerical parity failed: " + label)
    audio, rate = sf.read(audio_path, frames=128 * (64 + 256), dtype="float32", always_2d=True)
    require(rate == 44100 and audio.shape == (128 * 320, 2), "Timing input differs")
    timings = []
    for cycle, order in enumerate((("original", "encoder_gemm", "both_gemm"), ("both_gemm", "encoder_gemm", "original"),
                                  ("encoder_gemm", "original", "both_gemm"), ("both_gemm", "original", "encoder_gemm"))):
        for label in order:
            session = session_for(data[label])
            row = {"cycle": cycle + 1, "variant": label, **benchmark(session, audio.T, 64, 256)}
            timings.append(row)
            del session
            print(json.dumps({k: row[k] for k in ("cycle", "variant", "p50_ms", "p95_ms", "p99_ms")}), flush=True)
    pooled = {label: np.asarray([t for row in timings if row["variant"] == label for t in row["times_ms"]]) for label in data}
    summary = {label: {"p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)),
                      "p99_ms": float(np.percentile(values, 99)), "p50_ratio_to_original": float(np.percentile(values, 50) / np.percentile(pooled["original"], 50))}
               for label, values in pooled.items()}
    verify_inputs({"source_bindings": bindings})
    require(state_sha256(parent.state_dict()) == initial_parent_sha and state_sha256(model.state_dict()) == BASE_STATE
            and not torch.cuda.is_initialized(), "Native model or CPU scope changed")
    write(out / "result.json", {"status": "pass", "transforms": transforms,
          "all_variants_native_torch_waveform_and_state_parity": True, "summary": summary, "timing_cycles": timings,
          "graph_sha256": {label: hashlib.sha256(value).hexdigest() for label, value in data.items()},
          "native_plugin_unchanged": True, "graph_saved": False, "source_bindings_unchanged": True,
          "gpu_used": False, "target_m4_qualified": False, "full14_quality_measured": False,
          "elapsed_seconds": time.monotonic() - began, "counted_bytes_after": require_space(source, 370_000_000),
          "limitations": "Linux CPU1 screening while GPU training is active; speed ratios and parity do not establish M4 plugin deadlines."})
    print(json.dumps({"status": "pass", "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
