"""Check the selected checkpoint's declared integer inference without saving weights."""
import hashlib
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs


def main():
    import onnxruntime as ort
    import torch
    from research.direct.check_latency58_best_onnx_memory import selected_endpoint
    from research.direct.latency58_attention_int8 import build, make_reference
    from research.direct.latency58_attention_int8_verify import session_for, run_case
    from research.direct.latency58_best_onnx import interface, TOLERANCES
    from research.direct.latency58_best_onnx_verify import cases
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
            and ort.__version__ == "1.26.0", "Use CPU1 and the shipping ORT runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    budget = read(PHASE / "temporal-attention-001/plan.json")
    require_space(budget, 5_000_000)
    native, payload, training, checkpoint, quality_path, review_path = selected_endpoint()
    fingerprint, contract = state_sha256(native.state_dict()), interface(native)
    old_failure = PHASE / "best-model-onnx-long-001/result.json"
    require(read(old_failure)["status"] == "fail", "Preserve the FP32 long-state parity failure")
    old_diagnostic = PHASE / "attention-onnx-drift-diagnostic-stage-005/execution.json"
    require(read(old_diagnostic)["actual_exit_code"] == 0 and read(old_diagnostic)["source_bindings_unchanged"],
            "Complete the bounded FP32 arithmetic diagnostics first")
    music = Path("/home/axel/HS-TasNet/data/musdb18hq/train/Actions - One Minute Smile/mixture.wav")
    names = ("latency58_attention_int8.py", "latency58_attention_int8_precision.py", "latency58_attention_int8_verify.py",
             "latency58_best_onnx.py", "latency58_best_onnx_verify.py", "latency58_best_onnx_export.py",
             "check_latency58_best_onnx_memory.py", "latency58_conv_gemm.py", "latency58_quadrature_all_s8.py",
             "latency58_int8_reference.py", "latency58_int8_precise_float.py")
    paths = [Path(__file__).resolve(), review_path, quality_path, old_failure, old_diagnostic, music,
             *(ROOT / "research/direct" / name for name in names)]
    bindings = {**read(PHASE / "best-model-onnx-memory-001/plan.json")["source_bindings"],
                **{str(p): sha(p) for p in paths}}
    verify_inputs({"source_bindings": bindings})
    out = PHASE / "attention-int8-screen-001"
    require(not out.exists(), "Preserve integer checks")
    out.mkdir()
    write(out / "plan.json", {"source_bindings": bindings, "checkpoint": checkpoint, "interface": contract,
          "tolerances": TOLERANCES, "verification_hops": 1024, "repetitions": 2,
          "optimizations": ["disabled", "all"], "graph_saved": False,
          "reference": "Independent signed quantization reconstruction and CPU PyTorch int32 dots with declared floating precision",
          "parity_to_unquantized_fp32_source_claimed": False,
          "full14_quality_and_quiet_native_timing_required_before_handoff": True})
    began = time.monotonic()
    integer, graph, conversion = build(native, payload, training, checkpoint, quality_path)
    reference, independent = make_reference(native, integer, conversion)
    data = graph.SerializeToString()
    graph_sha = hashlib.sha256(data).hexdigest()
    require(len(data) < 40_000_000, "Integer graph exceeded the handoff estimate")
    write(out / "graph.json", {"graph_sha256": graph_sha, "graph_bytes": len(data),
          "metadata": {v.key: v.value for v in graph.metadata_props}, "conversion": conversion,
          "independent_integer_projections": independent})
    del integer, graph
    rows = []
    for optimization in ("disabled", "all"):
        session = session_for(data, contract, optimization)
        for case in cases(native, hops=1024, audio_paths=[music]):
            repetitions = [run_case(reference, session, case, contract) for _ in range(2)]
            exact = repetitions[0]["all_output_and_state_trajectory_sha256"] == repetitions[1]["all_output_and_state_trajectory_sha256"]
            row = {"case": case["name"], "optimization": optimization, "repetitions": repetitions,
                   "reset_replay_all_outputs_and_states_bit_exact": exact,
                   "passed": exact and all(v["passed"] for v in repetitions)}
            rows.append(row)
            write(out / f"case-{len(rows)}.json", row)
            print(json.dumps({"case": case["name"], "optimization": optimization, "passed": row["passed"],
                              "errors": repetitions[0]["maximum_errors_in_physical_units"]}), flush=True)
        del session
    verify_inputs({"source_bindings": bindings})
    require(not torch.cuda.is_initialized() and state_sha256(native.state_dict()) == fingerprint,
            "Source model or CPU scope changed")
    passed = all(row["passed"] for row in rows)
    write(out / "result.json", {"status": "pass" if passed else "fail", "strict_parity_passed": passed,
          "cases": rows, "source_bindings_unchanged": True, "checkpoint": checkpoint,
          "graph_sha256": graph_sha, "graph_bytes": len(data), "graph_saved": False,
          "tolerances_changed": False, "parity_to_unquantized_fp32_source_claimed": False,
          "quality_measured": False, "native_host_qualified": False,
          "elapsed_seconds": time.monotonic() - began, "counted_bytes_after": require_space(budget, 0)})
    require(passed, "Attention integer reference parity failed")
    print(json.dumps({"status": "pass", "graph_sha256": graph_sha, "graph_bytes": len(data)}), flush=True)


if __name__ == "__main__":
    main()
