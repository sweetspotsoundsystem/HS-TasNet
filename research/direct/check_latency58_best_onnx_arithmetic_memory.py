"""Check the selected saved model's FP32 export without writing graph weights."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs


from research.direct.check_latency58_best_onnx_memory import selected_endpoint


def selected_variant():
    result_path = PHASE / "attention-onnx-drift-diagnostic-004/result.json"
    execution_path = PHASE / "attention-onnx-drift-diagnostic-stage-004/execution.json"
    result, execution = read(result_path), read(execution_path)
    require(result["status"] == "diagnostic_complete" and result["source_bindings_unchanged"]
            and execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
            and not execution["timed_out"], "Wait for completed arithmetic diagnostics")
    plan_path = result_path.parent / "plan.json"
    verify_inputs(read(plan_path))
    priority = ("fused_gru", "split_gru_bias", "precise_gru_gemm")
    require(tuple(v["variant"] for v in result["variants"]) == priority, "Diagnostic variants differ")
    passing = [v for v in result["variants"] if v["original_tolerances_passed"]]
    require(passing, "No arithmetic variant passed the original long-stream thresholds")
    return passing[0], (result_path, plan_path, execution_path)


def main():
    import onnxruntime as ort
    import torch
    from research.direct.latency58_attention_onnx_arithmetic import build
    from research.direct.latency58_best_onnx_verify import verify_memory
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
            and ort.__version__ == "1.26.0", "Require CPU1 and the shipping runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    budget = read(PHASE / "temporal-attention-001/plan.json")
    require_space(budget, 5_000_000)
    model, payload, training, checkpoint, quality_path, review_path = selected_endpoint()
    variant, diagnostic_paths = selected_variant()
    fingerprint = state_sha256(model.state_dict())
    music = Path("/home/axel/HS-TasNet/data/musdb18hq/train/Actions - One Minute Smile/mixture.wav")
    paths = [Path(__file__).resolve(), review_path, quality_path, Path(checkpoint["path"]), music, *diagnostic_paths]
    paths.extend(ROOT / "research/direct" / name for name in (
        "check_latency58_best_onnx_memory.py", "latency58_attention_onnx_arithmetic.py",
        "latency58_best_onnx.py", "latency58_best_onnx_export.py", "latency58_best_onnx_verify.py",
        "latency58_asymmetric_onnx.py", "latency58_quadrature_onnx.py"))
    paths.append(ROOT / "export_onnx.py")
    bindings = {**training["source_bindings"], **read(quality_path)["source_bindings"],
                **{str(path): sha(path) for path in paths}}
    verify_inputs({"source_bindings": bindings})
    out = PHASE / "best-model-onnx-arithmetic-memory-001"
    require(not out.exists(), "Preserve export checks")
    out.mkdir()
    write(out / "plan.json", {"source_bindings": bindings, "checkpoint": checkpoint,
          "arithmetic_variant": variant["variant"],
          "quality_result": str(quality_path), "training_plan": str(quality_path.parents[1] / "plan.json"),
          "selection_review": str(review_path), "verification_hops": 1024, "repetitions": 2,
          "optimizations": ["disabled", "all"], "graph_saved": False, "native_host_qualified": False,
          "purpose": "User-requested PR with the validated best model for M4 and M4 Pro testing; the 5 dB research goal remains active."})
    began = time.monotonic()
    wrapper, graph = build(model, payload, training, checkpoint, quality_path, variant=variant["variant"])
    data = graph.SerializeToString()
    digest = hashlib.sha256(data).hexdigest()
    require(digest == variant["graph_sha256"], "Rebuilt arithmetic graph differs from its long diagnostic")
    write(out / "graph.json", {"graph_sha256": digest, "graph_bytes": len(data),
          "metadata": {value.key: value.value for value in graph.metadata_props}})
    del graph
    reports = []
    for optimization in ("disabled", "all"):
        report = verify_memory(model, wrapper, data, hops=1024, audio_paths=[music], optimization=optimization)
        reports.append(report)
        write(out / (optimization + ".json"), report)
    verify_inputs({"source_bindings": bindings})
    require(state_sha256(model.state_dict()) == state_sha256(wrapper.model.state_dict()) == fingerprint
            and not torch.cuda.is_initialized(), "Source tensors or CPU scope changed")
    passed = all(report["passed"] for report in reports)
    write(out / "result.json", {"status": "pass" if passed else "fail", "reports": reports,
          "checkpoint": checkpoint, "model_state_sha256": fingerprint,
          "graph_sha256": digest, "graph_bytes": len(data), "graph_saved": False,
          "source_bindings_unchanged": True, "tolerances_changed": False, "plugin_modified": False,
          "native_host_qualified": False, "elapsed_seconds": time.monotonic() - began,
          "counted_bytes_after": require_space(budget, 0)})
    require(passed, "Best-model arithmetic streaming parity failed")
    print(json.dumps({"status": "pass", "graph_sha256": digest, "graph_bytes": len(data)}), flush=True)


if __name__ == "__main__":
    main()
