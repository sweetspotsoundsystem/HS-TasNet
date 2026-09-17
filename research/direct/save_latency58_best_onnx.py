"""Save only the exact integer graph that passed strict reference checks and full14 quality."""
import hashlib
import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs


def main():
    import onnxruntime as ort
    import torch
    from research.direct.check_latency58_best_onnx_memory import selected_endpoint
    from research.direct.latency58_attention_int8 import build
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "" and ort.__version__ == "1.26.0",
            "Require CPU and shipping runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    budget = read(PHASE / "temporal-attention-001/plan.json")
    # Includes the export, plugin copy, native build and prospective LFS copy.
    before = require_space(budget, 600_000_000)
    paths, bindings, checks = [Path(__file__).resolve()], {}, []
    for name in ("attention-int8-screen", "attention-int8-long"):
        root = PHASE / (name + "-001")
        result, plan = read(root / "result.json"), read(root / "plan.json")
        execution_path = PHASE / (name + "-stage-001/execution.json")
        execution = read(execution_path)
        require(result["status"] == "pass" and result["source_bindings_unchanged"]
                and execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                and not execution["timed_out"], "Both streaming checks must have closed")
        verify_inputs(plan)
        checks.append(result)
        bindings.update(plan["source_bindings"])
        paths.extend((root / "result.json", root / "plan.json", execution_path))
    require(checks[0]["graph_sha256"] == checks[1]["graph_sha256"], "Short and long graphs differ")
    model, payload, training, checkpoint, quality_path, review_path = selected_endpoint()
    fingerprint = state_sha256(model.state_dict())
    integer, graph, conversion = build(model, payload, training, checkpoint, quality_path)
    del integer
    data = graph.SerializeToString()
    require(hashlib.sha256(data).hexdigest() == checks[0]["graph_sha256"]
            and len(data) == checks[0]["graph_bytes"] and len(data) < 40_000_000
            and state_sha256(model.state_dict()) == fingerprint and not torch.cuda.is_initialized(),
            "Exact graph or saved checkpoint changed")
    deployment_quality_path = PHASE / "attention-int8-full14-memory-001/result.json"
    deployment_execution_path = PHASE / "attention-int8-full14-memory-stage-001/execution.json"
    deployment_quality, deployment_execution = read(deployment_quality_path), read(deployment_execution_path)
    require(deployment_quality["status"] == "pass" and deployment_quality["quality_handoff_gate_passed"]
            and deployment_quality["source_bindings_unchanged"] and deployment_quality["graph_sha256"] == checks[0]["graph_sha256"]
            and deployment_quality["track_count"] == 14 and deployment_quality["excerpt_count"] == 28
            and deployment_execution["actual_exit_code"] == 0 and deployment_execution["source_bindings_unchanged"]
            and not deployment_execution["timed_out"], "Deployment quality must close and meet the predeclared handoff gate")
    paths.extend((deployment_quality_path, deployment_execution_path, PHASE / "attention-int8-full14-memory-001/plan.json"))
    bindings.update(deployment_quality["source_bindings"])
    retirement = PHASE / "historical-ola-resume-retirement-005"
    receipt = read(retirement / "receipt.json")
    require(receipt["status"] == "complete" and receipt["preserved_bindings_unchanged"]
            and receipt["all_inference_models_and_source_audio_preserved"]
            and receipt["reserved_bytes"] == 600_000_000, "Handoff storage reservation is incomplete")
    paths.extend((retirement / "intent.json", retirement / "receipt.json", review_path, quality_path))
    bindings.update({str(p): sha(p) for p in paths})
    verify_inputs({"source_bindings": bindings})
    out = PHASE / "best-model-onnx-saved-001"
    require(not out.exists(), "Preserve saved exports")
    out.mkdir()
    write(out / "plan.json", {"source_bindings": bindings, "checkpoint": checkpoint,
          "graph_sha256": checks[0]["graph_sha256"], "counted_bytes_before": before,
          "reserved_handoff_bytes": 600_000_000, "purpose": "Validated best-model PR for user M4/M4 Pro tests"})
    path = out / "model.onnx"
    with path.open("xb") as stream:
        require(stream.write(data) == len(data), "Incomplete graph write")
        stream.flush()
        os.fsync(stream.fileno())
    require(sha(path) == checks[0]["graph_sha256"] and path.stat().st_size == len(data), "Saved graph readback differs")
    descriptor = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    verify_inputs({"source_bindings": bindings})
    write(out / "result.json", {"status": "pass", "source_bindings_unchanged": True,
          "checkpoint": checkpoint, "model_state_sha256": fingerprint,
          "onnx": {"path": str(path), "sha256": sha(path), "bytes": len(data)},
          "file_identical_to_checked_memory_graph": True, "native_host_qualified": False,
          "quality_result": {"path": str(deployment_quality_path), "sha256": sha(deployment_quality_path)},
          "source_checkpoint_quality_result": {"path": str(quality_path), "sha256": sha(quality_path)},
          "conversion": conversion,
          "deployment_graph_full14_sdr_db": deployment_quality["results"][0]["aggregate"]["full_sdr_db"],
          "saved_bytes_identical_to_full14_scored_graph": True,
          "saved_checkpoint_full14_sdr_db": read(quality_path)["results"][0]["aggregate"]["full_sdr_db"],
          "full14_rerun_from_saved_onnx": False, "plugin_modified": False,
          "counted_bytes_after": require_space(budget, 560_000_000)})
    print(json.dumps(read(out / "result.json")), flush=True)


if __name__ == "__main__":
    main()
