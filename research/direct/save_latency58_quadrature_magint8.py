"""Save the exact ten-projection graph tested in memory, with byte provenance."""
import hashlib
import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs

GRAPH_SHA = "6cfcc9d9ad70473dcaf9ce822af413e3d01fc303820798b8b726976e0b08cfcf"


def main():
    import onnxruntime as ort
    import torch
    from research.direct.latency58_pending_training_space import require_cpu_space
    from research.direct.latency58_quadrature_checkpoint import load_model
    from research.direct.latency58_quadrature_magint8 import build
    from research.direct.train_latency58 import state_sha256
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and ort.__version__ == "1.26.0", "Use the reviewed CPU workspace and runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    source_path = PHASE / "quadrature-continuation-001/plan.json"
    source = read(source_path)
    verify_inputs(source)
    before = require_cpu_space(source, 97_000_000)
    native_root = PHASE / "m4-quadrature-memory-native-001"
    quality_root = PHASE / "m4-quadrature-magint8-full14-memory-001"
    review, native = read(native_root / "selection-review.json"), read(native_root / "result.json")
    quality = read(quality_root / "result.json")
    require(review["status"] == "eligible_for_saved_graph_and_M4_package"
            and review["actual_root_exit_code"] == 0 and native["status"] == "pass"
            and native["native_speed_gate_passed"] and native["source_bindings_unchanged"]
            and len(native["cycles"]) == 18, "Require successful reviewed native comparison")
    require(quality["status"] == "pass" and quality["track_count"] == 14
            and quality["excerpt_count"] == 28 and quality["source_bindings_unchanged"]
            and not quality["graph_saved"]
            and quality["results"][0]["checkpoint"]["sha256"] == GRAPH_SHA,
            "Require completed full-panel scoring of the exact memory graph")
    paths = [source_path, Path(__file__).resolve()]
    for root in (native_root, quality_root):
        paths.extend(root / name for name in ("plan.json", "result.json", "selection-review.json"))
    for stage in ("m4-quadrature-magint8-screen-stage-001", "m4-quadrature-magint8-long-stage-001",
                  "m4-quadrature-magint8-full14-memory-stage-001", "m4-quadrature-memory-native-stage-001"):
        path = PHASE / stage / "execution.json"
        execution = read(path)
        require(execution["actual_exit_code"] == 0 and not execution["timed_out"]
                and execution["source_bindings_unchanged"], "A required wrapper did not close successfully")
        paths.append(path)
    bindings = {**review["source_bindings"], **{str(p): sha(p) for p in paths}}
    verify_inputs({"source_bindings": bindings})
    screen = read(PHASE / "m4-quadrature-magint8-screen-001/result.json")
    parent = source["parent_checkpoint"]
    require(parent == screen["checkpoint"] == read(quality_root / "plan.json")["parent_float_checkpoint"],
            "Wrong saved parent")
    model, _ = load_model(parent)
    fingerprint = state_sha256(model.state_dict())
    floating, integer, graph, proof = build(model)
    data = graph.SerializeToString()
    require(hashlib.sha256(data).hexdigest() == GRAPH_SHA == screen["graph_sha256"]
            == read(native_root / "plan.json")["runtime_graphs"]["quadrature_ten"]["sha256"]
            and len(data) == 32_132_574 and state_sha256(model.state_dict()) == fingerprint
            and not torch.cuda.is_initialized(), "Graph or parent changed during conversion")
    del floating, integer, graph, model
    out = PHASE / "m4-quadrature-magint8-saved-001"
    require(not out.exists(), "Preserve prior saved graphs")
    out.mkdir()
    write(out / "plan.json", {"source_bindings": bindings, "graph_sha256": GRAPH_SHA,
          "source_checkpoint": parent, "model_state_sha256": fingerprint,
          "counted_bytes_before": before, "graph_reservation_bytes": 35_000_000,
          "archive_reservation_bytes": 62_000_000,
          "quality_provenance": "Serialized file must match every byte of the completed in-memory full14 graph."})
    saved = out / "model.onnx"
    with saved.open("xb") as stream:
        require(stream.write(data) == len(data), "Incomplete graph write")
        stream.flush()
        os.fsync(stream.fileno())
    require(saved.stat().st_size == len(data) and sha(saved) == GRAPH_SHA, "Saved graph readback differs")
    descriptor = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    verify_inputs({"source_bindings": bindings})
    write(out / "result.json", {"status": "pass", "plan_sha256": sha(out / "plan.json"),
          "checkpoint": {"kind": "saved_onnx", "path": str(saved), "sha256": sha(saved), "bytes": len(data)},
          "graph_saved": True, "source_bindings_unchanged": True, "source_checkpoint": parent,
          "file_identical_to_quality_tested_memory_graph": True, "full14_rerun_from_saved_file": False,
          "full14_sdr_db": quality["results"][0]["aggregate"]["full_sdr_db"],
          "quality_result": {"path": str(quality_root / "result.json"), "sha256": sha(quality_root / "result.json")},
          "native_host_qualified": False, "mac_execution_performed": False, "plugin_modified": False,
          "counted_bytes_after": require_cpu_space(source, 62_000_000)})
    print(json.dumps(read(out / "result.json")), flush=True)


if __name__ == "__main__":
    main()
