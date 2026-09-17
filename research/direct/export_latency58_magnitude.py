"""Export a verified 5 dB magnitude-adapter checkpoint with unchanged recurrent parity checks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import PHASE, ROOT, PYTHON, read, require, sha, write, execute


def run(plan_path, plan_sha):
    require(sha(plan_path) == plan_sha, "Residual export plan changed")
    plan = read(plan_path)
    require(plan["schema"] == "latency58-direct-sdr-export-plan-v1"
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
            and all(sha(p) == s for p, s in plan["source_bindings"].items()), "Require frozen CPU export inputs")
    import torch
    import onnx
    from research.direct.latency58_evaluate import model_state_sha256
    from research.direct.latency58_magnitude_checkpoint import SCHEMA, load_model
    from research.direct.latency58_residual_onnx import (
        INPUT_NAMES, INPUT_SHAPES, OUTPUT_NAMES, OUTPUT_SHAPES, STATE_SHAPES, make_export_copy, verify_onnx,
    )
    from research.direct.export_latency58_magnitude_metadata import metadata
    from research.direct.latency58_sdr_checkpoint import require_space
    budget = read(plan["training_plan"])
    counted = require_space(budget, 130_000_000)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoint = plan["checkpoint"]
    model, payload = load_model(checkpoint)
    require(payload["schema"] == SCHEMA and payload["plan_sha256"] == sha(plan["training_plan"]),
            "Export requires the saved direct-SDR training endpoint")
    fingerprint = model_state_sha256(model)
    rng = torch.get_rng_state().clone()
    wrapper = make_export_copy(model)
    props = metadata(plan, model, payload, budget, STATE_SHAPES)
    out = Path(plan["output_directory"])
    pending, final = out / "model.pending.onnx", out / "model.onnx"
    require(not pending.exists() and not final.exists(), "Preserve existing exports")
    began = time.monotonic()
    inputs = (torch.zeros(INPUT_SHAPES[0]), *model.initial_state(1))
    with torch.inference_mode():
        example = wrapper(*inputs)
        torch.onnx.export(wrapper, inputs, str(pending), export_params=True, opset_version=17,
                          do_constant_folding=True, input_names=list(INPUT_NAMES), output_names=list(OUTPUT_NAMES),
                          dynamo=False, external_data=False)
    graph = onnx.load(str(pending), load_external_data=False)
    require(tuple(v.name for v in graph.graph.output) == OUTPUT_NAMES
            and not any(v.data_location == onnx.TensorProto.EXTERNAL for v in graph.graph.initializer),
            "Export must retain the self-contained static five-output ABI")
    for value, tensor, shape in zip(graph.graph.output, example, OUTPUT_SHAPES, strict=True):
        require(tuple(tensor.shape) == shape, "Export-copy output shape differs")
        dims = value.type.tensor_type.shape
        dims.ClearField("dim")
        for size in shape:
            dims.dim.add().dim_value = size
    onnx.helper.set_model_props(graph, props)
    onnx.save(graph, str(pending))
    onnx.checker.check_model(str(pending), full_check=True)
    verification = verify_onnx(model, wrapper, pending, hops=48, audio_paths=plan["audio_paths"], threads=1)
    require(verification["passed"] and model_state_sha256(model) == model_state_sha256(wrapper.model) == fingerprint
            and torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
            and all(sha(p) == s for p, s in plan["source_bindings"].items()),
            "Saved-model/export-copy/ONNX recurrent parity failed or inputs changed")
    pending.rename(final)
    write(out / "verification.json", {"schema": "latency58-direct-sdr-export-result-v1", "status": "pass",
          "plan_sha256": plan_sha, "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
          "checkpoint": checkpoint, "model_state_sha256": fingerprint, "metadata": props,
          "onnx": {"path": str(final), "sha256": sha(final), "bytes": final.stat().st_size},
          "verification": verification, "counted_bytes_before": counted,
          "counted_bytes_after": require_space(budget, 2_000_000), "elapsed_seconds": time.monotonic() - began,
          "plugin_changed": False, "target_hardware_deadlines_qualified": False})
    print(json.dumps({"status": "pass", "onnx": str(final), "model_state_sha256": fingerprint}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-name")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    args = parser.parse_args()
    if args.plan is not None:
        run(args.plan, args.plan_sha256)
        return
    require(args.pilot_name and all(c.isalnum() or c in "-_" for c in args.pilot_name), "Specify an existing pilot name")
    source = PHASE / args.pilot_name
    training_path = source / "plan.json"
    training = read(training_path)
    qualification = read(source / "checkpoint-audit.json")
    result = read(source / "result.json")
    quality = read(source / "full14/result.json")
    execution = read(source / "full14/execution.json")
    checkpoint = qualification["checkpoint"]
    require(qualification["status"] == "pass" and qualification["source_bindings_unchanged"]
            and result["status"] == "training_audit_and_full14_complete" and result["target_reached"]
            and result["checkpoint"] == checkpoint == quality["results"][0]["checkpoint"]
            and quality["status"] == "pass" and quality["target_reached"] and quality["source_bindings_unchanged"]
            and quality["track_count"] == 14 and quality["excerpt_count"] == 28
            and quality["results"][0]["aggregate"]["full_sdr_db"] >= 5.0
            and quality["results"][0]["model"]["model_state_sha256"] == qualification["model_state_sha256"]
            and execution["actual_exit_code"] == 0 and not execution["timed_out"] and execution["source_bindings_unchanged"],
            "Export requires an audited saved checkpoint scoring at least 5 dB on all 14 tracks")
    manifest_path = ROOT / "research/manifests/valid.json"
    manifest = read(manifest_path)
    audio = [str(Path(manifest["root"]) / manifest["tracks"][i]["mixture"]) for i in (1, 10)]
    paths = [Path(__file__).resolve(), ROOT / "research/direct/export_latency58_magnitude_metadata.py",
             ROOT / "research/direct/latency58_residual_onnx.py", ROOT / "research/direct/latency58_asymmetric_onnx.py",
             ROOT / "research/direct/export_latency58_teacher.py", ROOT / "export_onnx.py", manifest_path,
             ROOT / "research/direct/export_latency58_direct_sdr_metadata.py",
             training_path, source / "checkpoint-audit.json", source / "result.json", source / "full14/result.json",
             source / "full14/execution.json", Path(checkpoint["path"]), *map(Path, audio)]
    bindings = {**training["source_bindings"], **quality["source_bindings"], **{str(p): sha(p) for p in paths}}
    out = source / "onnx"
    require(not out.exists() and all(sha(p) == s for p, s in bindings.items()), "Preserve exports and frozen inputs")
    plan = {"schema": "latency58-direct-sdr-export-plan-v1", "checkpoint": checkpoint,
            "training_plan": str(training_path), "quality_result": str(source / "full14/result.json"),
            "audio_paths": audio, "source_bindings": bindings, "output_directory": str(out)}
    out.mkdir()
    write(out / "plan.json", plan)
    execute([PYTHON, "-u", "-m", "research.direct.export_latency58_magnitude", "--plan", str(out / "plan.json"),
             "--plan-sha256", sha(out / "plan.json")], out, "export", 600, bindings,
            {"plan_sha256": sha(out / "plan.json")})


if __name__ == "__main__":
    main()
