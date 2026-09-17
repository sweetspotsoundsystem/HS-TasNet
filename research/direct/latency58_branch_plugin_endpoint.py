"""Authenticate the saved EMA checkpoint selected for PR #13 on September 13."""
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha
from research.direct.train_latency58 import verify_inputs

SOURCE = PHASE / "branch-pitch-ema-002"


def selected_endpoint():
    review_path = SOURCE / "selection-review.json"
    quality_path = SOURCE / "full14-ema/result.json"
    training_path = SOURCE / "plan.json"
    review, quality, training = (read(p) for p in (review_path, quality_path, training_path))
    checkpoint = review["best_research_checkpoint"]
    require(review["status"] == "selected_for_research" and review["actual_root_exit_code"] == 0
            and review["selected_weight_role"] == "ema" and review["total_training_updates"] == 39250
            and review["best_research_reference_result"] == str(quality_path)
            and review["best_full_sdr_db"] == 4.46515742201644,
            "The reviewed saved EMA endpoint differs")
    bindings = {}
    for value in (training, quality, review):
        for path, digest in value["source_bindings"].items():
            require(path not in bindings or bindings[path] == digest, "Conflicting source binding")
            bindings[path] = digest
    for relative in ("root-execution.json", "production-stage/execution.json", "full14-ema/execution.json"):
        path = SOURCE / relative
        execution = read(path)
        require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                and not execution.get("timed_out", False), "Saved endpoint execution is incomplete")
        bindings[str(path)] = sha(path)
    for path in (review_path, quality_path, training_path):
        bindings[str(path)] = sha(path)
    verify_inputs({"source_bindings": bindings})
    from research.direct.latency58_branch_memory_checkpoint import load_model
    model, payload = load_model(checkpoint)
    endpoint = quality["results"][0]
    require(quality["status"] == "pass" and quality["source_bindings_unchanged"]
            and quality["track_count"] == 14 and quality["excerpt_count"] == 28
            and endpoint["checkpoint"] == checkpoint
            and endpoint["aggregate"]["full_sdr_db"] == review["best_full_sdr_db"]
            and endpoint["model"]["model_state_sha256_after"] == payload["model_state_sha256"]
            and payload["plan_sha256"] == sha(training_path)
            and payload["provenance"]["checkpoint_weight_role"] == "averaged_inference",
            "Saved tensors, provenance or complete validation binding differs")
    return model, payload, training, checkpoint, quality_path, review_path, bindings


def export_bindings(bindings):
    names = (
        "latency58_branch_plugin_endpoint.py", "latency58_branch_onnx.py", "latency58_branch_onnx_export.py",
        "latency58_branch_int8.py", "latency58_branch_int8_verify.py", "latency58_attention_int8_precision.py",
        "latency58_int8_precise_float.py", "latency58_int8_reference.py", "latency58_conv_gemm.py",
        "latency58_quadrature_all_s8.py", "latency58_best_onnx.py", "latency58_asymmetric_onnx.py",
    )
    paths = [ROOT / "research/direct" / name for name in names] + [ROOT / "export_onnx.py"]
    return {**bindings, **{str(p): sha(p) for p in paths}}
