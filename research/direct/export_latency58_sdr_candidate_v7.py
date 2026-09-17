"""CPU ONNX export for an audited quality-recovery endpoint and its evidence.

Requires a bound execution plan. Import is stdlib-only. Actual ONNX parity
does not establish a plugin queue, deadline performance or retained quality.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import tempfile
import time

from research.direct.latency58_checkpoint import require, sha

ROOT = Path(__file__).resolve().parents[2]


from research.direct.export_latency58_teacher import metadata as original_metadata

CANDIDATE_FAMILIES = {
    "sdr_candidate": "sdr",
    "context_candidate": "context",
    "log_relative_candidate": "log_relative",
    "sdr_softcap_candidate": "sdr_softcap",
    "sdr_softcap_strong_candidate": "sdr_softcap_strong",
    "lrrestart_candidate": "sdr_lrrestart",
    "accum_candidate": "sdr_accum",
    "drum_candidate": "sdr_drum_v3",
    "drum_accum_candidate": "sdr_drum_accum",
    "counterfactual_candidate": "counterfactual",
    "controlled_deployed_candidate": "controlled_deployed",
    "leader_cleanup_candidate": "leader_cleanup",
}


def checkpoint_loader(family):
    return "latency58_leader_cleanup_checkpoint_v2.py" if family == "leader_cleanup" \
        else "latency58_" + family + "_checkpoint.py"


def metadata(plan, model, step, fingerprint, shapes):
    result = original_metadata(plan, model, step, fingerprint, shapes)
    loader = checkpoint_loader(CANDIDATE_FAMILIES[plan["model_kind"]])
    result.update({"hs_tasnet.exporter_sha256": sha(__file__),
                   "hs_tasnet.checkpoint_loader_sha256": sha(ROOT / "research/direct" / loader),
                   "hs_tasnet.teacher_kind": model.provenance["teacher_kind"],
                   "hs_tasnet.sdr_trial_updates": str(model.provenance["sdr_trial_updates"]),
                   "hs_tasnet.context_trial_updates": str(model.provenance.get("context_trial_updates", 0)),
                   "hs_tasnet.log_relative_trial_updates": str(model.provenance.get("log_relative_trial_updates", 0)),
                   "hs_tasnet.log_relative_weight": str(model.provenance.get("log_relative_weight", 0.0)),
                   "hs_tasnet.log_relative_version": model.provenance.get("log_relative_version", "none"),
                   "hs_tasnet.sdr_softcap_trial_updates": str(model.provenance.get("sdr_softcap_trial_updates", 0)),
                   "hs_tasnet.sdr_softcap_strong_trial_updates": str(model.provenance.get("sdr_softcap_strong_trial_updates", 0)),
                   "hs_tasnet.sdr_softcap_weight": str(model.provenance.get("sdr_softcap_weight", 0.0)),
                   "hs_tasnet.sdr_softcap_version": model.provenance.get("sdr_softcap_version", "none"),
                   "hs_tasnet.sdr_softcap_error_ratio_floor": str(model.provenance.get("sdr_softcap_error_ratio_floor", 0.0)),
                   "hs_tasnet.lr_restart_trial_updates": str(model.provenance.get("lr_restart_trial_updates", 0)),
                   "hs_tasnet.accum_trial_updates": str(model.provenance.get("accum_trial_updates", 0)),
                   "hs_tasnet.drum_accum_trial_updates": str(model.provenance.get("drum_accum_trial_updates", 0)),
                   "hs_tasnet.drum_emphasis_trial_updates": str(model.provenance.get("drum_emphasis_trial_updates", 0)),
                   "hs_tasnet.drum_weight": str(model.provenance.get("drum_weight", 1)),
                   "hs_tasnet.drum_objective_version": model.provenance.get("objective_version", "none"),
                   "hs_tasnet.trial_augmented_examples": str(model.provenance.get("trial_augmented_examples", 0)),
                   "hs_tasnet.trial_microbatches": str(model.provenance.get("trial_microbatches", 0)),
                   "hs_tasnet.gradient_accumulation_steps": str(model.provenance.get("gradient_accumulation_steps", 1)),
                   "hs_tasnet.training_microbatch_size": str(model.provenance.get("training_microbatch_size", 4)),
                   "hs_tasnet.context_used_only_during_training": str("context_trial_updates" in model.provenance).lower(),
                   "hs_tasnet.training_warmup_samples": str(model.provenance.get("warmup_samples", 0)),
                   "hs_tasnet.training_scored_samples": str(model.provenance.get("scored_samples", 88064)),
                   "hs_tasnet.training_carried_state": str(model.provenance.get("carry_state", False)).lower()})
    if plan["model_kind"] in ("counterfactual_candidate", "controlled_deployed_candidate"):
        require(model.provenance["local_mask_mixer"] is False
                and model.provenance["teacher_mode"] == "ordinary_only", "Unsupported vocal export architecture")
        result.update({
            "hs_tasnet.vocal_focus_trial_updates": str(model.provenance["vocal_focus_trial_updates"]),
            "hs_tasnet.counterfactual_trial_updates": str(model.provenance["counterfactual_trial_updates"]),
            "hs_tasnet.counterfactual_version": model.provenance["counterfactual_version"],
            "hs_tasnet.teacher_mode": model.provenance["teacher_mode"],
            "hs_tasnet.teacher_batch_divisor": model.provenance["teacher_batch_divisor"],
            "hs_tasnet.focused_augmentation": str(model.provenance["focused_augmentation"]).lower(),
            "hs_tasnet.augmentation_version": model.provenance["augmentation_version"],
            "hs_tasnet.local_mask_mixer": "false",
        })
        if plan["model_kind"] == "controlled_deployed_candidate":
            require(model.provenance["additional_loss_weight"] == .5, "Different deployed training term")
            result.update({
                "hs_tasnet.controlled_deployed_trial_updates": str(model.provenance["controlled_deployed_trial_updates"]),
                "hs_tasnet.additional_loss_version": model.provenance["additional_loss_version"],
                "hs_tasnet.additional_loss_weight": str(model.provenance["additional_loss_weight"]),
                "hs_tasnet.additional_loss_used_only_during_training": "true",
                "hs_tasnet.deployed_truth_stem_weights": json.dumps(model.provenance["deployed_truth_stem_weights"]),
                "hs_tasnet.deployed_truth_divisor": str(model.provenance["deployed_truth_divisor"]),
            })
    if plan["model_kind"] == "leader_cleanup_candidate":
        provenance = model.provenance
        require(provenance["leader_cleanup_trial_updates"] == step == 250
                and provenance["local_mask_mixer"] is False
                and provenance["teacher_mode"] == "ordinary_only"
                and provenance["additional_loss_weight"] == .5
                and provenance["comparison_variable"] == "training_parent"
                and provenance["matched_loss_effect_from_leader_claimed"] is False,
                "Different leader-cleanup export recipe or endpoint")
        result.update({
            "hs_tasnet.leader_cleanup_trial_updates": str(provenance["leader_cleanup_trial_updates"]),
            "hs_tasnet.vocal_focus_trial_updates": str(provenance["vocal_focus_trial_updates"]),
            "hs_tasnet.counterfactual_version": provenance["counterfactual_version"],
            "hs_tasnet.teacher_mode": provenance["teacher_mode"],
            "hs_tasnet.teacher_batch_divisor": provenance["teacher_batch_divisor"],
            "hs_tasnet.focused_augmentation": str(provenance["focused_augmentation"]).lower(),
            "hs_tasnet.augmentation_version": provenance["augmentation_version"],
            "hs_tasnet.local_mask_mixer": "false",
            "hs_tasnet.additional_loss_version": provenance["additional_loss_version"],
            "hs_tasnet.additional_loss_weight": str(provenance["additional_loss_weight"]),
            "hs_tasnet.additional_loss_used_only_during_training": "true",
            "hs_tasnet.deployed_truth_stem_weights": json.dumps(provenance["deployed_truth_stem_weights"]),
            "hs_tasnet.deployed_truth_divisor": str(provenance["deployed_truth_divisor"]),
            "hs_tasnet.parent_model_state_sha256": provenance["parent_model_state_sha256"],
            "hs_tasnet.parent_checkpoint_sha256": provenance["parent_checkpoint"]["sha256"],
            "hs_tasnet.parent_training_updates": str(provenance["parent_training_updates"]),
            "hs_tasnet.reference_training_plan_sha256": provenance["reference_training_plan"]["sha256"],
            "hs_tasnet.comparison_variable": provenance["comparison_variable"],
            "hs_tasnet.matched_loss_effect_from_leader_claimed": "false",
        })
    return result


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Export plan changed")
    plan = json.loads(args.plan.read_text())
    require(plan["schema"] == "latency58-sdr-candidate-onnx-plan-v1" and Path.cwd() == ROOT,
            "Export schema or working directory differs")
    bindings = plan["source_bindings"]
    require(plan["model_kind"] in CANDIDATE_FAMILIES
            and plan["purpose"] in ("development_parity", "selected_candidate_parity"), "Unknown candidate or export purpose")
    family = CANDIDATE_FAMILIES[plan["model_kind"]]
    loader_name = checkpoint_loader(family)
    evaluator_name = "evaluate_latency58_" + family
    for relative in ("research/direct/export_latency58_sdr_candidate_v7.py", "research/direct/export_latency58_teacher.py",
                     "research/direct/report_latency58_sdr.py", "research/direct/evaluate_latency58_sdr.py",
                     "research/direct/" + evaluator_name + ".py",
                     "research/direct/latency58_asymmetric.py", "research/direct/latency58_asymmetric_onnx.py",
                     "research/direct/latency58.py", "research/direct/latency58_gpu.py", "research/direct/latency58_encoder_window.py",
                     "research/direct/latency58_checkpoint.py", "research/direct/" + loader_name, "export_onnx.py"):
        require(str(ROOT / relative) in bindings, "Missing export source binding: " + relative)
    require(all(sha(path) == digest for path, digest in bindings.items())
            and bindings.get(plan["checkpoint"]["path"]) == plan["checkpoint"]["sha256"]
            and bindings.get(plan["training_plan"]["path"]) == plan["training_plan"]["sha256"], "Export inputs changed")
    from research.direct.report_latency58_sdr import load_completed
    from research.direct.train_latency58 import read
    for mode in ("full14", "probes", "actions60"):
        path = Path(plan["quality_directories"][mode])
        quality_plan, report = load_completed(path, {})
        state = report["model_state_sha256"] if mode == "probes" else report["results"][0]["model"]["model_state_sha256"]
        require(state == plan["model_state_sha256"] and quality_plan["generation"] == plan["generation"]
                and quality_plan["step"] == plan["step"]
                and all(bindings.get(str(path / name)) == sha(path / name) for name in ("plan.json", "result.json", "execution.json")),
                "Export does not identify the completed quality endpoint")
        require(report["status"] == "pass" if mode == "probes" else report["inputs_unchanged"], "Quality evaluation failed")
        if mode == "full14":
            require(len(report["results"][0]["tracks"]) == 14, "Export requires the complete primary panel")
    audit, execution = read(plan["audit"]["path"]), read(plan["audit_execution"]["path"])
    require(audit["status"] == "pass" and audit["step"] == plan["step"]
            and audit["model_state_sha256"] == plan["model_state_sha256"] and audit["checkpoint"] == plan["checkpoint"]
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and audit["source_bindings_unchanged"] and execution["source_bindings_unchanged"]
            and all(bindings.get(plan[k]["path"]) == plan[k]["sha256"] for k in ("audit", "audit_execution")),
            "Export lacks its actual saved-state audit")
    require(plan["quality_evidence"] and all(sha(row["path"]) == row["sha256"]
            and bindings.get(row["path"]) == row["sha256"] for row in plan["quality_evidence"]), "Quality evidence binding differs")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(name) == "1" for name in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CUDA-hidden CPU1")
    require(type(plan["verify_hops"]) is int and plan["verify_hops"] >= 8
            and all(bindings.get(row["path"]) == row["sha256"] for row in plan["verify_audio"]),
            "Verification geometry or audio binding differs")
    output = Path(plan["output"])
    require(output.is_absolute() and output.suffix == ".onnx" and output.parent.is_dir(), "Output directory must already exist")
    result_path, failed_path = output.with_suffix(".verification.json"), output.with_suffix(".failed-verification.json")
    failed_graph = output.with_suffix(".failed.onnx")
    require(all(not path.exists() and not path.is_symlink() for path in (output, result_path, failed_path, failed_graph)),
            "Preserve existing outputs")
    from research.direct.latency58_sdr_checkpoint import require_space
    phase = ROOT / "research/direct/runs/latency58"
    require(output.is_relative_to(phase), "Keep candidate export inside the artifact allowance")
    before_bytes = require_space(plan, 240_000_000)
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    import soundfile as sf
    evaluator = importlib.import_module("research.direct." + evaluator_name)
    load_evaluation_model = evaluator.load_evaluation_model
    from research.direct.latency58_evaluate import model_state_sha256
    from research.direct.latency58_asymmetric_onnx import (
        INPUT_NAMES, INPUT_SHAPES, OUTPUT_NAMES, OUTPUT_SHAPES, STATE_SHAPES, make_export_copy, verify_onnx,
    )
    versions = {"torch": torch.__version__, "numpy": np.__version__, "onnx": onnx.__version__,
                "onnxruntime": ort.__version__, "soundfile": sf.__version__}
    require(versions == plan["runtime_versions"] and not torch.cuda.is_initialized(), "CPU runtime versions differ")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260907)
    torch.use_deterministic_algorithms(True)
    model, receipt = load_evaluation_model(plan)
    step = receipt["step"]
    fingerprint = model_state_sha256(model)
    require(step == plan["step"] and fingerprint == plan["model_state_sha256"], "Export checkpoint state differs")
    rng = torch.get_rng_state().clone()
    wrapper = make_export_copy(model)
    props = metadata(plan, model, step, fingerprint, STATE_SHAPES)
    descriptor, name = tempfile.mkstemp(prefix=output.stem + ".", suffix=".pending.onnx", dir=output.parent)
    os.close(descriptor)
    temporary = Path(name)
    started = time.monotonic()
    report = {"schema": "latency58-sdr-candidate-onnx-verification-v1", "plan_sha256": args.plan_sha256,
              "checkpoint": plan["checkpoint"], "model_state_sha256": fingerprint, "step": step,
              "model_provenance": model.provenance,
              "metadata": props, "runtime_versions": versions, "quality_evidence": plan["quality_evidence"],
              "source_bindings": bindings, "native_host_qualified": False, "host_queue_implemented": False,
              "quality_retention_decision": None, "training_updates_executed": 0,
              "export_purpose": plan["purpose"], "counted_bytes_before": before_bytes}
    try:
        inputs = (torch.zeros(INPUT_SHAPES[0]), *model.initial_state(1))
        with torch.inference_mode():
            example = wrapper(*inputs)
            torch.onnx.export(wrapper, inputs, str(temporary), export_params=True, opset_version=17,
                              do_constant_folding=True, input_names=list(INPUT_NAMES), output_names=list(OUTPUT_NAMES),
                              dynamo=False, external_data=False)
        graph = onnx.load(str(temporary), load_external_data=False)
        require(tuple(value.name for value in graph.graph.output) == OUTPUT_NAMES
                and not any(value.data_location == onnx.TensorProto.EXTERNAL for value in graph.graph.initializer),
                "Graph output order or self-contained storage differs")
        for value, tensor, shape in zip(graph.graph.output, example, OUTPUT_SHAPES, strict=True):
            require(tuple(tensor.shape) == shape, "Export-copy shape differs")
            dims = value.type.tensor_type.shape
            dims.ClearField("dim")
            for size in shape:
                dims.dim.add().dim_value = size
        onnx.helper.set_model_props(graph, props)
        onnx.save(graph, str(temporary))
        onnx.checker.check_model(str(temporary), full_check=True)
        report["verification"] = verify_onnx(model, wrapper, temporary, hops=plan["verify_hops"],
                                             audio_paths=tuple(row["path"] for row in plan["verify_audio"]), threads=1)
        require(report["verification"]["passed"]
                and model_state_sha256(model) == model_state_sha256(wrapper.model) == fingerprint
                and torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
                and all(sha(path) == digest for path, digest in bindings.items()),
                "ONNX parity, unchanged source/tensors/RNG or CPU scope failed")
        report.update(status="passed_cpu_numerical_verification_only", onnx_sha256=sha(temporary),
                      onnx_bytes=temporary.stat().st_size, elapsed_seconds=time.monotonic() - started,
                      source_bindings_unchanged=True)
        os.link(temporary, output)
        write_new(result_path, report)
        print(json.dumps({"status": report["status"], "output": str(output), "onnx_sha256": report["onnx_sha256"]}))
    except Exception as error:
        report.update(status="failed_not_qualified", error=repr(error), elapsed_seconds=time.monotonic() - started)
        if temporary.stat().st_size:
            os.link(temporary, failed_graph)
        write_new(failed_path, report)
        raise
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
