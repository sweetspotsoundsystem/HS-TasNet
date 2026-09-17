"""Inference ABI metadata with the actual current-stage training provenance."""
from __future__ import annotations

import json
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, read, require, sha


def metadata(plan, model, payload, training, shapes):
    from research.direct.export_latency58_teacher import metadata as geometry_metadata
    from research.direct.latency58_residual_model import VERSION

    provenance = payload["provenance"]
    quality = read(plan["quality_result"])
    endpoint = quality["results"][0]
    require(payload["plan_sha256"] == provenance["direct_sdr_training_plan_sha256"] == sha(plan["training_plan"])
            and provenance["direct_sdr_objective_version"] == training["objective_version"]
            and provenance["direct_sdr_updates"] == payload["step"] == training["config"]["steps"]
            and provenance["teacher_kind"] == "none" and provenance["teacher_weight"] == 0
            and provenance["teacher_model_state_sha256"] is None
            and endpoint["checkpoint"] == plan["checkpoint"]
            and endpoint["model"]["model_state_sha256"] == payload["model_state_sha256"]
            and quality["status"] == "pass" and quality["track_count"] == 14 and quality["excerpt_count"] == 28
            and quality["source_bindings_unchanged"] and endpoint["aggregate"]["full_sdr_db"] >= 5.0,
            "Direct-SDR metadata must identify its own completed training and quality evidence")
    # The base helper provides the unchanged inference geometry. Its historical
    # training-plan and teacher fields are all replaced before serialization.
    props = geometry_metadata(plan, model, payload["step"], payload["model_state_sha256"], shapes)
    props.pop("hs_tasnet.asymmetric_training_updates", None)
    props.update({
        "hs_tasnet.training_plan_sha256": payload["plan_sha256"],
        "hs_tasnet.training_objective": training["objective_version"],
        "hs_tasnet.teacher_kind": "none",
        "hs_tasnet.teacher_weight": "0.0",
        "hs_tasnet.teacher_model_state_sha256": "none",
        "hs_tasnet.online_teacher_used_in_current_stage": "false",
        "hs_tasnet.teacher_generated_targets_in_current_stage": str(
            training["config"]["root_weights"].get("recordpool_best200_v1", 0) > 0).lower(),
        "hs_tasnet.current_stage_sampling_root_weights": json.dumps(training["config"]["root_weights"]),
        "hs_tasnet.current_stage_augmentation": training["config"].get("augmentation", "base_subset_and_vocal_derangement"),
        "hs_tasnet.additional_training_updates": str(payload["step"]),
        "hs_tasnet.direct_sdr_updates": str(payload["step"]),
        "hs_tasnet.parent_training_updates": str(provenance["direct_sdr_parent_updates"]),
        "hs_tasnet.parent_checkpoint_sha256": provenance["direct_sdr_parent_checkpoint"]["sha256"],
        "hs_tasnet.parent_model_state_sha256": provenance["direct_sdr_parent_model_state_sha256"],
        "hs_tasnet.training_precision": training["precision_policy"],
        "hs_tasnet.training_batch_size": str(training["config"]["batch_size"]),
        "hs_tasnet.training_microbatch_size": str(training["config"]["microbatch_size"]),
        "hs_tasnet.gradient_accumulation_steps": str(training["accumulation_steps"]),
        "hs_tasnet.training_warmup_samples": str(training["warmup_samples"]),
        "hs_tasnet.training_scored_samples": str(training["scored_samples"]),
        "hs_tasnet.training_carried_state": "true",
        "hs_tasnet.context_used_only_during_training": "true",
        "hs_tasnet.output_policy": "DBV += (delayed mixture - sum(native raw4))/16; Other = delayed mixture - sum(corrected DBV)",
        "hs_tasnet.output_policy_version": VERSION,
        "hs_tasnet.fixed_residual_share": "0.0625",
        "hs_tasnet.extra_audio_buffering_samples": "0",
        "hs_tasnet.extra_stream_state_tensors": "0",
        "hs_tasnet.full14_sdr_db": str(endpoint["aggregate"]["full_sdr_db"]),
        "hs_tasnet.full14_result_sha256": sha(plan["quality_result"]),
        "hs_tasnet.model_source_sha256": sha(ROOT / "research/direct/latency58_residual_model.py"),
        "hs_tasnet.export_copy_source_sha256": sha(ROOT / "research/direct/latency58_residual_onnx.py"),
        "hs_tasnet.exporter_sha256": sha(ROOT / "research/direct/export_latency58_direct_sdr.py"),
        "hs_tasnet.checkpoint_loader_sha256": sha(ROOT / "research/direct/latency58_direct_sdr_checkpoint.py"),
        "hs_tasnet.metadata_source_sha256": sha(Path(__file__)),
    })
    require(all(isinstance(k, str) and isinstance(v, str) for k, v in props.items()),
            "ONNX metadata requires string keys and values")
    return props
