"""Build a faithful FP32 graph for the reviewed research winner in memory."""
from __future__ import annotations

import io
import json
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, read, require, sha


def metadata(model, payload, training, checkpoint, quality_path):
    from research.direct.latency58_best_onnx import interface
    contract = interface(model)
    quality = read(quality_path)
    endpoint = quality["results"][0]
    require(quality["status"] == "pass" and quality["track_count"] == 14 and quality["excerpt_count"] == 28
            and quality["source_bindings_unchanged"] and endpoint["checkpoint"] == checkpoint
            and endpoint["model"]["model_state_sha256_after"] == payload["model_state_sha256"]
            and payload["step"] == training["config"]["steps"], "Bind the actual completed saved quality endpoint")
    architecture = model.architecture_metadata
    family = "temporal_attention" if len(contract["state_names"]) == 6 else "fusion_refinement"
    values = {
        "kind": "cropped1024_asymmetric_hop128", "mode": "streaming",
        "architecture_version": architecture["version"], "state_family": architecture["state_family"],
        "state_interchangeable_with_old_ola512": "false",
        "sample_rate": "44100", "hop_samples": "128", "analysis_fft_samples": "1024",
        "carrier_fft_samples": "1024", "synthesis_fft_samples": "1024", "spectral_mask_bins": "513",
        "spectral_output_crop": "[768,1024]", "synthesis_frame_samples": "256",
        "waveform_decoder_samples": "256", "analysis_history_samples": "896",
        "graph_output_delay_samples": "128", "alignment_samples": "128", "future_context_samples": "128",
        "future_callbacks_beyond_received_input": "0", "flush_required": "true", "flush_hops": "1",
        "initial_state": "all_zeros", "preroll": "discard_first_output_hop_after_reset",
        "external_host_queue_implemented": "false", "intended_external_host_queue_samples": "128",
        "intended_total_latency_samples": "256", "graph_qualified": "false",
        "native_host_timing_qualified": "false", "runtime_variant": family + "-fp32-v1",
        "output_policy": "DBV += (delayed mixture - sum(native raw4))/16; Other = delayed mixture - sum(corrected DBV)",
        "fixed_residual_share": "0.0625", "source_order": "drums,bass,vocals,other",
        "state_names": json.dumps(contract["state_names"]), "state_shapes": json.dumps(contract["state_shapes"]),
        "public_fusion_state_scale": str(2.0 ** -18),
        "output_source_scales": json.dumps(model.output_source_scales.tolist()),
        "checkpoint_sha256": checkpoint["sha256"], "model_state_sha256": payload["model_state_sha256"],
        "training_plan_sha256": payload["plan_sha256"], "snapshot_step": str(payload["step"]),
        "training_updates": str(payload["provenance"]["training_updates"]),
        "parent_checkpoint_sha256": training["parent_checkpoint"]["sha256"],
        "parent_model_state_sha256": training["parent_model_state_sha256"],
        "parent_training_updates": str(training["parent_training_updates"]),
        "training_objective": training["objective_version"],
        "training_precision": training["precision_policy"],
        "current_stage_augmentation": training["config"]["augmentation"],
        "training_batch_size": str(training["config"]["batch_size"]),
        "training_microbatch_size": str(training["config"]["microbatch_size"]),
        "gradient_accumulation_steps": str(training["accumulation_steps"]),
        "training_warmup_samples": str(training["warmup_samples"]),
        "training_scored_samples": str(training["scored_samples"]), "training_carried_state": "true",
        "online_teacher_used_in_current_stage": "false", "teacher_used_in_inference": "false",
        "teacher_generated_targets_in_current_stage": "false",
        "additional_audio_buffering_samples": "0",
        "additional_state_tensors": str(len(contract["state_names"]) - 4),
        "full14_sdr_db": str(endpoint["aggregate"]["full_sdr_db"]), "full14_result_sha256": sha(quality_path),
        "architecture": json.dumps(architecture, sort_keys=True),
        "exporter_sha256": sha(Path(__file__)),
        "model_source_sha256": sha(ROOT / "research/direct" / ("latency58_" + family + ".py")),
        "export_copy_source_sha256": sha(ROOT / "research/direct/latency58_best_onnx.py"),
        "export_helpers_sha256": sha(ROOT / "export_onnx.py"),
        "checkpoint_loader_sha256": sha(ROOT / "research/direct" / ("latency58_" + family + "_checkpoint.py")),
        "fft_implementation": "DFT1024; explicit513 Hermitian endpoints/interior; inverse1024 then crop768:1024",
        "external_data": "false", "analysis_window": "Wang2021 K1024 M128 d0",
        "spectral_synthesis_window": "matched asymmetric pair", "waveform_synthesis_window": "periodic Hann256",
        "analysis_synthesis_product": "periodic Hann256; hop128 overlap unity",
    }
    require(not training["online_teacher_used"] and not training["teacher_generated_targets_in_current_stage"],
            "Current-stage teacher provenance differs")
    require(all(isinstance(value, str) for value in values.values()), "Metadata must contain strings")
    return {"hs_tasnet." + key: value for key, value in values.items()}


def build(model, payload, training, checkpoint, quality_path):
    import onnx
    import torch
    from research.direct.latency58_best_onnx import interface, make_export_copy
    from research.direct.train_latency58 import state_sha256
    fingerprint = state_sha256(model.state_dict())
    require(fingerprint == payload["model_state_sha256"], "Native model differs from its saved payload")
    contract = interface(model)
    wrapper = make_export_copy(model)
    with io.BytesIO() as stream, torch.inference_mode():
        inputs = (torch.zeros(1, 2, 128), *model.initial_state(1))
        outputs = wrapper(*inputs)
        require(tuple(tuple(value.shape) for value in outputs) == contract["output_shapes"], "Export-copy shape differs")
        torch.onnx.export(wrapper, inputs, stream, export_params=True, opset_version=17,
                          do_constant_folding=True, input_names=list(contract["input_names"]),
                          output_names=list(contract["output_names"]), dynamo=False, external_data=False)
        graph = onnx.load_model_from_string(stream.getvalue())
    for value, shape in zip(graph.graph.output, contract["output_shapes"], strict=True):
        dims = value.type.tensor_type.shape
        dims.ClearField("dim")
        for size in shape:
            dims.dim.add().dim_value = size
    onnx.helper.set_model_props(graph, metadata(model, payload, training, checkpoint, quality_path))
    onnx.checker.check_model(graph, full_check=True)
    require(not any(value.data_location == onnx.TensorProto.EXTERNAL for value in graph.graph.initializer)
            and state_sha256(model.state_dict()) == state_sha256(wrapper.model.state_dict()) == fingerprint,
            "Export changed model tensors or uses external weights")
    return wrapper, graph
