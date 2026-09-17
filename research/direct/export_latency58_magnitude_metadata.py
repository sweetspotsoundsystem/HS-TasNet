"""Describe the actual magnitude architecture and frozen-parent training stage."""
from __future__ import annotations

import json
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, require, sha


def metadata(plan, model, payload, training, shapes):
    from research.direct.export_latency58_direct_sdr_metadata import metadata as direct_metadata
    from research.direct.latency58_magnitude import ADAPTER, VERSION
    props = direct_metadata(plan, model, payload, training, shapes)
    architecture = model.architecture_metadata
    provenance = payload["provenance"]
    require(architecture["version"] == VERSION == training["model_version"]
            and training["trainable_parameter_names"] == [ADAPTER]
            and provenance["magnitude_updates"] == payload["step"]
            and provenance["inherited_tensors_frozen_during_magnitude_training"],
            "Magnitude metadata must identify the actual trained architecture")
    props.update({
        "hs_tasnet.architecture_version": architecture["version"],
        "hs_tasnet.state_family": architecture["state_family"],
        "hs_tasnet.magnitude_features": architecture["magnitude_features"],
        "hs_tasnet.magnitude_normalization_axes": architecture["magnitude_normalization_axes"],
        "hs_tasnet.additional_neural_parameters": str(architecture["additional_neural_parameters"]),
        "hs_tasnet.trainable_parameter_names_in_current_stage": json.dumps([ADAPTER]),
        "hs_tasnet.inherited_tensors_frozen_in_current_stage": "true",
        "hs_tasnet.inherited_model_state_sha256": provenance["magnitude_parent_state_sha256"],
        "hs_tasnet.parent_model_state_sha256": provenance["magnitude_parent_state_sha256"],
        "hs_tasnet.initialized_magnitude_state_sha256": provenance["direct_sdr_parent_model_state_sha256"],
        "hs_tasnet.magnitude_updates": str(payload["step"]),
        "hs_tasnet.model_source_sha256": sha(ROOT / "research/direct/latency58_magnitude.py"),
        "hs_tasnet.exporter_sha256": sha(ROOT / "research/direct/export_latency58_magnitude.py"),
        "hs_tasnet.checkpoint_loader_sha256": sha(ROOT / "research/direct/latency58_magnitude_checkpoint.py"),
        "hs_tasnet.metadata_source_sha256": sha(Path(__file__)),
    })
    require(all(isinstance(k, str) and isinstance(v, str) for k, v in props.items()), "ONNX properties must be strings")
    return props
