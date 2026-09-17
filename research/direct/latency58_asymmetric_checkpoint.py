"""Load an inference snapshot or a separately audited resume for CPU scoring.

Resume files are trusted local training artifacts. Their full SHA256 and a
successful separate saved-state audit are required before deserialization.
No optimizer is constructed and no random state is restored.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_model_state(model, checkpoint):
    import torch
    from research.direct.latency58_asymmetric import VERSION
    from research.direct.latency58_evaluate import model_state_sha256

    require(not torch.cuda.is_initialized() and all(value.device.type == "cpu" for value in model.state_dict().values()),
            "Checkpoint scoring must stay on CPU")
    path = Path(checkpoint["path"])
    require(sha(path) == checkpoint["sha256"], "Checkpoint bytes changed")
    kind = checkpoint.get("kind", "inference")
    require(kind in ("inference", "audited_resume"), "Unknown checkpoint kind")
    audit = None
    if kind == "audited_resume":
        audit_binding, execution_binding = checkpoint["audit"], checkpoint["audit_execution"]
        require(sha(audit_binding["path"]) == audit_binding["sha256"]
                and sha(execution_binding["path"]) == execution_binding["sha256"], "Audit evidence changed")
        audit = json.loads(Path(audit_binding["path"]).read_text())
        execution = json.loads(Path(execution_binding["path"]).read_text())
        require(audit["status"] == "pass" and audit["resume_sha256"] == checkpoint["sha256"]
                and execution["actual_exit_code"] == 0 and not execution.get("timed_out", False)
                and execution["resume_sha256_before"] == execution["resume_sha256_after"] == checkpoint["sha256"],
                "Resume has no successful observed saved-state audit")
    payload = torch.load(path, map_location="cpu", weights_only=(kind == "inference"))
    expected_schema = "latency58-asymmetric-inference-v1" if kind == "inference" else "latency58-asymmetric-resume-v1"
    require(payload["schema"] == expected_schema and payload["architecture"] == model.architecture_metadata
            and payload["provenance"]["version"] == VERSION
            and type(payload["step"]) is int and payload["step"] > 0,
            "Checkpoint architecture, family or update count differs")
    provenance = payload["provenance"]
    require(provenance["asymmetric_training_updates"] == provenance["tail_updates"] == payload["step"]
            and provenance["training_updates"] == 4250 + payload["step"]
            and provenance["pilot_updates"] == 2000 + payload["step"]
            and provenance["training_objective"] == "raw4_l1", "Asymmetric update lineage differs")
    if audit is not None:
        require(payload["step"] == audit["step"] and payload["plan_sha256"] == audit["plan_sha256"]
                and payload["model_state_sha256"] == audit["model_state_sha256"], "Audited model identity differs")
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    model.load_state_dict(payload["model"], strict=True)
    require(all(value.dtype == torch.float32 and torch.isfinite(value).all().item()
                for value in model.state_dict().values())
            and all(torch.equal(value, buffers[name]) for name, value in model.named_buffers())
            and model_state_sha256(model) == payload["model_state_sha256"],
            "Checkpoint tensors, fixed buffers or fingerprint differ")
    model.provenance = payload["provenance"]
    require(sha(path) == checkpoint["sha256"] and not torch.cuda.is_initialized(),
            "Checkpoint changed during load or CUDA was initialized")
    return payload["step"]


def make_model(parent_checkpoint):
    """Reconstruct the frozen buffers from the authenticated Hann parent."""
    from research.direct.latency58 import Latency58Model
    from research.direct.latency58_checkpoint import load_model_state as load_parent
    from research.direct.latency58_asymmetric import Latency58AsymmetricModel

    parent = Latency58Model.from_accepted().eval().requires_grad_(False)
    require(load_parent(parent, parent_checkpoint) == 2000, "Use the frozen Hann +2000 parent")
    return Latency58AsymmetricModel.from_hann_model(parent).eval().requires_grad_(False)
