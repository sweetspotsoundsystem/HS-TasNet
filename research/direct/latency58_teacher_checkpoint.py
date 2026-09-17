"""Strict CPU loader for a completed matched teacher-trial inference file."""
from __future__ import annotations

from research.direct.latency58_asymmetric_checkpoint import make_model
from research.direct.train_latency58 import require, sha, state_sha256


def load_model_state(model, checkpoint):
    import torch
    from research.direct.latency58_asymmetric import VERSION

    require(checkpoint.get("kind") == "inference" and sha(checkpoint["path"]) == checkpoint["sha256"]
            and not torch.cuda.is_initialized()
            and all(v.device.type == "cpu" for v in model.state_dict().values()), "Use authenticated CPU inference bytes")
    payload = torch.load(checkpoint["path"], map_location="cpu", weights_only=True)
    provenance = payload["provenance"]
    require(payload["schema"] == "latency58-teacher-inference-v1" and type(payload["step"]) is int
            and payload["step"] == 250 and payload["architecture"] == model.architecture_metadata
            and provenance["version"] == VERSION and provenance["training_updates"] == 5000
            and provenance["pilot_updates"] == 2750 and provenance["asymmetric_training_updates"] == 750
            and provenance["tail_updates"] == provenance["teacher_trial_updates"] == 250
            and provenance["teacher_weight"] in (0.0, 0.5)
            and provenance["training_objective"] == ("raw4_l1" if provenance["teacher_weight"] == 0 else "raw4_l1_plus_teacher_l1")
            and provenance["parent_checkpoint"]["sha256"] ==
            "889f24e482328601bd70d84e0fc16c7b94776ac91e604e2782539bec153108aa"
            and provenance["parent_model_state_sha256"] ==
            "fa6bffecde6c426b014d1c9e833e0b2aa1846e884387ffdf6424899c9094cff2"
            and provenance["teacher_model_state_sha256"] ==
            "a12c215810026c603a1fd394383c1646219b8b3f764ebe9c2a83856404443aa4"
            and provenance["training_plan_sha256"] == payload["plan_sha256"], "Trial lineage or architecture differs")
    frozen = {name: value.clone() for name, value in model.named_buffers()}
    model.load_state_dict(payload["model"], strict=True)
    require(all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in model.state_dict().values())
            and all(torch.equal(v, frozen[name]) for name, v in model.named_buffers())
            and state_sha256(model.state_dict()) == payload["model_state_sha256"], "Saved tensors or fixed buffers differ")
    model.provenance = provenance
    require(sha(checkpoint["path"]) == checkpoint["sha256"] and not torch.cuda.is_initialized(), "Load changed inputs or used CUDA")
    return payload["step"]
