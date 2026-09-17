"""Authenticate explicit inference roles without writing unpacked model files."""
from pathlib import Path

from research.direct.run_latency58_quality import read, require, sha
from research.direct.train_latency58 import state_sha256
from research.direct.train_latency58_four_second_shared import budget_snapshot as snapshot


def training_plan(plan):
    bound = plan["packed_training_plan"]
    require(sha(bound["path"]) == bound["sha256"], "Packed training plan changed")
    prepared = read(bound["path"])
    require(prepared["name"] == "branch-four-second-015" and prepared["config"]["steps"] == 2000,
            "Evaluation belongs to another training recipe")
    from research.direct.train_latency58_four_second_shared import validate_recipe
    validate_recipe(prepared)
    return prepared, bound["sha256"]


def evaluation_storage(plan):
    prepared, _ = training_plan(plan)
    return snapshot(prepared)


def load_model(plan, spec):
    import torch
    from research.direct.latency58_four_second_recovery_files import load_inference, FINAL
    prepared, digest = training_plan(plan)
    role, checkpoint = spec["checkpoint_role"], spec["checkpoint"]
    require(role in ("raw", "ema") and checkpoint["step"] == prepared["config"]["steps"]
            and Path(checkpoint["path"]) == Path(prepared["output_directory"]) / "production-run" / FINAL,
            "Require an explicit role from the final saved generation")
    model, payload = load_inference(checkpoint, prepared, digest, role=role)
    require(not torch.cuda.is_initialized() and model.algorithmic_latency_samples == 256
            and state_sha256(model.state_dict()) == payload["model_state_sha256"] == spec["model_state_sha256"]
            and payload["provenance"]["checkpoint_weight_role"] ==
                {"raw": "raw_optimizer_endpoint", "ema": "averaged_inference"}[role]
            and payload["plan_sha256"] == digest and payload["step"] == checkpoint["step"],
            "Packed role, tensor identity, training plan or latency differs")
    return model, payload
