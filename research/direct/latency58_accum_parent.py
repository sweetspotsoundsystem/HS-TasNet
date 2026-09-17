"""Load an authenticated accumulation endpoint as a fresh fine-tuning parent.

Only inference weights are inherited. The original accumulation loader checks
the complete lineage and fixed buffers without requiring its retired optimizer.
The caller is responsible for a completed quality review before training.
"""
from __future__ import annotations

from pathlib import Path

from research.direct.train_latency58 import read, require, sha, state_sha256


def load_parent(plan):
    import torch
    from research.direct.latency58_sdr_accum_checkpoint import load_model

    require(not torch.cuda.is_initialized(), "Load accumulation parent before CUDA")
    parent = plan["parent"]
    require(parent["kind"] == "accum_candidate" and type(parent["step"]) is int
            and parent["step"] in (250, 500, 1000), "Require a planned accumulation endpoint")
    binding = parent["training_plan"]
    generation = Path(parent["generation"])
    require(sha(binding["path"]) == binding["sha256"]
            and plan["source_bindings"].get(binding["path"]) == binding["sha256"]
            and parent["checkpoint"]["path"] == str(generation / "model.pt")
            and sha(parent["checkpoint"]["path"]) == parent["checkpoint"]["sha256"],
            "Accumulation parent bytes or training plan changed")
    for name in ("model.pt", "receipt.json", "rng.pt", "metrics.jsonl"):
        path = generation / name
        require(plan["source_bindings"].get(str(path)) == sha(path), "Unbound parent generation: " + name)
    training = read(binding["path"])
    require(training["schema"] == "latency58-sdr-accum-training-v1"
            and generation == Path(training["run_dir"]) / "checkpoints" / f"step-{parent['step']:06d}"
            and training["carry_state"] and training["accumulation_steps"] == 4
            and training["microbatch_size"] == 4 and training["config"]["batch_size"] == 16,
            "Parent accumulation recipe or directory differs")
    model, receipt = load_model(generation, training, expected_plan_sha=binding["sha256"])
    require(receipt["step"] == parent["step"] and receipt["carry_state"]
            and receipt["files"]["model.pt"]["sha256"] == parent["checkpoint"]["sha256"],
            "Wrong accumulation parent endpoint")
    payload = torch.load(parent["checkpoint"]["path"], map_location="cpu", weights_only=True)
    require(payload["provenance"] == model.provenance == parent["provenance"]
            and model.provenance["training_updates"]
            == training["parent"]["provenance"]["training_updates"] + parent["step"]
            and state_sha256(payload["model"]) == state_sha256(model.state_dict())
            == parent["model_state_sha256"] == receipt["model_state_sha256"]
            and payload["architecture"] == model.architecture_metadata
            and len(list(model.parameters())) == 21 and len(list(model.buffers())) == 6
            and not torch.cuda.is_initialized(), "Parent lineage, state or inference geometry differs")
    return model
