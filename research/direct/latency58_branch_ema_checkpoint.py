"""Publish distinct raw/Adam and EMA inference endpoints in one generation."""
from __future__ import annotations

import os
from pathlib import Path

from research.direct.run_latency58_quality import read, require, sha, write
from research.direct.train_latency58 import state_sha256
from research.direct.latency58_branch_ema import BranchParameterEMA, SCHEMA as EMA_SCHEMA
from research.direct.latency58_branch_memory_checkpoint import make_payloads as make_raw_payloads
from research.direct.latency58_branch_memory_checkpoint import load_payload, audit_resume

SCHEMA = "latency58-branch-ema-generation-v1"
FILES = {"model.pt", "raw-model.pt", "raw-optimizer.pt", "ema.json"}
RESERVE_BYTES = 600_000_000


def policy(decay):
    return {"schema": EMA_SCHEMA, "decay": decay, "update_every": 1,
            "initialization": "raw_parent", "fixed_buffers": "unchanged"}


def _provenance(metadata):
    return {key: metadata[key] for key in ("schema", "decay", "updates", "base_state_sha256",
                                          "raw_state_sha256", "ema_parameters_sha256")}


def make_payloads(model, optimizer, ema, step, plan, plan_sha):
    require(type(ema) is BranchParameterEMA and plan["ema"] == policy(ema.decay)
            and ema.updates == step == plan["config"]["steps"]
            and ema.base_state_sha256 == plan["parent_model_state_sha256"],
            "EMA save policy or endpoint differs from its plan")
    raw, resume = make_raw_payloads(model, optimizer, step, plan, plan_sha)
    # A future EMA parent may carry an averaging history. The newly optimized
    # raw endpoint itself is no longer an averaged inference endpoint.
    provenance = dict(raw["provenance"])
    if "weight_averaging" in provenance:
        provenance["initial_parent_weight_averaging"] = provenance.pop("weight_averaging")
    raw["provenance"] = {**provenance, "checkpoint_weight_role": "raw_optimizer_endpoint"}
    state = ema.state_dict(model)
    metadata = {key: value for key, value in state.items() if key not in ("parameters", "buffers")}
    tensors = {**state["parameters"], **state["buffers"]}
    averaged = {**raw, "model": tensors, "model_state_sha256": state_sha256(tensors),
                "provenance": {**raw["provenance"], "checkpoint_weight_role": "averaged_inference",
                               "weight_averaging": _provenance(metadata), "quality_measured": False}}
    return raw, resume, averaged, metadata


def audit_payloads(raw, resume, averaged, metadata, plan, plan_sha):
    """Reconstruct EMA state from its inference tensors, without storing a duplicate."""
    require(raw["step"] == averaged["step"] == plan["config"]["steps"]
            and raw["plan_sha256"] == averaged["plan_sha256"] == plan_sha
            and raw["architecture"] == averaged["architecture"]
            and raw["parameter_names"] == averaged["parameter_names"]
            and raw["fixed_buffers_sha256"] == averaged["fixed_buffers_sha256"] == plan["fixed_buffers_sha256"],
            "Raw and averaged payloads belong to different endpoints")
    raw_model, raw = load_payload(raw)
    averaged_model, averaged = load_payload(averaged)
    audit_resume(raw_model, raw, resume, plan, plan_sha)
    require(raw["provenance"].get("checkpoint_weight_role") == "raw_optimizer_endpoint"
            and "weight_averaging" not in raw["provenance"]
            and raw["provenance"]["branch_memory_parent_checkpoint"] == plan["parent_checkpoint"]
            and raw["provenance"]["branch_memory_parent_model_state_sha256"] == plan["parent_model_state_sha256"]
            and raw["provenance"]["branch_memory_parent_updates"] == plan["parent_training_updates"]
            and averaged["provenance"] == {**raw["provenance"], "checkpoint_weight_role": "averaged_inference",
                                           "weight_averaging": _provenance(metadata), "quality_measured": False},
            "Raw optimizer ownership or averaged provenance differs")
    require("parameters" not in metadata and "buffers" not in metadata
            and plan["ema"] == policy(metadata["decay"]), "EMA metadata or policy differs")
    names = raw["parameter_names"]
    buffers = list(dict(raw_model.named_buffers()))
    state = {**metadata, "parameters": {name: averaged["model"][name] for name in names},
             "buffers": {name: averaged["model"][name] for name in buffers}}
    restored_ema = BranchParameterEMA.from_state_dict(raw_model, state, expected_step=raw["step"],
        decay=plan["ema"]["decay"], base_state_sha256=plan["parent_model_state_sha256"])
    require(restored_ema.raw_state_sha256 == resume["model_state_sha256"] == raw["model_state_sha256"],
            "EMA and Adam belong to different raw weights")
    return raw_model, resume, averaged_model, restored_ema


def _flush_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_generation(model, optimizer, ema, step, plan, plan_sha, run):
    import torch
    from research.direct.latency58_sdr_checkpoint import require_space
    run = Path(run)
    pending, final = run / "checkpoint.pending", run / "checkpoint"
    require(not pending.exists() and not final.exists(), "Preserve existing EMA generations")
    require_space(plan, RESERVE_BYTES + plan.get("concurrent_training_reservation_bytes", 0))
    raw, resume, averaged, metadata = make_payloads(model, optimizer, ema, step, plan, plan_sha)
    audit_payloads(raw, resume, averaged, metadata, plan, plan_sha)
    pending.mkdir()
    for name, payload in (("raw-model.pt", raw), ("raw-optimizer.pt", resume), ("model.pt", averaged)):
        with (pending / name).open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
    write(pending / "ema.json", metadata)
    with (pending / "ema.json").open("rb") as stream:
        os.fsync(stream.fileno())
    require({p.name for p in pending.iterdir()} == FILES, "Wrong pending EMA inventory")
    receipt = {"schema": SCHEMA, "step": step, "plan_sha256": plan_sha,
               "raw_model_state_sha256": raw["model_state_sha256"],
               "model_state_sha256": averaged["model_state_sha256"],
               "optimizer_owner": "raw-model.pt", "inference_model": "model.pt",
               "parameter_names": raw["parameter_names"], "ema_policy": plan["ema"],
               "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in pending.iterdir()},
               "metrics_sha256": sha(run / "metrics.jsonl")}
    write(pending / "receipt.json", receipt)
    with (pending / "receipt.json").open("rb") as stream:
        os.fsync(stream.fileno())
    _flush_directory(pending)
    pending.rename(final)
    _flush_directory(run)
    return {"path": str(final / "model.pt"), "sha256": sha(final / "model.pt")}


def load_generation(binding, plan, plan_sha):
    import torch
    path = Path(binding["path"])
    directory = path.parent
    require(path.name == "model.pt" and path.is_file() and not path.is_symlink()
            and not directory.is_symlink() and sha(path) == binding["sha256"],
            "Averaged inference checkpoint bytes changed")
    require({p.name for p in directory.iterdir()} == FILES | {"receipt.json"},
            "EMA generation inventory differs")
    require(all(p.is_file() and not p.is_symlink() for p in directory.iterdir()),
            "EMA generation must contain only regular files")
    receipt = read(directory / "receipt.json")
    require(receipt["schema"] == SCHEMA and set(receipt["files"]) == FILES
            and receipt["step"] == plan["config"]["steps"] and receipt["plan_sha256"] == plan_sha
            and receipt["optimizer_owner"] == "raw-model.pt" and receipt["inference_model"] == "model.pt"
            and receipt["ema_policy"] == plan["ema"]
            and receipt["metrics_sha256"] == sha(directory.parent / "metrics.jsonl"),
            "EMA generation receipt, ownership or journal differs")
    for name, expected in receipt["files"].items():
        item = directory / name
        require(item.stat().st_size == expected["bytes"] and sha(item) == expected["sha256"],
                "EMA generation file changed: " + name)
    raw, resume, averaged = (torch.load(directory / name, map_location="cpu", weights_only=True)
                            for name in ("raw-model.pt", "raw-optimizer.pt", "model.pt"))
    metadata = read(directory / "ema.json")
    require(raw["model_state_sha256"] == receipt["raw_model_state_sha256"]
            and averaged["model_state_sha256"] == receipt["model_state_sha256"]
            and raw["parameter_names"] == receipt["parameter_names"], "EMA receipt tensor identities differ")
    models = audit_payloads(raw, resume, averaged, metadata, plan, plan_sha)
    return (*models, {"raw": raw, "averaged": averaged, "metadata": metadata, "receipt": receipt})


def audit_saved(binding, plan, plan_sha):
    import torch
    raw, resume, averaged, ema, payloads = load_generation(binding, plan, plan_sha)
    with torch.inference_mode():
        audio = torch.linspace(-.1, .1, 2 * 8 * 128).reshape(1, 2, 8 * 128)
        original, replay = averaged.render(audio), ema.inference_copy(raw).render(audio)
        require(all(torch.equal(getattr(original, k).view(torch.int32), getattr(replay, k).view(torch.int32))
                    for k in ("raw", "deployed", "native_raw", "spectral", "waveform", "delayed_mixture"))
                and len(original.state) == len(replay.state) == 8
                and all(torch.equal(a.view(torch.int32), b.view(torch.int32))
                        for a, b in zip(original.state, replay.state, strict=True)),
                "Saved EMA inference and reconstructed average differ")
        closure = float((original.deployed.sum(1) - original.delayed_mixture).abs().max())
        require(closure < 1e-6 and raw.algorithmic_latency_samples == averaged.algorithmic_latency_samples == 256,
                "Saved EMA closure or latency differs")
    return {"status": "pass", "step": ema.updates, "checkpoint": binding,
            "model_state_sha256": payloads["averaged"]["model_state_sha256"],
            "raw_model_state_sha256": payloads["raw"]["model_state_sha256"],
            "optimizer_owner": "raw-model.pt", "saved_optimizer_tensor_count": len(resume["optimizer"]["state"]),
            "ema_reconstructed_without_duplicate_tensors": True, "averaged_outputs_and_eight_states_bit_exact": True,
            "algorithmic_latency_samples": 256, "closure_max_abs": closure, "quality_measured": False}
