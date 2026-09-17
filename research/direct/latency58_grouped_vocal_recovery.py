"""Atomic rolling recovery of raw weights, Adam, EMA, RNG and journal prefix.

The training schedule always comes from the original plan. The existing EMA
serializer receives an endpoint validation view for the completed step only;
that view is never used for training or presented as the training plan.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

from research.direct.run_latency58_quality import read, require, sha, write
from research.direct.train_latency58 import state_sha256
from research.direct.latency58_branch_ema_checkpoint import make_payloads, audit_payloads

SCHEMA = "latency58-grouped-rolling-recovery-v1"
MAX_BYTES = 600_000_000


def policy():
    return {"schema": SCHEMA, "interval_updates": 50, "optimizer_owner": "raw",
            "atomic_replace_only": "recovery.pt", "retain_generation_receipts": True,
            "retain_existing_baselines": True, "maximum_file_bytes": MAX_BYTES,
            "restore": ["raw", "adam", "ema", "torch_rng", "cuda_rng", "python_rng", "numpy_rng", "data_cursor"],
            "quality_claim": False}


def endpoint_view(plan, step):
    require(type(step) is int and 0 < step <= plan["config"]["steps"], "Invalid recovery endpoint")
    return {**plan, "config": {**plan["config"], "steps": step}}


def validate_journal(journal, plan, step, raw_sha, ema_sha):
    require(isinstance(journal, bytes) and journal.endswith(b"\n"), "Incomplete recovery journal")
    rows = [json.loads(line) for line in journal.splitlines()]
    config = plan["config"]
    require(len(rows) == step and [row["step"] for row in rows] == list(range(1, step + 1))
            and all(row["first_sample_index"] == config["data_start"] + (index - 1) * config["batch_size"]
                    and row["next_sample_index"] == config["data_start"] + index * config["batch_size"]
                    for index, row in enumerate(rows, 1))
            and rows[-1]["raw_model_state_sha256"] == raw_sha
            and rows[-1]["ema_parameters_sha256"] == ema_sha,
            "Recovery journal does not describe the saved raw/EMA endpoint")
    return rows


def validate_rng(resume):
    import numpy as np
    import torch
    require(isinstance(resume["torch_rng"], torch.Tensor) and resume["torch_rng"].dtype == torch.uint8
            and resume["torch_rng"].device.type == "cpu" and resume["torch_rng"].ndim == 1,
            "Invalid saved CPU RNG")
    torch.Generator().set_state(resume["torch_rng"])
    random.Random().setstate(resume["python_rng"])
    state = resume["numpy_rng"]
    require(isinstance(state, list) and len(state) == 5 and state[0] == "MT19937"
            and isinstance(state[1], torch.Tensor) and state[1].dtype == torch.int64
            and state[1].device.type == "cpu" and state[1].shape == (624,)
            and bool(((state[1] >= 0) & (state[1] <= 2**32 - 1)).all()), "Invalid saved NumPy RNG")
    np.random.RandomState().set_state((state[0], state[1].numpy().astype(np.uint32), *state[2:]))
    require(isinstance(resume["cuda_rng"], list) and len(resume["cuda_rng"]) <= 1
            and all(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                    and value.dtype == torch.uint8 and value.ndim == 1 and value.numel() > 0
                    for value in resume["cuda_rng"]), "Invalid saved CUDA RNG")


def make_snapshot(model, optimizer, ema, step, plan, plan_sha, journal):
    require(plan["recovery_checkpoint"] == policy(), "Unprepared rolling recovery policy")
    raw, resume, average, metadata = make_payloads(model, optimizer, ema, step, endpoint_view(plan, step), plan_sha)
    validate_rng(resume)
    validate_journal(journal, plan, step, raw["model_state_sha256"], metadata["ema_parameters_sha256"])
    return {"schema": SCHEMA, "plan_sha256": plan_sha, "planned_stop_step": plan["config"]["steps"],
            "training_config": copy.deepcopy(plan["config"]), "step": step, "policy": policy(),
            "raw": raw, "resume": resume, "average": average, "ema_metadata": metadata,
            "journal": journal, "journal_sha256": hashlib.sha256(journal).hexdigest()}


def audit_snapshot(snapshot, plan, plan_sha):
    require(snapshot["schema"] == SCHEMA and snapshot["policy"] == plan["recovery_checkpoint"] == policy()
            and snapshot["plan_sha256"] == plan_sha and snapshot["training_config"] == plan["config"]
            and snapshot["planned_stop_step"] == plan["config"]["steps"]
            and snapshot["step"] == snapshot["raw"]["step"] == snapshot["resume"]["step"]
            == snapshot["average"]["step"] == snapshot["ema_metadata"]["updates"]
            and hashlib.sha256(snapshot["journal"]).hexdigest() == snapshot["journal_sha256"],
            "Recovery snapshot plan, schedule or endpoint changed")
    step = snapshot["step"]
    validate_rng(snapshot["resume"])
    validate_journal(snapshot["journal"], plan, step, snapshot["raw"]["model_state_sha256"],
                     snapshot["ema_metadata"]["ema_parameters_sha256"])
    return audit_payloads(snapshot["raw"], snapshot["resume"], snapshot["average"],
                          snapshot["ema_metadata"], endpoint_view(plan, step), plan_sha)


def flush_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_snapshot(snapshot, directory, plan, plan_sha, *, before_replace=None):
    """Keep the old file usable until the complete new file is durable.

Only this run's explicitly designated rolling recovery file is replaced. An
interruption leaves its pending file and both immutable receipts for review.
"""
    import torch
    from research.direct.run_latency58_deployed_vocal_views import budget_snapshot
    require(snapshot["plan_sha256"] == plan_sha and snapshot["policy"] == plan["recovery_checkpoint"] == policy(),
            "Unprepared recovery publication")
    audit_snapshot(snapshot, plan, plan_sha)
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(), "Recovery directory must already exist")
    current, pending = directory / "recovery.pt", directory / "recovery.pending.pt"
    receipts = directory / "recovery-receipts"
    require(not pending.exists() and not current.is_symlink() and not receipts.is_symlink(),
            "Preserve interrupted recovery publication")
    previous = None
    if current.exists():
        previous_digest = sha(current)
        candidates = [read(path) for path in receipts.glob("step-*.json")]
        matches = [row for row in candidates if row["sha256"] == previous_digest]
        require(len(matches) == 1 and matches[0]["plan_sha256"] == plan_sha
                and matches[0]["step"] < snapshot["step"] and current.stat().st_size == matches[0]["bytes"],
                "Existing rolling file is not an earlier authenticated snapshot from this run")
        previous = {k: matches[0][k] for k in ("step", "sha256", "bytes")}
    budget_snapshot(plan["storage_budget"])
    receipts.mkdir(exist_ok=True)
    receipt_path = receipts / f"step-{snapshot['step']:06d}.json"
    require(not receipt_path.exists(), "Preserve previous recovery receipt")
    began = time.monotonic()
    with pending.open("xb") as stream:
        torch.save(snapshot, stream)
        stream.flush()
        os.fsync(stream.fileno())
    require(pending.stat().st_size <= MAX_BYTES, "Recovery snapshot exceeded its reserved size")
    persisted = torch.load(pending, map_location="cpu", weights_only=True)
    audit_snapshot(persisted, plan, plan_sha)
    require(persisted["journal_sha256"] == snapshot["journal_sha256"]
            and persisted["raw"]["model_state_sha256"] == snapshot["raw"]["model_state_sha256"]
            and persisted["ema_metadata"]["ema_parameters_sha256"] == snapshot["ema_metadata"]["ema_parameters_sha256"],
            "Serialized recovery snapshot changed")
    del persisted
    receipt = {"schema": SCHEMA, "step": snapshot["step"], "plan_sha256": plan_sha,
               "sha256": sha(pending), "bytes": pending.stat().st_size, "previous": previous,
               "journal_sha256": snapshot["journal_sha256"],
               "raw_model_state_sha256": snapshot["raw"]["model_state_sha256"],
               "ema_parameters_sha256": snapshot["ema_metadata"]["ema_parameters_sha256"],
               "planned_stop_step": snapshot["planned_stop_step"], "quality_claim": False}
    write(receipt_path, receipt)
    with receipt_path.open("rb") as stream:
        os.fsync(stream.fileno())
    flush_directory(receipts)
    if before_replace is not None:
        before_replace()
    os.replace(pending, current)
    flush_directory(directory)
    return {"path": str(current), "sha256": receipt["sha256"], "receipt": str(receipt_path),
            "receipt_sha256": sha(receipt_path), "step": snapshot["step"],
            "publication_seconds": time.monotonic() - began}


def read_snapshot(binding, plan, plan_sha):
    import torch
    path, receipt_path = Path(binding["path"]), Path(binding["receipt"])
    require(path.is_file() and not path.is_symlink() and not path.parent.is_symlink()
            and receipt_path.parent == path.parent / "recovery-receipts"
            and sha(receipt_path) == binding["receipt_sha256"], "Recovery receipt changed")
    receipt = read(receipt_path)
    require(receipt["schema"] == SCHEMA and receipt["plan_sha256"] == plan_sha
            and receipt["step"] == binding["step"]
            and path.stat().st_size == receipt["bytes"] <= MAX_BYTES
            and sha(path) == receipt["sha256"] == binding["sha256"], "Recovery file changed")
    snapshot = torch.load(path, map_location="cpu", weights_only=True)
    require(snapshot["step"] == receipt["step"] and snapshot["journal_sha256"] == receipt["journal_sha256"]
            and snapshot["raw"]["model_state_sha256"] == receipt["raw_model_state_sha256"]
            and snapshot["ema_metadata"]["ema_parameters_sha256"] == receipt["ema_parameters_sha256"],
            "Recovery file and receipt disagree")
    return snapshot, audit_snapshot(snapshot, plan, plan_sha)


def restore_training(snapshot, audited, plan, *, device, precision):
    import numpy as np
    import torch
    from research.direct.latency58_branch_ema import BranchParameterEMA
    from research.direct.latency58_branch_memory_checkpoint import audit_live
    raw, resume, _average, cpu_ema = audited
    ema_state = cpu_ema.state_dict(raw)
    raw.to(device).train().requires_grad_(True)
    raw.training_precision = precision
    ema = BranchParameterEMA.from_state_dict(raw, ema_state, expected_step=snapshot["step"],
        decay=plan["ema"]["decay"], base_state_sha256=plan["parent_model_state_sha256"])
    optimizer = torch.optim.Adam(raw.parameters(), lr=plan["config"]["lr"], foreach=False)
    optimizer.load_state_dict(resume["optimizer"])
    audit_live(raw, optimizer, snapshot["step"], dict(raw.named_buffers()))
    require(state_sha256(dict(raw.named_buffers())) == plan["fixed_buffers_sha256"], "Restored fixed buffers differ")
    using_cuda = next(raw.parameters()).is_cuda
    require(len(resume["cuda_rng"]) == (1 if using_cuda else 0), "Recovery RNG device inventory differs")
    random.setstate(resume["python_rng"])
    state = resume["numpy_rng"]
    np.random.set_state((state[0], state[1].numpy().astype(np.uint32), *state[2:]))
    torch.set_rng_state(resume["torch_rng"])
    if using_cuda:
        torch.cuda.set_rng_state_all(resume["cuda_rng"])
    return raw, optimizer, ema
