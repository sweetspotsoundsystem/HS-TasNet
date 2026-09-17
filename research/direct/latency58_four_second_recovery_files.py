"""Atomic publication of one packed raw/Adam/EMA recovery generation."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import read, require, sha, write
from research.direct.latency58_grouped_vocal_recovery import audit_snapshot, flush_directory
from research.direct.latency58_lossless_recovery_codec_v3 import (
    SCHEMA, MAX_FILE_BYTES, pack_snapshot, unpack_snapshot)

CURRENT = "recovery.packed.pt"
PENDING = "recovery.packed.pending.pt"
FINAL = "checkpoint.packed.pt"
RECEIPTS = "packed-recovery-receipts"


def identity(snapshot):
    return {"step": snapshot["step"], "journal_sha256": snapshot["journal_sha256"],
            "raw_model_state_sha256": snapshot["raw"]["model_state_sha256"],
            "ema_model_state_sha256": snapshot["average"]["model_state_sha256"],
            "ema_parameters_sha256": snapshot["ema_metadata"]["ema_parameters_sha256"],
            "planned_stop_step": snapshot["planned_stop_step"]}


def production_preflight(directory, plan, serialized_bytes):
    from research.direct.latency58_four_second_storage import ARTIFACT_ROOT, POLICY, snapshot
    directory = Path(directory).resolve(strict=True)
    require(directory.is_relative_to(ARTIFACT_ROOT) and str(directory) == plan["packed_publication_directory"]
            and plan["new_storage_policy_binding"] == {"path": str(POLICY), "sha256": sha(POLICY)}
            and type(serialized_bytes) is int and 0 < serialized_bytes <= MAX_FILE_BYTES,
            "Packed publication differs from the new root allocation")
    budget = snapshot()
    require(budget["new_root_actual_bytes"] + serialized_bytes <= budget["new_root_reserved_peak_bytes"],
            "Pending packed generation would exceed the allocated new root")
    return budget


def _publish_bytes(serialized, directory, endpoint, plan_sha, *, preflight, audit, before_replace=None):
    """Shared filesystem transaction; CPU fixtures supply their own reserve accounting."""
    directory = Path(directory)
    require(type(serialized) is bytes and 0 < len(serialized) <= MAX_FILE_BYTES
            and directory.is_dir() and not directory.is_symlink(), "Invalid packed publication")
    current, pending, receipts = directory / CURRENT, directory / PENDING, directory / RECEIPTS
    require(not pending.exists() and not pending.is_symlink() and not current.is_symlink()
            and not receipts.is_symlink() and not (directory / FINAL).exists(),
            "Preserve interrupted or finalized packed generation")
    previous = None
    if current.exists():
        digest = sha(current)
        matches = [read(path) for path in receipts.glob("step-*.json") if not path.is_symlink()]
        matches = [row for row in matches if row["sha256"] == digest]
        require(len(matches) == 1 and matches[0]["schema"] == SCHEMA
                and matches[0]["plan_sha256"] == plan_sha and matches[0]["step"] < endpoint["step"]
                and current.stat().st_size == matches[0]["bytes"], "Existing packed generation identity differs")
        previous = {key: matches[0][key] for key in ("step", "sha256", "bytes")}
    budget = preflight(len(serialized))
    require(audit(serialized) == endpoint, "Packed publication input audit differs")
    receipts.mkdir(exist_ok=True)
    receipt_path = receipts / f"step-{endpoint['step']:06d}.json"
    require(not receipt_path.exists() and not receipt_path.is_symlink(), "Preserve packed generation receipt")
    began = time.monotonic()
    with pending.open("xb") as stream:
        require(stream.write(serialized) == len(serialized), "Incomplete packed file write")
        stream.flush(); os.fsync(stream.fileno())
    digest = hashlib.sha256(serialized).hexdigest()
    require(pending.stat().st_size == len(serialized) and sha(pending) == digest
            and audit(pending.read_bytes()) == endpoint, "Persisted packed generation differs")
    receipt = {"schema": SCHEMA, "plan_sha256": plan_sha, **endpoint,
               "sha256": digest, "bytes": len(serialized), "previous": previous, "quality_claim": False}
    write(receipt_path, receipt)
    with receipt_path.open("rb") as stream:
        os.fsync(stream.fileno())
    flush_directory(receipts)
    if before_replace is not None:
        before_replace()
    os.replace(pending, current)
    flush_directory(directory)
    return {"path": str(current), "sha256": digest, "receipt": str(receipt_path),
            "receipt_sha256": sha(receipt_path), "step": endpoint["step"],
            "publication_seconds": time.monotonic() - began, "storage_preflight": budget}


def publish_snapshot(snapshot, directory, plan, plan_sha, *, before_replace=None):
    serialized, stats = pack_snapshot(snapshot, plan, plan_sha)
    binding = _publish_bytes(serialized, directory, identity(snapshot), plan_sha,
        preflight=lambda size: production_preflight(directory, plan, size),
        audit=lambda data: identity(unpack_snapshot(data, plan, plan_sha)[0]), before_replace=before_replace)
    return {**binding, "packing": stats}


def read_snapshot(binding, plan, plan_sha):
    path, receipt_path = Path(binding["path"]), Path(binding["receipt"])
    require(path.name in (CURRENT, FINAL) and path.is_file() and not path.is_symlink()
            and not path.parent.is_symlink() and not receipt_path.is_symlink()
            and not receipt_path.parent.is_symlink() and receipt_path.parent == path.parent / RECEIPTS
            and sha(receipt_path) == binding["receipt_sha256"], "Packed recovery receipt changed")
    receipt = read(receipt_path)
    require(receipt["schema"] == SCHEMA and receipt["plan_sha256"] == plan_sha
            and receipt["step"] == binding["step"] and path.stat().st_size == receipt["bytes"] <= MAX_FILE_BYTES
            and sha(path) == receipt["sha256"] == binding["sha256"], "Packed recovery file changed")
    snapshot, _ = unpack_snapshot(path.read_bytes(), plan, plan_sha)
    require(all(receipt[key] == value for key, value in identity(snapshot).items()),
            "Packed snapshot and receipt disagree")
    return snapshot, audit_snapshot(snapshot, plan, plan_sha)


def finalize(binding, plan, plan_sha):
    """Seal the completed rolling file by rename; no second tensor archive."""
    snapshot, audited = read_snapshot(binding, plan, plan_sha)
    require(snapshot["step"] == plan["config"]["steps"], "Cannot finalize an incomplete training schedule")
    del audited, snapshot
    current = Path(binding["path"])
    final = current.parent / FINAL
    require(current.name == CURRENT and not final.exists() and not final.is_symlink()
            and not (current.parent / PENDING).exists(), "Preserve existing packed finalization")
    current.rename(final)
    flush_directory(final.parent)
    return {**binding, "path": str(final)}


def load_inference(binding, plan, plan_sha, *, role):
    from research.direct.latency58_branch_memory_checkpoint import load_payload
    require(role in ("raw", "ema"), "Packed inference role must be raw or ema")
    snapshot, audited = read_snapshot(binding, plan, plan_sha)
    del audited
    payload = snapshot["raw" if role == "raw" else "average"]
    model, payload = load_payload(payload)
    require(model.algorithmic_latency_samples == 256, "Packed inference latency differs")
    return model, payload
