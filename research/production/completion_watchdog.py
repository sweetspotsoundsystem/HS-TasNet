#!/usr/bin/env python3
"""Recover incomplete final publication without modifying the sealed trainer."""

from __future__ import annotations

import hashlib
import contextlib
import fcntl
import json
import os
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


BASE = Path(__file__).resolve().parent
RUN_DIR = BASE / "runs/c91-full-best200-v1-seed60"
WATCHDOG_LOCK = BASE / ".completion-watchdog.lock"
AUTORESTART_SERVICE = "hs-tasnet-c91-full-v1-autoresume.service"
PRIMARY_SERVICE = "hs-tasnet-c91-full-v1.service"
SERVICES = (PRIMARY_SERVICE, AUTORESTART_SERVICE)
ACTIVE_STATES = {"active", "activating", "reloading", "deactivating"}
EXPECTED_STEP = 300_000
STEP250 = 250_000
EXPECTED_CONTRACT = "787724005bef19d0d0cc00c58b37be1138967390581a5902cf9ed46134330514"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
TEMPORARY_PATTERN = re.compile(
    r"^\..+\.(?P<pid>[1-9][0-9]*)(?:\.[0-9a-f]{32})?\.tmp(?:\.sha256)?$"
)
CHECKPOINT_GENERATION_PATTERN = re.compile(
    r"^step-(?P<step>[0-9]{12})-[0-9a-f]{32}\.pt$"
)
VERIFIED_CHECKPOINT_EVENTS = {
    "initial_checkpoint_verified",
    "checkpoint_verified",
    "emergency_checkpoint_verified",
    "pause_checkpoint_verified",
}


class PublicationError(RuntimeError):
    pass


class PublicationContradiction(PublicationError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicationContradiction(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    content = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def strict_json_bytes(content: bytes, context: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key in {context}: {key}")
            result[key] = value
        return result

    def no_nonfinite(token: str) -> None:
        raise PublicationContradiction(f"non-finite JSON constant in {context}: {token}")

    try:
        value = json.loads(content, object_pairs_hook=no_duplicates, parse_constant=no_nonfinite)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PublicationContradiction(f"malformed JSON: {context}") from error
    require(isinstance(value, dict), f"JSON is not an object: {context}")
    return value


def strict_json(path: Path) -> tuple[dict[str, Any], bytes]:
    require(path.is_file() and not path.is_symlink(), f"missing/symlink JSON: {path}")
    content = path.read_bytes()
    return strict_json_bytes(content, str(path)), content


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    require(path.is_file() and not path.is_symlink(), f"event log is not a regular file: {path}")
    records: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_bytes().splitlines(keepends=True), 1):
        require(line.endswith(b"\n"), f"event log has an incomplete trailing record at line {index}")
        records.append(strict_json_bytes(line, f"{path}:{index}"))
    return records


def verify_sidecar(path: Path, expected: str) -> None:
    require(SHA256_PATTERN.fullmatch(expected) is not None, f"invalid expected hash for {path}")
    require(path.is_file() and not path.is_symlink(), f"missing/symlink artifact: {path}")
    require(sha256_file(path) == expected, f"artifact hash mismatch: {path}")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    require(sidecar.is_file() and not sidecar.is_symlink(), f"missing/symlink sidecar: {sidecar}")
    require(sidecar.read_text(encoding="ascii").split() == [expected, path.name], f"sidecar mismatch: {path}")


def resolve_report_artifact(run_dir: Path, value: Any, expected_name: str) -> tuple[Path, dict[str, Any]]:
    require(isinstance(value, dict), f"final report artifact is malformed: {expected_name}")
    path = Path(str(value.get("path", "")))
    require(path.resolve() == (run_dir / expected_name).resolve(), f"final report artifact path changed: {expected_name}")
    verify_sidecar(path, str(value.get("sha256", "")))
    require(SHA256_PATTERN.fullmatch(str(value.get("model_state_sha256", ""))) is not None, f"invalid model-state hash: {expected_name}")
    return path, value


def verify_terminal_artifacts(run_dir: Path, report: dict[str, Any]) -> None:
    _, raw = resolve_report_artifact(run_dir, report.get("raw_artifact"), "final-training-raw.pt")
    deployment_path, deployment = resolve_report_artifact(run_dir, report.get("deployment"), "final-deployment.pt")
    require(raw.get("model_state_sha256") != deployment.get("model_state_sha256"), "raw and deployment states unexpectedly coincide")
    metadata, _ = strict_json(deployment_path.with_suffix(deployment_path.suffix + ".json"))
    require(metadata == deployment, "deployment metadata disagrees with final report")

    pointer, _ = strict_json(run_dir / "checkpoints/latest.json")
    require(pointer.get("kind") == "hs_tasnet_c91_checkpoint_pointer", "final checkpoint pointer kind changed")
    require(pointer.get("step") == EXPECTED_STEP, "final checkpoint pointer is not step300000")
    generation = pointer.get("generation")
    require(isinstance(generation, str) and Path(generation).name == generation, "final checkpoint generation is invalid")
    require(pointer.get("path") == f"checkpoints/{generation}", "final checkpoint relative path changed")
    verify_sidecar(run_dir / "checkpoints" / generation, str(pointer.get("sha256", "")))
    require(SHA256_PATTERN.fullmatch(str(pointer.get("model_state_sha256", ""))) is not None, "final checkpoint model-state hash invalid")
    require(pointer.get("model_state_sha256") == raw.get("model_state_sha256"), "final raw/checkpoint model states disagree")


def validate_audit_receipt(
    path: Path,
    run_dir: Path,
    report: dict[str, Any],
    publication: dict[str, Any],
) -> dict[str, Any]:
    receipt, _ = strict_json(path)
    require(
        set(receipt)
        == {
            "schema_version",
            "kind",
            "audit_status",
            "audited_at_utc",
            "auditor",
            "service",
            "run",
            "completion",
            "artifacts",
            "step250000_diagnostic_prerequisite",
            "functional_verification",
            "filesystem",
            "audit_payload_sha256",
        },
        "final audit receipt fields changed",
    )
    require(receipt.get("schema_version") == 1, "final audit receipt schema changed")
    require(receipt.get("kind") == "hs_tasnet_c91_final_audit_receipt", "final audit receipt kind changed")
    require(receipt.get("audit_status") == "pass", "final audit receipt did not pass")
    digest = receipt.get("audit_payload_sha256")
    require(SHA256_PATTERN.fullmatch(str(digest)) is not None, "final audit receipt payload hash is invalid")
    unhashed = dict(receipt)
    unhashed.pop("audit_payload_sha256")
    require(canonical_sha256(unhashed) == digest, "final audit receipt payload hash mismatch")

    auditor = receipt.get("auditor")
    require(isinstance(auditor, dict), "final audit receipt auditor evidence is malformed")
    require(auditor.get("cuda_visible_devices") == "", "final audit receipt exposed CUDA")
    require(auditor.get("torch_cuda_available") is False, "final audit receipt reports CUDA available")
    require(auditor.get("torch_cuda_initialized") is False, "final audit receipt reports CUDA initialized")
    service = receipt.get("service")
    require(isinstance(service, dict) and service.get("name") == AUTORESTART_SERVICE, "final audit receipt service changed")
    run = receipt.get("run")
    require(isinstance(run, dict), "final audit receipt run evidence is malformed")
    require(run.get("contract_identity_sha256") == EXPECTED_CONTRACT, "final audit receipt contract changed")

    completion = receipt.get("completion")
    require(isinstance(completion, dict), "final audit receipt completion evidence is malformed")
    require(completion.get("status") == "complete", "final audit receipt completion status changed")
    require(completion.get("step") == EXPECTED_STEP and completion.get("total_steps") == EXPECTED_STEP, "final audit receipt completion step changed")
    require(completion.get("final_report_sha256") == publication["final_report_sha256"], "final audit receipt report hash changed")
    require(completion.get("status_sha256") == sha256_file(run_dir / "status.json"), "final audit receipt status hash changed")
    require(completion.get("events_sha256") == publication["events_sha256"], "final audit receipt events hash changed")
    require(completion.get("peak_vram_pass") is True, "final audit receipt VRAM gate did not pass")
    require(completion.get("peak_vram_bytes") == report.get("peak_vram_bytes"), "final audit receipt peak VRAM changed")
    require(
        completion.get("maximum_peak_vram_bytes") == report.get("maximum_peak_vram_bytes"),
        "final audit receipt VRAM ceiling changed",
    )

    artifacts = receipt.get("artifacts")
    require(isinstance(artifacts, dict), "final audit receipt artifacts are malformed")
    raw = artifacts.get("final_training_raw")
    deployment = artifacts.get("final_deployment")
    checkpoint = artifacts.get("final_checkpoint")
    pointer, _ = strict_json(run_dir / "checkpoints/latest.json")
    for label, recorded, expected in (
        ("raw", raw, report.get("raw_artifact")),
        ("deployment", deployment, report.get("deployment")),
        ("checkpoint", checkpoint, pointer),
    ):
        require(isinstance(recorded, dict) and isinstance(expected, dict), f"final audit receipt {label} evidence is malformed")
        require(recorded.get("sha256") == expected.get("sha256"), f"final audit receipt {label} hash changed")
        require(
            recorded.get("model_state_sha256") == expected.get("model_state_sha256"),
            f"final audit receipt {label} model-state hash changed",
        )

    step250 = receipt.get("step250000_diagnostic_prerequisite")
    require(isinstance(step250, dict) and step250.get("status") == "validated", "final audit receipt lacks validated step250 evidence")
    require(step250.get("step") == 250_000, "final audit receipt step250 counter changed")
    outcome = step250.get("canonical_outcome")
    require(isinstance(outcome, dict) and outcome.get("exactly_one_sealed") is True, "final audit receipt step250 outcome is not unique")
    require(SHA256_PATTERN.fullmatch(str(outcome.get("sha256", ""))) is not None, "final audit receipt step250 outcome hash is invalid")

    functional = receipt.get("functional_verification")
    require(isinstance(functional, dict), "final audit receipt functional evidence is malformed")
    transformation = functional.get("transformation")
    streaming = functional.get("streaming")
    require(
        isinstance(transformation, dict)
        and transformation.get("post_optimizer_only") is True
        and transformation.get("decoder_weight_equals_raw_times_hann_exact") is True,
        "final audit receipt transformation evidence failed",
    )
    require(
        isinstance(streaming, dict)
        and streaming.get("device") == "cpu"
        and streaming.get("finite") is True
        and streaming.get("reset_determinism_exact") is True,
        "final audit receipt streaming evidence failed",
    )
    filesystem = receipt.get("filesystem")
    require(isinstance(filesystem, dict) and filesystem.get("temporary_files") == [], "final audit receipt retained temporary files")
    return receipt


def pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def remove_stale_temporary_files(run_dir: Path) -> list[str]:
    removed: list[str] = []
    synced_directories: set[Path] = set()
    if not run_dir.exists():
        return removed
    for path in sorted(run_dir.rglob("*")):
        match = TEMPORARY_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        require(path.is_file() and not path.is_symlink(), f"temporary path is not a regular file: {path}")
        owner_pid = int(match.group("pid"))
        require(not pid_is_alive(owner_pid), f"temporary file owner PID is still alive: {path}")
        relative = str(path.relative_to(run_dir))
        path.unlink()
        removed.append(relative)
        synced_directories.add(path.parent)
    for directory in sorted(synced_directories):
        fsync_directory(directory)
    return removed


def atomic_replace_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def recover_missing_pointer_checkpoint_event(
    run_dir: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    pointer_path = run_dir / "checkpoints/latest.json"
    if not pointer_path.exists():
        return None
    pointer, _ = strict_json(pointer_path)
    require(pointer.get("schema_version") == 1, "checkpoint pointer schema changed")
    require(pointer.get("kind") == "hs_tasnet_c91_checkpoint_pointer", "checkpoint pointer kind changed")
    step = pointer.get("step")
    generation = pointer.get("generation")
    relative = pointer.get("path")
    require(isinstance(step, int) and not isinstance(step, bool) and step >= 0, "checkpoint pointer step is invalid")
    require(isinstance(generation, str) and Path(generation).name == generation, "checkpoint pointer generation is invalid")
    require(relative == f"checkpoints/{generation}", "checkpoint pointer relative path changed")
    artifact = run_dir / "checkpoints" / generation
    expected_hash = str(pointer.get("sha256", ""))
    verify_sidecar(artifact, expected_hash)
    model_hash = str(pointer.get("model_state_sha256", ""))
    require(SHA256_PATTERN.fullmatch(model_hash) is not None, "checkpoint pointer model-state hash is invalid")
    verified = [record for record in records if record.get("event") in VERIFIED_CHECKPOINT_EVENTS]
    same_step = [record for record in verified if record.get("step") == step]
    if same_step:
        if len(same_step) == 2:
            require(step == STEP250, f"multiple verified checkpoint events already exist at step {step}")
            indexed = [
                (index, record)
                for index, record in enumerate(records)
                if record.get("event") in VERIFIED_CHECKPOINT_EVENTS and record.get("step") == step
            ]
            emergency = [item for item in indexed if item[1].get("event") == "emergency_checkpoint_verified"]
            normal = [item for item in indexed if item[1].get("event") == "checkpoint_verified"]
            require(
                len(emergency) == len(normal) == 1 and emergency[0][0] < normal[0][0],
                "step250000 duplicate checkpoint events are not an emergency replay pair",
            )
            emergency_index, emergency_record = emergency[0]
            normal_index, existing = normal[0]
            require(
                emergency_index > 0
                and records[emergency_index - 1].get("event") == "training_exception"
                and records[emergency_index - 1].get("step") == STEP250,
                "step250000 replay emergency lacks its adjacent exception",
            )
            replay_resumes = [
                record
                for record in records[emergency_index + 1 : normal_index]
                if record.get("event") == "resumed"
                and isinstance(record.get("step"), int)
                and not isinstance(record.get("step"), bool)
                and 0 <= record["step"] < STEP250
                and SHA256_PATTERN.fullmatch(str(record.get("model_state_sha256", ""))) is not None
            ]
            require(replay_resumes, "step250000 emergency checkpoint was not replayed from an earlier verified state")
            emergency_path = Path(str(emergency_record.get("path", "")))
            emergency_match = CHECKPOINT_GENERATION_PATTERN.fullmatch(emergency_path.name)
            require(
                emergency_path.is_absolute()
                and emergency_path.parent.resolve() == artifact.parent.resolve()
                and emergency_match is not None
                and int(emergency_match.group("step")) == STEP250
                and emergency_path.resolve() != artifact.resolve()
                and Path(str(emergency_record.get("pointer", ""))).resolve() == pointer_path.resolve()
                and SHA256_PATTERN.fullmatch(str(emergency_record.get("sha256", ""))) is not None
                and SHA256_PATTERN.fullmatch(str(emergency_record.get("model_state_sha256", ""))) is not None,
                "step250000 emergency replay evidence is malformed",
            )
        else:
            require(len(same_step) == 1, f"multiple verified checkpoint events already exist at step {step}")
            existing = same_step[0]
        require(existing.get("path") == str(artifact), "existing checkpoint event path disagrees with pointer")
        require(existing.get("pointer") == str(pointer_path), "existing checkpoint event pointer path disagrees")
        require(existing.get("sha256") == expected_hash, "existing checkpoint event hash disagrees with pointer")
        require(existing.get("model_state_sha256") == model_hash, "existing checkpoint event model hash disagrees")
        return None
    event = "checkpoint_verified"
    if records and records[-1].get("event") == "training_exception":
        exception_step = records[-1].get("step")
        require(
            isinstance(exception_step, int) and not isinstance(exception_step, bool) and exception_step >= 0,
            "training exception preceding a missing checkpoint event has an invalid step",
        )
        require(
            exception_step == step,
            "checkpoint pointer does not match the preceding training exception step",
        )
        event = "emergency_checkpoint_verified"
    prior_steps = [record.get("step") for record in verified]
    require(
        all(isinstance(value, int) and not isinstance(value, bool) and value < step for value in prior_steps),
        "checkpoint pointer regressed behind existing verified events",
    )
    return {
        "event": event,
        "time": datetime.now(timezone.utc).isoformat(),
        "path": str(artifact),
        "sha256": expected_hash,
        "step": step,
        "model_state_sha256": model_hash,
        "pointer": str(pointer_path),
        "reconstructed_by_completion_watchdog": True,
    }


def reconcile_step250_emergency_checkpoint(
    run_dir: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Replay step 250k when an exception bypassed its required native snapshot."""

    emergencies = [
        (index, record)
        for index, record in enumerate(records)
        if record.get("event") == "emergency_checkpoint_verified" and record.get("step") == STEP250
    ]
    if not emergencies:
        return None
    require(len(emergencies) == 1, "multiple emergency checkpoint events exist at step250000")
    emergency_index, emergency = emergencies[0]
    normal_indices = [
        index
        for index, record in enumerate(records)
        if record.get("event") == "checkpoint_verified" and record.get("step") == STEP250
    ]
    if any(index > emergency_index for index in normal_indices):
        return None
    require(not normal_indices, "a normal step250000 checkpoint precedes its emergency checkpoint")
    snapshots = [
        record
        for record in records
        if record.get("event") == "snapshot_verified" and record.get("step") == STEP250
    ]
    require(not snapshots, "step250000 emergency checkpoint unexpectedly coexists with a native snapshot")
    require(
        emergency_index > 0
        and records[emergency_index - 1].get("event") == "training_exception"
        and records[emergency_index - 1].get("step") == STEP250,
        "step250000 emergency checkpoint is not adjacent to its training exception",
    )
    retry_records = records[emergency_index + 1 :]
    require(
        all(
            record.get("event")
            in {"inventory_verified", "run_opened", "resume_candidates_rejected", "resumed"}
            for record in retry_records
        ),
        "training advanced after the unreplayed step250000 emergency checkpoint",
    )
    retry_resumes = [record for record in retry_records if record.get("event") == "resumed"]
    require(len(retry_resumes) <= 1, "multiple resumes followed the unreplayed step250000 emergency checkpoint")

    prior = [
        record
        for record in records[:emergency_index]
        if record.get("event") == "checkpoint_verified" and record.get("step") == STEP250 - 1_000
    ]
    require(len(prior) == 1, "step250000 emergency rollback has no unique step249000 anchor")
    anchor = prior[0]
    checkpoint_dir = run_dir / "checkpoints"
    pointer_path = checkpoint_dir / "latest.json"

    def bound_event_artifact(record: dict[str, Any], step: int, label: str) -> tuple[Path, str, str]:
        path = Path(str(record.get("path", "")))
        require(path.parent.resolve() == checkpoint_dir.resolve(), f"{label} checkpoint path escaped its directory")
        match = CHECKPOINT_GENERATION_PATTERN.fullmatch(path.name)
        require(match is not None and int(match.group("step")) == step, f"{label} checkpoint filename changed")
        require(Path(str(record.get("pointer", ""))).resolve() == pointer_path.resolve(), f"{label} pointer path changed")
        digest = str(record.get("sha256", ""))
        model_hash = str(record.get("model_state_sha256", ""))
        require(SHA256_PATTERN.fullmatch(digest) is not None, f"{label} checkpoint hash is invalid")
        require(SHA256_PATTERN.fullmatch(model_hash) is not None, f"{label} model-state hash is invalid")
        return path, digest, model_hash

    anchor_path, anchor_hash, anchor_model_hash = bound_event_artifact(anchor, STEP250 - 1_000, "anchor")
    emergency_path, emergency_hash, emergency_model_hash = bound_event_artifact(
        emergency, STEP250, "emergency"
    )
    if retry_resumes:
        retry_resume = retry_resumes[0]
        require(retry_resume.get("step") == STEP250 - 1_000, "step250000 emergency retry resumed from the wrong step")
        require(
            retry_resume.get("model_state_sha256") == anchor_model_hash,
            "step250000 emergency retry resume does not match the verified anchor",
        )
    verify_sidecar(anchor_path, anchor_hash)
    pointer, _ = strict_json(pointer_path)
    require(pointer.get("schema_version") == 1, "checkpoint pointer schema changed")
    require(pointer.get("kind") == "hs_tasnet_c91_checkpoint_pointer", "checkpoint pointer kind changed")
    pointer_matches_emergency = (
        pointer.get("step") == STEP250
        and pointer.get("generation") == emergency_path.name
        and pointer.get("path") == f"checkpoints/{emergency_path.name}"
        and pointer.get("sha256") == emergency_hash
        and pointer.get("model_state_sha256") == emergency_model_hash
    )
    pointer_matches_anchor = (
        pointer.get("step") == STEP250 - 1_000
        and pointer.get("generation") == anchor_path.name
        and pointer.get("path") == f"checkpoints/{anchor_path.name}"
        and pointer.get("sha256") == anchor_hash
        and pointer.get("model_state_sha256") == anchor_model_hash
    )
    require(pointer_matches_emergency or pointer_matches_anchor, "checkpoint pointer matches neither emergency nor rollback anchor")
    restored_pointer = False
    if pointer_matches_emergency:
        verify_sidecar(emergency_path, emergency_hash)
        atomic_replace_bytes(
            pointer_path,
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "hs_tasnet_c91_checkpoint_pointer",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "generation": anchor_path.name,
                    "path": f"checkpoints/{anchor_path.name}",
                    "sha256": anchor_hash,
                    "step": STEP250 - 1_000,
                    "model_state_sha256": anchor_model_hash,
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8") + b"\n",
        )
        restored_pointer = True

    removed: list[str] = []
    emergency_sidecar = emergency_path.with_suffix(emergency_path.suffix + ".sha256")
    if emergency_sidecar.exists() or emergency_sidecar.is_symlink():
        require(emergency_sidecar.is_file() and not emergency_sidecar.is_symlink(), "emergency sidecar is invalid")
        emergency_sidecar.unlink()
        removed.append(str(emergency_sidecar.relative_to(run_dir)))
    if emergency_path.exists() or emergency_path.is_symlink():
        require(emergency_path.is_file() and not emergency_path.is_symlink(), "emergency checkpoint is invalid")
        emergency_path.unlink()
        removed.append(str(emergency_path.relative_to(run_dir)))
    if removed:
        fsync_directory(checkpoint_dir)
    return {
        "action": "rolled_back_step250000_emergency_checkpoint",
        "restored_pointer": restored_pointer,
        "anchor_step": STEP250 - 1_000,
        "anchor_generation": anchor_path.name,
        "removed": removed,
    }


def reconcile_unpublished_checkpoint_generations(
    run_dir: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Roll back sealed generations published before their pointer transaction."""

    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None
    require(
        checkpoint_dir.is_dir() and not checkpoint_dir.is_symlink(),
        f"checkpoint directory is invalid: {checkpoint_dir}",
    )
    candidates: dict[str, Path] = {}
    for item in checkpoint_dir.iterdir():
        name = item.name[:-7] if item.name.endswith(".sha256") else item.name
        if CHECKPOINT_GENERATION_PATTERN.fullmatch(name) is None:
            continue
        require(not item.is_symlink(), f"checkpoint generation member is a symlink: {item}")
        candidates[name] = checkpoint_dir / name

    pointer_path = checkpoint_dir / "latest.json"
    if not pointer_path.exists():
        complete = [
            path
            for path in candidates.values()
            if path.is_file() and path.with_suffix(path.suffix + ".sha256").is_file()
        ]
        require(not complete, "sealed checkpoint generations exist without a checkpoint pointer")
        return None

    pointer, _ = strict_json(pointer_path)
    require(pointer.get("schema_version") == 1, "checkpoint pointer schema changed")
    require(pointer.get("kind") == "hs_tasnet_c91_checkpoint_pointer", "checkpoint pointer kind changed")
    pointer_step = pointer.get("step")
    pointer_generation = pointer.get("generation")
    require(
        isinstance(pointer_step, int) and not isinstance(pointer_step, bool) and 0 <= pointer_step <= EXPECTED_STEP,
        "checkpoint pointer step is invalid",
    )
    require(
        isinstance(pointer_generation, str)
        and CHECKPOINT_GENERATION_PATTERN.fullmatch(pointer_generation) is not None,
        "checkpoint pointer generation is invalid",
    )
    generation_match = CHECKPOINT_GENERATION_PATTERN.fullmatch(pointer_generation)
    assert generation_match is not None
    require(int(generation_match.group("step")) == pointer_step, "checkpoint pointer filename step changed")
    require(pointer.get("path") == f"checkpoints/{pointer_generation}", "checkpoint pointer relative path changed")
    pointer_artifact = checkpoint_dir / pointer_generation
    verify_sidecar(pointer_artifact, str(pointer.get("sha256", "")))
    require(
        SHA256_PATTERN.fullmatch(str(pointer.get("model_state_sha256", ""))) is not None,
        "checkpoint pointer model-state hash is invalid",
    )

    unpublished: list[tuple[int, Path, Path]] = []
    for path in sorted(candidates.values()):
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if not (path.exists() and sidecar.exists()):
            continue
        require(path.is_file() and not path.is_symlink(), f"checkpoint generation is invalid: {path}")
        require(sidecar.is_file() and not sidecar.is_symlink(), f"checkpoint sidecar is invalid: {sidecar}")
        fields = sidecar.read_text(encoding="ascii").split()
        require(
            len(fields) == 2
            and SHA256_PATTERN.fullmatch(fields[0]) is not None
            and fields[1] == path.name,
            f"checkpoint generation sidecar is malformed: {sidecar}",
        )
        require(sha256_file(path) == fields[0], f"checkpoint generation hash mismatch: {path}")
        match = CHECKPOINT_GENERATION_PATTERN.fullmatch(path.name)
        assert match is not None
        step = int(match.group("step"))
        if path == pointer_artifact:
            continue
        require(step != pointer_step, "alternate sealed checkpoint generation exists at the pointer step")
        if step > pointer_step:
            require(step <= EXPECTED_STEP, "checkpoint generation advances beyond the configured run")
            unpublished.append((step, path, sidecar))

    if not unpublished:
        return None
    unpublished_steps = [step for step, _path, _sidecar in unpublished]
    require(
        len(unpublished_steps) == len(set(unpublished_steps)),
        "multiple unpublished checkpoint generations exist at the same step",
    )
    require(
        not any(record.get("event") == "run_complete" for record in records),
        "terminal completion evidence exists during checkpoint-generation rollback",
    )
    terminal_paths = (
        "final_report.json",
        "final-training-raw.pt",
        "final-deployment.pt",
        "final_audit_receipt.json",
    )
    require(
        not any((run_dir / name).exists() or (run_dir / name).is_symlink() for name in terminal_paths),
        "terminal publication artifacts exist during checkpoint-generation rollback",
    )
    status_path = run_dir / "status.json"
    if status_path.exists() or status_path.is_symlink():
        status, _ = strict_json(status_path)
        require(
            set(status)
            == {
                "status",
                "step",
                "total_steps",
                "peak_vram_bytes",
                "contract_identity_sha256",
            },
            "nonterminal status fields changed during checkpoint-generation rollback",
        )
        status_step = status.get("step")
        require(status.get("status") == "paused", "terminal or contradictory status exists during checkpoint rollback")
        require(
            isinstance(status_step, int)
            and not isinstance(status_step, bool)
            and 0 <= status_step <= pointer_step,
            "paused status advances beyond the checkpoint pointer",
        )
        require(status.get("total_steps") == EXPECTED_STEP, "paused status total step changed")
        require(
            isinstance(status.get("peak_vram_bytes"), int)
            and not isinstance(status.get("peak_vram_bytes"), bool)
            and status["peak_vram_bytes"] >= 0,
            "paused status peak VRAM is invalid",
        )
        require(status.get("contract_identity_sha256") == EXPECTED_CONTRACT, "paused status contract changed")
        matching_pauses = []
        for record in records:
            if record.get("event") != "run_paused":
                continue
            payload = dict(record)
            payload.pop("event", None)
            payload.pop("time", None)
            if payload == status:
                matching_pauses.append(record)
        require(len(matching_pauses) == 1, "paused status has no unique matching run_paused event")
    verified_events = [record for record in records if record.get("event") in VERIFIED_CHECKPOINT_EVENTS]
    event_bindings: list[tuple[Path, str, int]] = []
    for record in verified_events:
        event_step = record.get("step")
        require(
            isinstance(event_step, int) and not isinstance(event_step, bool) and 0 <= event_step <= EXPECTED_STEP,
            "verified checkpoint event has an invalid step",
        )
        require(event_step <= pointer_step, "verified checkpoint events advance beyond the checkpoint pointer")
        event_path = Path(str(record.get("path", "")))
        require(
            event_path.parent.resolve() == checkpoint_dir.resolve(),
            "verified checkpoint event path escaped its directory",
        )
        event_match = CHECKPOINT_GENERATION_PATTERN.fullmatch(event_path.name)
        require(
            event_match is not None and int(event_match.group("step")) == event_step,
            "verified checkpoint event filename/step binding changed",
        )
        require(
            Path(str(record.get("pointer", ""))).resolve() == pointer_path.resolve(),
            "verified checkpoint event pointer path changed",
        )
        event_hash = str(record.get("sha256", ""))
        event_model_hash = str(record.get("model_state_sha256", ""))
        require(SHA256_PATTERN.fullmatch(event_hash) is not None, "verified checkpoint event hash is invalid")
        require(
            SHA256_PATTERN.fullmatch(event_model_hash) is not None,
            "verified checkpoint event model-state hash is invalid",
        )
        event_bindings.append((event_path.resolve(), event_hash, event_step))
    for step, path, sidecar in unpublished:
        digest = sidecar.read_text(encoding="ascii").split()[0]
        require(
            not any(bound_path == path.resolve() or bound_hash == digest for bound_path, bound_hash, _ in event_bindings),
            f"unpublished checkpoint generation is bound by a verified event: {path}",
        )
    removed: list[str] = []
    for _step, path, sidecar in sorted(unpublished, reverse=True):
        sidecar.unlink()
        removed.append(str(sidecar.relative_to(run_dir)))
        path.unlink()
        removed.append(str(path.relative_to(run_dir)))
    fsync_directory(checkpoint_dir)
    return {
        "action": "rolled_back_unpublished_checkpoint_generations",
        "pointer_step": pointer_step,
        "pointer_generation": pointer_generation,
        "removed": removed,
        "maximum_removed_step": max(step for step, _path, _sidecar in unpublished),
    }


def reconcile_incomplete_event_tail(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "events.jsonl"
    present = path.exists() or path.is_symlink()
    if present:
        require(path.is_file() and not path.is_symlink(), f"event log is not a regular file: {path}")
        content = path.read_bytes()
    else:
        content = b""

    action: str | None = None
    affected_bytes = 0
    recovered_event: Any = None
    if content and not content.endswith(b"\n"):
        split = content.rfind(b"\n") + 1
        prefix = content[:split]
        tail = content[split:]
        records: list[dict[str, Any]] = []
        for index, line in enumerate(prefix.splitlines(keepends=True), 1):
            require(line.endswith(b"\n"), f"event prefix has an incomplete record at line {index}")
            records.append(strict_json_bytes(line, f"{path}:{index}"))
        require(
            not any(record.get("event") == "run_complete" for record in records),
            "refusing to reconcile an event tail after a complete run",
        )
        try:
            recovered = strict_json_bytes(tail, f"{path}:unterminated-tail")
        except PublicationContradiction:
            replacement = prefix
            action = "truncated_invalid_unterminated_tail"
        else:
            records.append(recovered)
            replacement = content + b"\n"
            action = "completed_valid_unterminated_tail"
            recovered_event = recovered.get("event")
        affected_bytes = len(tail)
    else:
        replacement = content
        records = read_events(path) if present and content else []

    recovered_checkpoint: dict[str, Any] | None = None
    if not any(record.get("event") == "run_complete" for record in records):
        recovered_checkpoint = recover_missing_pointer_checkpoint_event(run_dir, records)
        if recovered_checkpoint is not None:
            replacement += json.dumps(
                recovered_checkpoint,
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8") + b"\n"
            if action is None:
                action = "reconstructed_missing_pointer_checkpoint_event"
    if action is None:
        return None

    atomic_replace_bytes(path, replacement)
    read_events(path)
    return {
        "action": action,
        "discarded_or_completed_bytes": affected_bytes,
        "recovered_event": recovered_event,
        "reconstructed_checkpoint_step": (
            recovered_checkpoint["step"] if recovered_checkpoint is not None else None
        ),
        "events_sha256": sha256_file(path),
    }


def reconcile_crash_leftovers(run_dir: Path) -> dict[str, Any]:
    removed = remove_stale_temporary_files(run_dir)
    event_tail = reconcile_incomplete_event_tail(run_dir) if run_dir.exists() else None
    records = read_events(run_dir / "events.jsonl") if run_dir.exists() else []
    step250_emergency = (
        reconcile_step250_emergency_checkpoint(run_dir, records) if run_dir.exists() else None
    )
    checkpoint_generations = (
        reconcile_unpublished_checkpoint_generations(run_dir, records) if run_dir.exists() else None
    )
    return {
        "removed_temporary_files": removed,
        "event_tail": event_tail,
        "step250_emergency": step250_emergency,
        "checkpoint_generations": checkpoint_generations,
    }


def publication_state(run_dir: Path = RUN_DIR) -> dict[str, Any]:
    receipt = run_dir / "final_audit_receipt.json"
    receipt_present = receipt.exists() or receipt.is_symlink()
    if receipt_present:
        require(receipt.is_file() and not receipt.is_symlink(), f"final audit receipt is missing/symlink: {receipt}")

    events_path = run_dir / "events.jsonl"
    events = read_events(events_path)
    completions = [(index, record) for index, record in enumerate(events) if record.get("event") == "run_complete"]
    require(len(completions) <= 1, "event log has duplicate run_complete records")
    if not completions:
        require(not receipt_present, "final audit receipt exists without a run_complete event")
        return {
            "status": "incomplete_recoverable",
            "step": EXPECTED_STEP,
            "event_count": len(events),
            "final_report_present": (run_dir / "final_report.json").is_file(),
            "status_present": (run_dir / "status.json").is_file(),
        }

    completion_index, completion = completions[0]
    require(completion_index == len(events) - 1, "run_complete is not the final event")
    report, report_bytes = strict_json(run_dir / "final_report.json")
    status, status_bytes = strict_json(run_dir / "status.json")
    require(report_bytes == status_bytes and report == status, "final report and status disagree")
    require(report.get("status") == "complete", "final report status is not complete")
    require(report.get("step") == EXPECTED_STEP and report.get("total_steps") == EXPECTED_STEP, "final report step changed")
    require(report.get("contract_identity_sha256") == EXPECTED_CONTRACT, "final report contract identity changed")
    require(report.get("peak_vram_pass") is True, "final report VRAM gate did not pass")
    peak = report.get("peak_vram_bytes")
    ceiling = report.get("maximum_peak_vram_bytes")
    require(isinstance(peak, int) and not isinstance(peak, bool) and peak > 0, "final report peak VRAM invalid")
    require(isinstance(ceiling, int) and not isinstance(ceiling, bool) and peak <= ceiling, "final report VRAM ceiling invalid")
    event_payload = dict(completion)
    event_payload.pop("event", None)
    event_payload.pop("time", None)
    require(event_payload == report, "run_complete payload disagrees with final report")
    verify_terminal_artifacts(run_dir, report)
    publication = {
        "status": "publication_complete",
        "step": EXPECTED_STEP,
        "completion_event_index": completion_index,
        "events_sha256": sha256_file(events_path),
        "final_report_sha256": sha256_file(run_dir / "final_report.json"),
    }
    if receipt_present:
        validated_receipt = validate_audit_receipt(receipt, run_dir, report, publication)
        return {
            "status": "audited_complete",
            "step": EXPECTED_STEP,
            "receipt": str(receipt),
            "receipt_sha256": sha256_file(receipt),
            "audit_payload_sha256": validated_receipt["audit_payload_sha256"],
            "publication": publication,
        }
    return publication


def service_properties(unit: str) -> dict[str, str]:
    completed = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "--property",
            "LoadState",
            "--property",
            "ActiveState",
            "--property",
            "SubState",
            "--property",
            "MainPID",
            "--property",
            "InvocationID",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise PublicationError(f"cannot inspect {unit}: {completed.stderr.strip()}")
    return dict(line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line)


def start_autoresume() -> dict[str, str]:
    completed = subprocess.run(
        ["systemctl", "--user", "start", AUTORESTART_SERVICE],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise PublicationError(f"cannot start {AUTORESTART_SERVICE}: {completed.stderr.strip()}")
    state = service_properties(AUTORESTART_SERVICE)
    require(state.get("LoadState") == "loaded", "autoresume service is not loaded")
    require(state.get("ActiveState") == "active" and state.get("SubState") == "running", "autoresume service did not become active")
    require(str(state.get("MainPID", "")).isdigit() and int(state["MainPID"]) > 0, "autoresume service has no PID")
    require(re.fullmatch(r"[0-9a-f]{32}", state.get("InvocationID", "")) is not None, "autoresume invocation is invalid")
    return state


@contextlib.contextmanager
def hold_reconciliation_lock(run_dir: Path = RUN_DIR) -> Iterator[bool]:
    """Take the trainer's durable run lock, or report an active/transitioning owner."""

    if not run_dir.exists():
        yield True
        return
    require(run_dir.is_dir() and not run_dir.is_symlink(), f"run directory is invalid: {run_dir}")
    lock_path = run_dir / ".run.lock"
    require(lock_path.is_file() and not lock_path.is_symlink(), f"existing run has no valid run lock: {lock_path}")
    with lock_path.open("rb") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def hold_watchdog_lock(path: Path = WATCHDOG_LOCK) -> Iterator[None]:
    require(path.is_file() and not path.is_symlink(), f"watchdog lock is missing/symlink: {path}")
    with path.open("rb") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PublicationContradiction("another completion watchdog owns the reconciliation lock") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def transitioning_services() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for unit in SERVICES:
        state = service_properties(unit)
        if state.get("ActiveState") in ACTIVE_STATES:
            result[unit] = state
    return result


def main() -> int:
    with hold_watchdog_lock():
        active = transitioning_services()
        if active:
            print(json.dumps({"event": "completion_watchdog_trainer_active", "services": active}, sort_keys=True), flush=True)
            return 0

        with hold_reconciliation_lock() as acquired:
            require(acquired, "run lock is busy while both systemd trainer units are inactive")
            # Eliminate the gap between the first service observation and the run lock.
            active = transitioning_services()
            if active:
                print(json.dumps({"event": "completion_watchdog_trainer_became_active", "services": active}, sort_keys=True), flush=True)
                return 0
            reconciliation = reconcile_crash_leftovers(RUN_DIR)
            state = publication_state()

        if (
            reconciliation["removed_temporary_files"]
            or reconciliation["event_tail"] is not None
            or reconciliation["step250_emergency"] is not None
            or reconciliation["checkpoint_generations"] is not None
        ):
            print(json.dumps({"event": "completion_watchdog_reconciled_crash_leftovers", **reconciliation}, sort_keys=True), flush=True)

        if state["status"] in {"publication_complete", "audited_complete"}:
            print(json.dumps({"event": "completion_watchdog_terminal_complete", **state}, sort_keys=True), flush=True)
            return 0

        # Never launch while owning .run.lock: the sealed trainer takes it nonblocking.
        active = transitioning_services()
        if active:
            print(json.dumps({"event": "completion_watchdog_trainer_started_before_replay", "services": active}, sort_keys=True), flush=True)
            return 0
        state = publication_state()
        if state["status"] in {"publication_complete", "audited_complete"}:
            print(json.dumps({"event": "completion_watchdog_terminal_completed_before_replay", **state}, sort_keys=True), flush=True)
            return 0
        service = start_autoresume()
        print(json.dumps({"event": "completion_watchdog_restarted_partial_run", "publication": state, "service": service}, sort_keys=True), flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
