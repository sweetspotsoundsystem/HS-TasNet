#!/usr/bin/env python3
"""CPU-only, fail-closed audit of the completed HS-TasNet c91 production run.

The auditor never mutates trainer outputs.  It writes one new, immutable audit
receipt only after the service, event log, reports, checkpoints, artifacts, and
functional checks all conclusively pass.
"""

from __future__ import annotations

import os

# This must happen before importing Torch, directly or through the model source.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hs-tasnet-matplotlib")

import argparse
import ast
import contextlib
import datetime as dt
import fcntl
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import pickle
import platform
import re
import resource
import shlex
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


AUDITOR_PATH = Path(__file__).resolve()
BASE_DIR = AUDITOR_PATH.parent
DEFAULT_RUN_DIR = BASE_DIR / "runs/c91-full-best200-v1-seed60"
STEP250_DIAGNOSTIC_DIR = BASE_DIR / "diagnostics/native-gate-step250000"
STEP250_ARMING_TERMINAL_NAME = "arming-terminal.json"
STEP250_CANONICAL_OUTCOME_NAMES = {
    "native_frozen_batched_comparison": "comparison.json",
    "native_fast_failure": "frozen-fast-gate-failure.json",
    "missing_native_snapshot_recovery": "recovery-terminal.json",
}
PRIMARY_SERVICE = "hs-tasnet-c91-full-v1.service"
AUTORESTART_SERVICE = "hs-tasnet-c91-full-v1-autoresume.service"
TRAINER_SERVICES = (PRIMARY_SERVICE, AUTORESTART_SERVICE)
DEFAULT_SERVICE = AUTORESTART_SERVICE
RECEIPT_NAME = "final_audit_receipt.json"
EXPECTED_TOTAL_STEPS = 300_000
EXPECTED_RUN_UUID = "cdd5e846-9ea5-4b2d-b71a-7947268f54cf"
EXPECTED_CONTRACT_IDENTITY = "787724005bef19d0d0cc00c58b37be1138967390581a5902cf9ed46134330514"
EXPECTED_TRAINER_SHA256 = "1de9ff67364d5727ff494f751af095ab58353e95aa1a84fc34083574b560744d"
EXPECTED_CONFIG_SHA256 = "987992e70d0e40e812e81d6bd18678d90897bb5df0fbcec47fecf7bbcc13a47a"
EXPECTED_MANIFEST_FILE_SHA256 = "300b0bfbd835e2ce40c8832d10219a35941d6453392fa50688b236499375061a"
EXPECTED_MANIFEST_CONTENT_SHA256 = "a2414fb30ca5fc68fd37517da0f934bcbb0383e4c476ea9a667b2a83fcb703f9"
EXPECTED_EXPERIMENT_SHA256 = "095a909f1ea896f4d36876c886d8a11e845f4067a429e4ed05422341f7d2840d"
EXPECTED_FINAL_APPLIED_LR = 2.960881373414992e-13
EXPECTED_GROUPED_PARAMETER_COUNT = 22
EXPECTED_ADAM_STATE_COUNT = 21
EXPECTED_STATELESS_PARAMETER_NAME = "conv_decode.bias"
EXPECTED_ADAM_STATE_KEYS = {"step", "exp_avg", "exp_avg_sq"}
EXPECTED_SOURCE_COMMIT = "e5b62db805dcd5cd91225843c1cb07c12a453a86"
EXPECTED_SOURCE_TREE = "79dab041ad89d852aa5b8d2233195d42e1ec6441"
EXPECTED_SOURCE_NAMES = ("drums", "bass", "vocals", "other")
EXPECTED_SOURCE_GAINS = (1.0, 1.0, 0.8, 1.12)
EXPECTED_STREAM_SHAPE = (4, 2, 512)
LONG_STREAM_CALLBACKS = 512
STREAM_WARMUP_CALLBACKS = 16
RESET_REPLAY_CALLBACKS = 8
MAX_STREAM_RSS_GROWTH_BYTES = 64 << 20
MAX_STREAM_RSS_BYTES = int(1.5 * (1 << 30))
CHECKPOINT_PATTERN = re.compile(r"^step-(\d{12})-([0-9a-f]{32})\.pt$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
INVOCATION_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class AuditFailure(RuntimeError):
    """A completed-looking run failed an integrity or functional check."""


class IncompleteRun(AuditFailure):
    """Completion cannot yet be proved; no receipt may be written."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def require_complete(condition: bool, message: str) -> None:
    if not condition:
        raise IncompleteRun(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _finite_float(value: Any, context: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{context} is not numeric")
    result = float(value)
    require(math.isfinite(result), f"{context} is not finite")
    return result


def derive_final_learning_rate_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Mirror the hash-bound c91 pre-update LR schedule at its exclusive endpoint."""

    model = config.get("model")
    schedule = config.get("schedule")
    require(isinstance(model, dict) and isinstance(schedule, dict), "bound LR config is malformed")
    require(
        schedule.get("endpoint_semantics")
        == "exclusive_optimizer_steps_0_through_total_steps_minus_1",
        "bound LR schedule has unexpected endpoint semantics",
    )
    base_lr = _finite_float(model.get("learning_rate"), "bound base learning rate")
    require(base_lr > 0.0, "bound base learning rate is not positive")
    total_steps = schedule.get("total_steps")
    decay_start = schedule.get("lr_decay_start")
    decay_end = schedule.get("lr_decay_end")
    require(
        all(isinstance(value, int) and not isinstance(value, bool) for value in (total_steps, decay_start, decay_end)),
        "bound LR schedule counters are malformed",
    )
    require(total_steps == EXPECTED_TOTAL_STEPS, "bound LR schedule total is not 300000")
    require(0 <= decay_start < decay_end == total_steps, "bound LR decay interval is invalid")

    def schedule_scale(preupdate_counter: int) -> float:
        if preupdate_counter <= decay_start:
            return 1.0
        if preupdate_counter >= decay_end:
            return 0.0
        progress = (preupdate_counter - decay_start) / (decay_end - decay_start)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    last_counter = total_steps - 1
    last_applied_scale = schedule_scale(last_counter)
    theoretical_endpoint_scale = schedule_scale(total_steps)
    last_applied_lr = base_lr * last_applied_scale
    theoretical_endpoint_lr = base_lr * theoretical_endpoint_scale
    require(
        last_applied_lr == EXPECTED_FINAL_APPLIED_LR,
        "hash-bound c91 config/code no longer derives the sealed last-applied learning rate",
    )
    require(theoretical_endpoint_lr == 0.0, "the theoretical schedule endpoint is not exactly zero")
    require(last_applied_lr != theoretical_endpoint_lr, "applied and unapplied endpoint learning rates collapsed")
    return {
        "endpoint_semantics": schedule["endpoint_semantics"],
        "base_learning_rate": base_lr,
        "decay_start_preupdate_counter": decay_start,
        "decay_end_preupdate_counter": decay_end,
        "optimizer_update_count": total_steps,
        "last_applied_preupdate_counter": last_counter,
        "last_applied_scale": last_applied_scale,
        "last_applied_learning_rate": last_applied_lr,
        "theoretical_unapplied_preupdate_counter": total_steps,
        "theoretical_unapplied_scale": theoretical_endpoint_scale,
        "theoretical_unapplied_learning_rate": theoretical_endpoint_lr,
    }


def verify_checkpoint_optimizer_learning_rate(
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    schedule = derive_final_learning_rate_contract(config)
    optimizer_meta = payload.get("optimizer_meta")
    optimizer_state = payload.get("optimizer")
    require(isinstance(optimizer_meta, dict), "final checkpoint optimizer metadata is missing")
    require(isinstance(optimizer_state, dict), "final checkpoint optimizer state is missing")
    base_lrs = optimizer_meta.get("base_lrs")
    groups = optimizer_state.get("param_groups")
    require(
        optimizer_meta.get("autoresearch_step") == EXPECTED_TOTAL_STEPS,
        "final checkpoint optimizer metadata counter is not 300000",
    )
    require(isinstance(base_lrs, (list, tuple)) and len(base_lrs) == 1, "final checkpoint base LR groups are invalid")
    require(isinstance(groups, list) and len(groups) == len(base_lrs), "final checkpoint optimizer param groups are invalid")
    normalized_base_lrs = tuple(
        _finite_float(value, f"final checkpoint base LR group {index}")
        for index, value in enumerate(base_lrs)
    )
    require(
        normalized_base_lrs == (schedule["base_learning_rate"],),
        "final checkpoint base LR groups differ from the bound config",
    )
    expected_group_lrs = tuple(
        base_lr * schedule["last_applied_scale"]
        for base_lr in normalized_base_lrs
    )
    actual_group_lrs: list[float] = []
    for index, group in enumerate(groups):
        require(isinstance(group, dict), f"final checkpoint optimizer param group {index} is malformed")
        actual = _finite_float(group.get("lr"), f"final checkpoint optimizer param-group LR {index}")
        require(
            actual == expected_group_lrs[index],
            f"final checkpoint optimizer param-group LR {index} is not the exact last-applied schedule value",
        )
        actual_group_lrs.append(actual)
    return {
        **schedule,
        "optimizer_param_group_count": len(actual_group_lrs),
        "optimizer_param_group_learning_rates": actual_group_lrs,
        "optimizer_param_groups_equal_last_applied_exact": True,
        "theoretical_zero_endpoint_remained_unapplied": True,
    }


def verify_serialized_adam_topology(optimizer_state: Any) -> dict[str, Any]:
    require(isinstance(optimizer_state, dict), "serialized Adam state is not a mapping")
    groups = optimizer_state.get("param_groups")
    states = optimizer_state.get("state")
    require(isinstance(groups, list) and len(groups) == 1, "serialized Adam must have exactly one param group")
    require(isinstance(states, dict), "serialized Adam parameter states are missing")
    group = groups[0]
    require(isinstance(group, dict), "serialized Adam param group is malformed")
    parameter_ids = group.get("params")
    require(isinstance(parameter_ids, list), "serialized Adam grouped parameter IDs are malformed")
    require(
        all(isinstance(identifier, int) and not isinstance(identifier, bool) for identifier in parameter_ids),
        "serialized Adam grouped parameter IDs are not integers",
    )
    require(
        len(parameter_ids) == EXPECTED_GROUPED_PARAMETER_COUNT,
        f"serialized Adam does not group exactly {EXPECTED_GROUPED_PARAMETER_COUNT} parameters",
    )
    require(len(set(parameter_ids)) == len(parameter_ids), "serialized Adam groups a parameter more than once")
    state_ids = set(states)
    require(
        all(isinstance(identifier, int) and not isinstance(identifier, bool) for identifier in state_ids),
        "serialized Adam state IDs are not integers",
    )
    require(len(state_ids) == EXPECTED_ADAM_STATE_COUNT, "serialized Adam state count is not 21")
    require(state_ids < set(parameter_ids), "serialized Adam states are not an exact proper subset of grouped parameters")
    missing_ids = set(parameter_ids) - state_ids
    require(len(missing_ids) == 1, "serialized Adam does not have exactly one stateless grouped parameter")
    missing_id = next(iter(missing_ids))
    return {
        "grouped_parameter_count": len(parameter_ids),
        "grouped_parameter_ids_unique": True,
        "adam_parameter_state_count": len(state_ids),
        "state_ids_owned_by_group_exact": True,
        "stateless_grouped_parameter_count": 1,
        "stateless_grouped_parameter_index": parameter_ids.index(missing_id),
    }


def verify_loaded_adam_coverage(model: Any, optimizer: Any, torch: Any) -> dict[str, Any]:
    named_parameters = list(model.named_parameters())
    model_parameters = list(model.parameters())
    require(len(named_parameters) == EXPECTED_GROUPED_PARAMETER_COUNT, "production model named-parameter count is not 22")
    require(len(model_parameters) == EXPECTED_GROUPED_PARAMETER_COUNT, "production model parameter count is not 22")
    names = [name for name, _ in named_parameters]
    parameter_by_id = {id(parameter): (name, parameter) for name, parameter in named_parameters}
    require(len(set(names)) == len(names), "production model has duplicate parameter names")
    require(len(parameter_by_id) == len(named_parameters), "production model has aliased named parameters")
    require(
        {id(parameter) for parameter in model_parameters} == set(parameter_by_id),
        "production model named and unnamed parameter inventories disagree",
    )

    require(isinstance(optimizer.param_groups, list) and optimizer.param_groups, "loaded Adam has no param groups")
    grouped_parameters: list[Any] = []
    for group_index, group in enumerate(optimizer.param_groups):
        require(isinstance(group, dict), f"loaded Adam param group {group_index} is malformed")
        parameters = group.get("params")
        require(isinstance(parameters, list), f"loaded Adam param group {group_index} parameters are malformed")
        for parameter in parameters:
            require(isinstance(parameter, torch.nn.Parameter), "loaded Adam groups a non-Parameter value")
            grouped_parameters.append(parameter)
    grouped_ids = [id(parameter) for parameter in grouped_parameters]
    require(
        len(grouped_ids) == EXPECTED_GROUPED_PARAMETER_COUNT,
        f"loaded Adam does not group exactly {EXPECTED_GROUPED_PARAMETER_COUNT} parameters",
    )
    require(len(set(grouped_ids)) == len(grouped_ids), "loaded Adam groups a parameter more than once")
    require(set(grouped_ids) == set(parameter_by_id), "loaded Adam has unknown or unowned model parameters")

    states = optimizer.state
    require(isinstance(states, Mapping), "loaded Adam state is not a mapping")
    state_ids = {id(parameter) for parameter in states}
    require(len(states) == EXPECTED_ADAM_STATE_COUNT, "loaded Adam state count is not 21")
    require(len(state_ids) == len(states), "loaded Adam state has duplicate parameter identities")
    require(state_ids < set(grouped_ids), "loaded Adam states are not an exact proper subset of grouped parameters")
    stateless_ids = set(grouped_ids) - state_ids
    require(len(stateless_ids) == 1, "loaded Adam does not have exactly one stateless grouped parameter")
    stateless_id = next(iter(stateless_ids))
    stateless_name, stateless_parameter = parameter_by_id[stateless_id]
    require(
        stateless_name == EXPECTED_STATELESS_PARAMETER_NAME,
        f"unexpected stateless grouped parameter: {stateless_name}",
    )
    require(
        bool(torch.isfinite(stateless_parameter.detach()).all()),
        f"stateless grouped parameter is non-finite: {stateless_name}",
    )

    adam_steps: list[int] = []
    moment_tensor_count = 0
    moment_element_count = 0
    for name, parameter in named_parameters:
        parameter_id = id(parameter)
        require(bool(torch.isfinite(parameter.detach()).all()), f"model parameter is non-finite: {name}")
        if name == EXPECTED_STATELESS_PARAMETER_NAME:
            require(parameter_id == stateless_id, "the bound stateless decoder bias unexpectedly has Adam state")
            continue
        require(parameter_id in state_ids, f"grouped model parameter has no Adam state: {name}")
        state = states[parameter]
        require(isinstance(state, dict), f"Adam state is malformed for {name}")
        require(set(state) == EXPECTED_ADAM_STATE_KEYS, f"Adam state keys are invalid for {name}")
        step_tensor = state["step"]
        require(isinstance(step_tensor, torch.Tensor), f"Adam step is not a tensor for {name}")
        require(step_tensor.ndim == 0 and step_tensor.numel() == 1, f"Adam step is not scalar for {name}")
        require(step_tensor.device == parameter.device, f"Adam step device differs from its parameter for {name}")
        require(bool(torch.isfinite(step_tensor).all()), f"Adam step is non-finite for {name}")
        scalar = float(step_tensor.item())
        require(scalar.is_integer() and int(scalar) == EXPECTED_TOTAL_STEPS, f"Adam step is not 300000 for {name}")
        adam_steps.append(int(scalar))
        for key in ("exp_avg", "exp_avg_sq"):
            moment = state[key]
            require(isinstance(moment, torch.Tensor), f"Adam {key} is not a tensor for {name}")
            require(moment.layout == torch.strided, f"Adam {key} is not strided for {name}")
            require(moment.shape == parameter.shape, f"Adam {key} shape differs from its parameter for {name}")
            require(moment.device == parameter.device, f"Adam {key} device differs from its parameter for {name}")
            require(moment.dtype == parameter.dtype, f"Adam {key} dtype differs from its parameter for {name}")
            require(moment.is_floating_point(), f"Adam {key} is not floating point for {name}")
            require(bool(torch.isfinite(moment).all()), f"Adam {key} is non-finite for {name}")
            if key == "exp_avg_sq":
                require(bool((moment >= 0).all()), f"Adam exp_avg_sq is negative for {name}")
            moment_tensor_count += 1
            moment_element_count += moment.numel()

    require(len(adam_steps) == EXPECTED_ADAM_STATE_COUNT, "validated Adam state count is not 21")
    return {
        "grouped_model_parameter_count": len(grouped_parameters),
        "grouped_model_parameters_unique_and_owned_exact": True,
        "adam_parameter_state_count": len(adam_steps),
        "adam_state_keys": sorted(EXPECTED_ADAM_STATE_KEYS),
        "adam_parameter_step_min": min(adam_steps),
        "adam_parameter_step_max": max(adam_steps),
        "adam_moment_tensor_count": moment_tensor_count,
        "adam_moment_element_count": moment_element_count,
        "adam_moments_shape_device_dtype_match_parameters": True,
        "adam_moments_all_finite": True,
        "adam_second_moments_nonnegative": True,
        "stateless_grouped_parameter_count": 1,
        "stateless_grouped_parameter_name": stateless_name,
        "stateless_grouped_parameter_index": grouped_ids.index(stateless_id),
        "stateless_grouped_parameter_finite": True,
    }


def sha256_file(path: Path, block_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_json_constant(token: str) -> None:
    raise AuditFailure(f"non-finite JSON number is forbidden: {token}")


def _json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def parse_strict_json(blob: str | bytes, context: str) -> Any:
    try:
        return json.loads(
            blob,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_json_object_without_duplicates,
        )
    except AuditFailure:
        raise
    except Exception as error:
        raise AuditFailure(f"invalid JSON in {context}: {error}") from error


def load_json_object(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON file: {path}")
    value = parse_strict_json(path.read_bytes(), str(path))
    require(isinstance(value, dict), f"expected a JSON object: {path}")
    return value


def strict_artifact_pair_state(path: Path, context: str) -> str:
    """Inventory a sealed-artifact pair without following or accepting symlinks."""

    sidecar = path.with_suffix(path.suffix + ".sha256")

    def entry_present(candidate: Path) -> bool:
        try:
            candidate.lstat()
        except FileNotFoundError:
            return False
        except OSError as error:
            raise AuditFailure(f"cannot inventory {context} path {candidate}: {error}") from error
        return True

    artifact_present = entry_present(path)
    sidecar_present = entry_present(sidecar)
    if artifact_present:
        require(path.is_file() and not path.is_symlink(), f"{context} artifact is not a regular file: {path}")
    if sidecar_present:
        require(
            sidecar.is_file() and not sidecar.is_symlink(),
            f"{context} sidecar is not a regular file: {sidecar}",
        )
    if not artifact_present and not sidecar_present:
        return "absent"
    if artifact_present and not sidecar_present:
        return "unsealed"
    if not artifact_present and sidecar_present:
        return "orphan_sidecar"
    return "sealed"


def load_strict_sealed_json(path: Path, context: str) -> tuple[dict[str, Any], str]:
    state = strict_artifact_pair_state(path, context)
    require(state == "sealed", f"{context} pair is not sealed ({state}): {path}")
    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    try:
        fields = sidecar.read_text(encoding="ascii").split()
    except (OSError, UnicodeError) as error:
        raise AuditFailure(f"cannot read {context} sidecar: {sidecar}: {error}") from error
    require(fields == [digest, path.name], f"{context} sidecar does not bind the exact artifact: {path}")
    return load_json_object(path), digest


def verify_strict_sha256_manifest(
    manifest: Path,
    root: Path,
    context: str,
) -> dict[str, Any]:
    """Verify a pinned checksum manifest before importing any code it governs."""

    require(manifest.is_file() and not manifest.is_symlink(), f"missing/symlink {context}: {manifest}")
    root = root.resolve()
    try:
        content = manifest.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise AuditFailure(f"cannot read {context}: {manifest}: {error}") from error
    require(content.endswith("\n"), f"{context} lacks a final newline: {manifest}")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(content.splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\n]+)", line)
        require(match is not None, f"malformed {context} line {line_number}: {line!r}")
        expected, relative = match.groups()
        relative_path = Path(relative)
        require(not relative_path.is_absolute() and relative not in entries, f"invalid/duplicate {context} path: {relative}")
        path = root / relative_path
        resolved = path.resolve()
        require(resolved.is_relative_to(root), f"{context} path escapes its root: {relative}")
        require(path.is_file() and not path.is_symlink(), f"{context} input missing/symlink: {path}")
        require(sha256_file(path) == expected, f"{context} checksum mismatch: {path}")
        entries[relative] = expected
    require(entries, f"{context} is empty: {manifest}")
    return {
        "path": str(manifest),
        "sha256": sha256_file(manifest),
        "entry_count": len(entries),
    }


def load_step250_validator_module(path: Path, label: str) -> tuple[Any, str]:
    require(path.is_file() and not path.is_symlink(), f"missing/symlink step250 validator: {path}")
    digest = sha256_file(path)
    module_name = f"_hs_tasnet_final_audit_step250_{label}_{digest[:16]}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    require(spec is not None and spec.loader is not None, f"cannot load step250 validator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        sys.modules.pop(module_name, None)
        raise AuditFailure(f"cannot import step250 validator {path}: {error}") from error
    return module, digest


def invoke_step250_validator(label: str, callback: Any, *arguments: Any) -> Any:
    require(callable(callback), f"step250 validator callback is missing: {label}")
    try:
        return callback(*arguments)
    except AuditFailure:
        raise
    except Exception as error:
        raise AuditFailure(f"step250 {label} validation failed: {error}") from error


@contextlib.contextmanager
def owned_step250_validator_import_path(diagnostic_dir: Path) -> Iterator[None]:
    """Give sealed validators import precedence without leaking their path edits.

    The hash-bound validators legitimately import the production trainer, whose
    source resolver moves the bound checkout to the front of ``sys.path``.
    Third-party imports may add helper paths as well.  The final auditor owns
    the surrounding interpreter state: it prepends the diagnostic directory,
    permits the sealed validators to manage their own imports, and restores the
    exact original list contents on both success and exception paths.

    Replacing the ``sys.path`` list object is not part of that contract.  Treat
    it as a fail-closed integrity error, after first restoring the caller's
    original object and entries.
    """

    original_path_object = sys.path
    require(isinstance(original_path_object, list), "sys.path is not a list")
    original_entries = tuple(original_path_object)
    import_path = str(diagnostic_dir)
    original_path_object.insert(0, import_path)
    try:
        require(sys.path is original_path_object, "step250 validator import path object changed before validation")
        require(sys.path and sys.path[0] == import_path, "step250 validator import path was not installed")
        yield
    finally:
        path_object_replaced = sys.path is not original_path_object
        sys.path = original_path_object
        original_path_object[:] = original_entries
        require(not path_object_replaced, "step250 validator replaced the sys.path object")
        require(tuple(sys.path) == original_entries, "step250 validator import path restoration failed")


def verify_step250_diagnostic_terminal(
    diagnostic_dir: Path = STEP250_DIAGNOSTIC_DIR,
    manifest_root: Path = BASE_DIR,
) -> dict[str, Any]:
    """Require one immutable, strongly validated step-250000 diagnostic outcome."""

    require(
        diagnostic_dir.is_dir() and not diagnostic_dir.is_symlink(),
        f"step250 diagnostic directory is missing or a symlink: {diagnostic_dir}",
    )
    diagnostic_dir = diagnostic_dir.resolve()
    arming_path = diagnostic_dir / STEP250_ARMING_TERMINAL_NAME
    outcome_paths = {
        name: diagnostic_dir / filename
        for name, filename in STEP250_CANONICAL_OUTCOME_NAMES.items()
    }
    arming_state = strict_artifact_pair_state(arming_path, "step250 arming terminal")
    outcome_states = {
        name: strict_artifact_pair_state(path, f"step250 canonical outcome {name}")
        for name, path in outcome_paths.items()
    }
    invalid_states = {
        name: state for name, state in outcome_states.items() if state not in {"absent", "sealed"}
    }
    require(not invalid_states, f"incomplete canonical step250 outcome pair(s): {invalid_states}")
    require(
        arming_state in {"absent", "sealed"},
        f"step250 arming terminal pair is incomplete ({arming_state}): {arming_path}",
    )
    sealed_outcomes = [name for name, state in outcome_states.items() if state == "sealed"]
    require(len(sealed_outcomes) <= 1, f"contradictory canonical step250 outcomes: {sealed_outcomes}")

    if arming_state == "absent" and not sealed_outcomes:
        raise IncompleteRun("step250 diagnostic has not sealed its arming or canonical terminal yet")
    require(arming_state == "sealed", "canonical step250 outcome exists without a sealed arming terminal")

    arming, arming_sha256 = load_strict_sealed_json(arming_path, "step250 arming terminal")
    automation_path = diagnostic_dir / "automation.sha256"
    preflight_path = diagnostic_dir / "armer-preflight.sha256"
    automation_manifest = verify_strict_sha256_manifest(
        automation_path,
        manifest_root,
        "step250 automation manifest",
    )
    armer_preflight_manifest = verify_strict_sha256_manifest(
        preflight_path,
        manifest_root,
        "step250 armer preflight manifest",
    )
    for name, path, verified in (
        ("automation", automation_path, automation_manifest),
        ("preflight", preflight_path, armer_preflight_manifest),
    ):
        recorded = arming.get(name)
        require(isinstance(recorded, dict), f"step250 arming {name} record is invalid")
        require(Path(str(recorded.get("path", ""))).resolve() == path.resolve(), f"step250 arming {name} path changed")
        require(recorded.get("sha256") == verified["sha256"], f"step250 arming {name} hash changed")
    validator_files: dict[str, dict[str, str]] = {}
    with owned_step250_validator_import_path(diagnostic_dir):
        armer_path = diagnostic_dir / "arm_step250_after_step225.py"
        armer, armer_sha256 = load_step250_validator_module(armer_path, "armer")
        invoke_step250_validator("arming terminal", getattr(armer, "validate_terminal", None), arming)
        validator_files["arming_terminal"] = {
            "path": str(armer_path),
            "sha256": armer_sha256,
        }

        if not sealed_outcomes:
            raise IncompleteRun("step250 diagnostic has no canonical terminal outcome yet")

        outcome_name = sealed_outcomes[0]
        outcome_path = outcome_paths[outcome_name]
        outcome, outcome_sha256 = load_strict_sealed_json(
            outcome_path,
            f"step250 canonical outcome {outcome_name}",
        )
        if outcome_name in {"native_frozen_batched_comparison", "native_fast_failure"}:
            gate_path = diagnostic_dir / "run_step250000_gate.py"
            gate, gate_sha256 = load_step250_validator_module(gate_path, "gate")
            validator_files["normal_gate"] = {
                "path": str(gate_path),
                "sha256": gate_sha256,
            }
            if outcome_name == "native_frozen_batched_comparison":
                # The strong comparison validator uses crash-reconciliation helpers.
                # Requiring every reconciled input to be sealed first makes this audit read-only.
                for filename, label in (
                    ("step250000-full14.json", "step250 comparison result"),
                    ("audit-receipt.json", "step250 artifact audit"),
                    ("lr-boundary-audit-receipt.json", "step250 LR boundary audit"),
                    ("step225-reference-binding.json", "step250 step225 reference binding"),
                ):
                    load_strict_sealed_json(diagnostic_dir / filename, label)
                comparison_path = diagnostic_dir / "compare_step250000.py"
                require(
                    comparison_path.is_file() and not comparison_path.is_symlink(),
                    f"missing/symlink step250 comparison validator: {comparison_path}",
                )
                validator_files["comparison"] = {
                    "path": str(comparison_path),
                    "sha256": sha256_file(comparison_path),
                }
                invoke_step250_validator(
                    "native comparison",
                    getattr(gate, "validate_comparison_marker", None),
                    outcome,
                )
            else:
                invoke_step250_validator(
                    "frozen-fast failure",
                    getattr(gate, "validate_failure_receipt", None),
                    outcome,
                )
        else:
            recovery_path = diagnostic_dir / "recover_missing_snapshot_step250000.py"
            recovery, recovery_sha256 = load_step250_validator_module(recovery_path, "recovery")
            validator_files["missing_snapshot_recovery"] = {
                "path": str(recovery_path),
                "sha256": recovery_sha256,
            }
            validated_recovery = invoke_step250_validator(
                "missing-snapshot recovery",
                getattr(recovery, "verify_committed_recovery", None),
            )
            require(validated_recovery == outcome, "strong recovery validator returned a different terminal")

        crashsafe_path = diagnostic_dir / "crashsafe_step250000.py"
        require(
            crashsafe_path.is_file() and not crashsafe_path.is_symlink(),
            f"missing/symlink step250 crash-safe helper: {crashsafe_path}",
        )
        validator_files["crashsafe_helper"] = {
            "path": str(crashsafe_path),
            "sha256": sha256_file(crashsafe_path),
        }

    return {
        "status": "validated",
        "step": 250_000,
        "diagnostic_directory": str(diagnostic_dir),
        "arming_terminal": {
            "path": str(arming_path),
            "sha256": arming_sha256,
            "step225_trigger_outcome_name": arming["outcome"]["name"],
            "step225_trigger_outcome_sha256": arming["outcome"]["sha256"],
            "step225_reference_binding_sha256": arming["binding"]["sha256"],
            "automation_manifest_sha256": arming["automation"]["sha256"],
            "armer_preflight_manifest_sha256": arming["preflight"]["sha256"],
        },
        "canonical_outcome": {
            "name": outcome_name,
            "path": str(outcome_path),
            "sha256": outcome_sha256,
            "pair_states": outcome_states,
            "exactly_one_sealed": True,
        },
        "validator_files": dict(sorted(validator_files.items())),
        "validator_manifests": {
            "armer_preflight": armer_preflight_manifest,
            "automation": automation_manifest,
        },
    }


def revalidate_step250_diagnostic_terminal(initial: Mapping[str, Any]) -> None:
    """Turn disappearance after an initial validation into a hard integrity failure."""

    try:
        current = verify_step250_diagnostic_terminal()
    except IncompleteRun as error:
        raise AuditFailure("validated step250 diagnostic evidence disappeared during the final CPU audit") from error
    require(current == initial, "step250 diagnostic evidence changed during the final CPU audit")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL file: {path}")
    content = path.read_bytes()
    require(content.endswith(b"\n"), f"JSONL file has an incomplete final line: {path}")
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(content.splitlines(), 1):
        try:
            value = parse_strict_json(raw, f"{path}:{line_number}")
        except Exception as error:
            raise AuditFailure(f"invalid JSONL record {path}:{line_number}: {error}") from error
        require(isinstance(value, dict), f"JSONL record is not an object: {path}:{line_number}")
        records.append(value)
    require(records, f"JSONL file is empty: {path}")
    return records


def _stat_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def observe_file(path: Path, observations: dict[str, dict[str, Any]]) -> str:
    resolved = path.resolve()
    require(path.is_file() and not path.is_symlink(), f"audited file is missing or a symlink: {path}")
    digest = sha256_file(path)
    observations[str(resolved)] = {
        "sha256": digest,
        "stat": _stat_identity(path),
    }
    return digest


def verify_observations_stable(observations: Mapping[str, Mapping[str, Any]]) -> None:
    for raw_path, record in observations.items():
        path = Path(raw_path)
        require(path.is_file() and not path.is_symlink(), f"audited input disappeared or became a symlink: {path}")
        require(_stat_identity(path) == record["stat"], f"audited input changed during the audit: {path}")
        require(sha256_file(path) == record["sha256"], f"audited input bytes changed during the audit: {path}")


SERVICE_PROPERTIES = (
    "LoadState",
    "ActiveState",
    "SubState",
    "Result",
    "ExecMainCode",
    "ExecMainStatus",
    "MainPID",
    "RemainAfterExit",
    "InvocationID",
    "ExecMainStartTimestamp",
    "ExecMainExitTimestamp",
    "CollectMode",
)


# systemctl(1) may render the ExecMainCode= CLD_* enum either by its symbolic
# name or by its numeric value.  CLD_EXITED is exactly 1; no other spelling or
# numeric value proves a normal process exit.
NORMAL_EXIT_EXEC_MAIN_CODES = frozenset({"exited", "1"})


def is_normal_exit_exec_main_code(value: object) -> bool:
    return type(value) is str and value in NORMAL_EXIT_EXEC_MAIN_CODES


def read_service_state(service: str) -> dict[str, str]:
    command = ["systemctl", "--user", "show", service]
    for name in SERVICE_PROPERTIES:
        command.extend(("--property", name))
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    require_complete(result.returncode == 0, f"cannot prove service completion for {service}: {result.stderr.strip()}")
    state: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            state[key] = value
    require_complete(all(name in state for name in SERVICE_PROPERTIES), f"service state is incomplete for {service}")
    return state


def validate_service_complete(state: Mapping[str, str], service: str) -> None:
    require_complete(state.get("LoadState") == "loaded", f"service definition is not loaded: {service} ({state.get('LoadState')})")
    require_complete(state.get("ActiveState") == "inactive", f"service is not inactive: {service} ({state.get('ActiveState')})")
    require_complete(state.get("SubState") in {"dead", "exited"}, f"service is not stopped: {service} ({state.get('SubState')})")
    require_complete(state.get("Result") == "success", f"service result is not success: {service} ({state.get('Result')})")
    require_complete(
        is_normal_exit_exec_main_code(state.get("ExecMainCode")),
        f"trainer exit code is not a normal exit: {state.get('ExecMainCode')}",
    )
    require_complete(state.get("ExecMainStatus") == "0", f"trainer exit status is not zero: {state.get('ExecMainStatus')}")
    require_complete(state.get("MainPID") == "0", f"trainer still has a main PID: {state.get('MainPID')}")
    require_complete(state.get("RemainAfterExit", "").lower() in {"no", "false", "0"}, "service completion semantics are unexpected")
    require_complete(
        bool(INVOCATION_ID_PATTERN.fullmatch(state.get("InvocationID", ""))),
        f"service invocation ID is missing or malformed: {service}",
    )
    require_complete(bool(state.get("ExecMainStartTimestamp")), "service start timestamp is missing")
    require_complete(bool(state.get("ExecMainExitTimestamp")), "service exit timestamp is missing")


def validate_service_not_active(state: Mapping[str, str], service: str) -> None:
    require_complete(
        state.get("ActiveState") not in {"active", "activating", "reloading", "deactivating"},
        f"service is active or transitioning: {service} ({state.get('ActiveState')})",
    )


def validate_loaded_post_reboot_state(state: Mapping[str, str], service: str) -> None:
    """Accept only systemd's exact volatile-reset shape; the journal remains authoritative."""

    require_complete(state.get("LoadState") == "loaded", f"service definition is not loaded: {service}")
    require_complete(state.get("ActiveState") == "inactive", f"post-reboot service is not inactive: {service}")
    require_complete(state.get("SubState") == "dead", f"post-reboot service is not dead: {service}")
    require_complete(state.get("Result") == "success", f"post-reboot service result is not success: {service}")
    require_complete(state.get("ExecMainCode") == "0", f"post-reboot service has residual exit-code state: {service}")
    require_complete(state.get("ExecMainStatus") == "0", f"post-reboot service has residual exit status: {service}")
    require_complete(state.get("MainPID") == "0", f"post-reboot service still has a PID: {service}")
    require_complete(state.get("RemainAfterExit", "").lower() in {"no", "false", "0"}, "post-reboot service completion semantics are unexpected")
    require_complete(state.get("InvocationID") == "", f"post-reboot service has an ambiguous invocation: {service}")
    require_complete(state.get("ExecMainStartTimestamp") == "", f"post-reboot service retained only a start timestamp: {service}")
    require_complete(state.get("ExecMainExitTimestamp") == "", f"post-reboot service retained only an exit timestamp: {service}")
    require_complete(state.get("CollectMode") == "inactive", f"post-reboot service collect mode changed: {service}")


def validate_loaded_failed_state(state: Mapping[str, str], service: str) -> None:
    require_complete(state.get("LoadState") == "loaded", f"failed service definition is not loaded: {service}")
    require_complete(state.get("ActiveState") == "failed", f"post-publication service is not failed: {service}")
    require_complete(state.get("SubState") in {"failed", "dead"}, f"post-publication service substate changed: {service}")
    require_complete(
        state.get("Result") in {"exit-code", "signal", "core-dump", "timeout", "watchdog", "resources", "protocol"},
        f"post-publication service result is not a recognized failure: {service}",
    )
    require_complete(state.get("MainPID") == "0", f"failed service still has a PID: {service}")
    require_complete(state.get("RemainAfterExit", "").lower() in {"no", "false", "0"}, "failed service completion semantics are unexpected")
    require_complete(
        bool(INVOCATION_ID_PATTERN.fullmatch(state.get("InvocationID", ""))),
        f"failed service invocation ID is missing or malformed: {service}",
    )
    require_complete(bool(state.get("ExecMainStartTimestamp")), "failed service start timestamp is missing")
    require_complete(bool(state.get("ExecMainExitTimestamp")), "failed service exit timestamp is missing")
    require_complete(state.get("CollectMode") == "inactive", f"failed service collect mode changed: {service}")
    require_complete(
        not is_normal_exit_exec_main_code(state.get("ExecMainCode")) or state.get("ExecMainStatus") != "0",
        "failed service paradoxically records a normal main-process exit",
    )


def companion_trainer_name(service: str) -> str | None:
    if service not in TRAINER_SERVICES:
        return None
    return next(candidate for candidate in TRAINER_SERVICES if candidate != service)


def validate_companion_trainer_inactive(service: str) -> dict[str, str] | None:
    companion = companion_trainer_name(service)
    if companion is None:
        return None
    state = read_service_state(companion)
    validate_service_not_active(state, companion)
    return state


PROCESS_EXIT_MESSAGE_ID = "98e322203f7a4ed290d09fe03c09fe15"
UNIT_FAILED_MESSAGE_ID = "d9b373ed55a64feb8242e02dbe79a49c"
RESOURCE_MESSAGE_ID = "ae8f7b866b0347b9af31fe1c80b127c0"


def read_service_journal(service: str) -> list[dict[str, Any]]:
    result = subprocess.run(
        ["journalctl", "--user", "--unit", service, "--output", "json", "--no-pager"],
        capture_output=True,
        timeout=60,
    )
    require_complete(
        result.returncode == 0,
        f"cannot read durable journal evidence for {service}: {result.stderr.decode('utf-8', 'replace').strip()}",
    )
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(result.stdout.splitlines(), 1):
        value = parse_strict_json(line, f"journal for {service}, line {line_number}")
        require(isinstance(value, dict), f"journal record is not an object for {service}, line {line_number}")
        records.append(value)
    require_complete(bool(records), f"journal has no evidence for collected service {service}")
    return records


def _record_invocation(record: Mapping[str, Any]) -> str:
    return str(record.get("_SYSTEMD_INVOCATION_ID") or record.get("USER_INVOCATION_ID") or "")


def expected_trainer_command(run_dir: Path, preflight: Mapping[str, Any]) -> list[str]:
    static = preflight["contract"].get("static")
    require(isinstance(static, dict), "run contract static payload is missing")
    return [
        "/home/axel/miniforge3/bin/python",
        str(BASE_DIR / "train_production.py"),
        "--config",
        str(BASE_DIR / "full_config.json"),
        "--manifest",
        str(BASE_DIR / "manifests/combined.manifest.json"),
        "--manifest-sha256",
        str(static.get("manifest_file_sha256")),
        "--run-dir",
        str(run_dir),
        "--resume",
        "auto",
        "--device",
        "cuda",
    ]


def prove_collected_service_from_journal(
    service: str,
    run_dir: Path,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    records = read_service_journal(service)
    report = preflight["report"]
    candidates: list[tuple[int, int, str, dict[str, Any]]] = []
    for start, record in enumerate(records):
        if (
            record.get("_SYSTEMD_USER_UNIT") != service
            or record.get("_TRANSPORT") != "stdout"
            or record.get("MESSAGE") != "{"
        ):
            continue
        invocation = _record_invocation(record)
        if not INVOCATION_ID_PATTERN.fullmatch(invocation):
            continue
        messages: list[str] = []
        for end in range(start, len(records)):
            current = records[end]
            if (
                current.get("_SYSTEMD_USER_UNIT") != service
                or current.get("_TRANSPORT") != "stdout"
                or _record_invocation(current) != invocation
            ):
                break
            message = current.get("MESSAGE")
            if not isinstance(message, str):
                break
            messages.append(message)
            if message == "}":
                try:
                    parsed = parse_strict_json("\n".join(messages), f"terminal stdout for {service}")
                except AuditFailure:
                    continue
                if parsed == report:
                    candidates.append((start, end, invocation, current))
                break
    require_complete(len(candidates) == 1, f"journal does not contain exactly one terminal report for {service}")
    start, end, invocation, terminal_record = candidates[0]

    terminal_stdout = [
        record
        for record in records
        if record.get("_SYSTEMD_USER_UNIT") == service
        and record.get("_TRANSPORT") == "stdout"
        and _record_invocation(record) == invocation
    ]
    require(terminal_stdout and records[end] is terminal_stdout[-1], "trainer emitted stdout after its terminal report")

    expected_command = expected_trainer_command(run_dir, preflight)
    command_lines = {
        record.get("_CMDLINE")
        for record in records[start : end + 1]
        if isinstance(record.get("_CMDLINE"), str)
    }
    require(len(command_lines) == 1, "terminal journal records do not have one bound trainer command")
    require(shlex.split(next(iter(command_lines))) == expected_command, "terminal journal command is not the sealed production command")

    started = [
        record
        for record in records[:start]
        if record.get("USER_UNIT") == service
        and record.get("USER_INVOCATION_ID") == invocation
        and str(record.get("MESSAGE", "")).startswith(f"Started {service} ")
    ]
    require_complete(len(started) == 1, f"journal start evidence is missing or ambiguous for {service}")
    event_time = preflight["completion_event"].get("time")
    require(isinstance(event_time, str), "run_complete event timestamp is missing")
    try:
        event_timestamp_us = int(dt.datetime.fromisoformat(event_time).timestamp() * 1_000_000)
        terminal_timestamp_us = int(records[start]["__REALTIME_TIMESTAMP"])
    except (KeyError, TypeError, ValueError) as error:
        raise AuditFailure("cannot compare journal and run_complete timestamps") from error
    require(terminal_timestamp_us >= event_timestamp_us, "terminal stdout predates run_complete publication")

    later_manager = [record for record in records[end + 1 :] if record.get("USER_UNIT") == service]
    failures = [
        record
        for record in later_manager
        if record.get("MESSAGE_ID") in {PROCESS_EXIT_MESSAGE_ID, UNIT_FAILED_MESSAGE_ID}
    ]
    require(not failures, "systemd recorded a process-exit or unit-failure event after terminal stdout")
    later_starts = [
        record
        for record in records[end + 1 :]
        if record.get("USER_UNIT") == service
        and str(record.get("MESSAGE", "")).startswith(f"Started {service} ")
    ]
    require(not later_starts, "a later service invocation followed the terminal report")
    resources = [
        record
        for record in later_manager
        if record.get("MESSAGE_ID") == RESOURCE_MESSAGE_ID
        and record.get("USER_INVOCATION_ID") == invocation
    ]
    require_complete(bool(resources), "terminal systemd resource record is missing")
    resource_record = resources[-1]

    return {
        "proof_source": "systemd_journal_after_transient_collection",
        "load_state": "not-found",
        "active_state": "inactive",
        "invocation_id": invocation,
        "normal_exit_inferred_from_terminal_stdout_and_no_failure_record": True,
        "terminal_report_equal_exact": True,
        "start_cursor": started[0].get("__CURSOR"),
        "terminal_stdout_start_cursor": records[start].get("__CURSOR"),
        "terminal_stdout_end_cursor": terminal_record.get("__CURSOR"),
        "terminal_stdout_realtime_timestamp_us": terminal_timestamp_us,
        "resource_cursor": resource_record.get("__CURSOR"),
        "resource_message": resource_record.get("MESSAGE"),
        "failure_message_ids_absent_after_terminal": [PROCESS_EXIT_MESSAGE_ID, UNIT_FAILED_MESSAGE_ID],
        "bound_command": expected_command,
    }


def prove_postpublication_failure_from_journal(
    service: str,
    run_dir: Path,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove a process failure happened only after the durable terminal publication."""

    records = read_service_journal(service)
    report = preflight["report"]
    event_time = preflight["completion_event"].get("time")
    require(isinstance(event_time, str), "run_complete event timestamp is missing")
    try:
        completion_timestamp_us = int(dt.datetime.fromisoformat(event_time).timestamp() * 1_000_000)
    except ValueError as error:
        raise AuditFailure("run_complete event timestamp is invalid") from error

    postpublication_failures: list[tuple[int, dict[str, Any]]] = []
    for index, record in enumerate(records):
        if record.get("USER_UNIT") != service or record.get("MESSAGE_ID") not in {PROCESS_EXIT_MESSAGE_ID, UNIT_FAILED_MESSAGE_ID}:
            continue
        invocation = _record_invocation(record)
        if not INVOCATION_ID_PATTERN.fullmatch(invocation):
            continue
        try:
            timestamp_us = int(record["__REALTIME_TIMESTAMP"])
        except (KeyError, TypeError, ValueError) as error:
            raise AuditFailure("service failure journal record has no valid timestamp") from error
        if timestamp_us >= completion_timestamp_us:
            postpublication_failures.append((index, record))
    invocations = {_record_invocation(record) for _, record in postpublication_failures}
    require_complete(len(invocations) == 1, f"journal does not identify one post-publication failed invocation for {service}")
    invocation = next(iter(invocations))
    failures = [(index, record) for index, record in postpublication_failures if _record_invocation(record) == invocation]
    first_failure_index = min(index for index, _ in failures)

    starts = [
        (index, record)
        for index, record in enumerate(records[: first_failure_index + 1])
        if record.get("USER_UNIT") == service
        and record.get("USER_INVOCATION_ID") == invocation
        and str(record.get("MESSAGE", "")).startswith(f"Started {service} ")
    ]
    require_complete(len(starts) == 1, f"post-publication failed invocation has no unique start: {service}")
    start_index, start_record = starts[0]
    try:
        start_timestamp_us = int(start_record["__REALTIME_TIMESTAMP"])
    except (KeyError, TypeError, ValueError) as error:
        raise AuditFailure("service start journal record has no valid timestamp") from error
    require(start_timestamp_us <= completion_timestamp_us, "failed service invocation started after run_complete")

    expected_command = expected_trainer_command(run_dir, preflight)
    stdout = [
        (index, record)
        for index, record in enumerate(records[start_index : first_failure_index + 1], start_index)
        if record.get("_SYSTEMD_USER_UNIT") == service
        and record.get("_TRANSPORT") == "stdout"
        and _record_invocation(record) == invocation
    ]
    require_complete(bool(stdout), "failed terminal invocation emitted no command-bound stdout")
    command_lines = {record.get("_CMDLINE") for _, record in stdout if isinstance(record.get("_CMDLINE"), str)}
    require(len(command_lines) == 1, "failed terminal invocation does not have one bound trainer command")
    require(shlex.split(next(iter(command_lines))) == expected_command, "failed terminal invocation command is not sealed")

    event_records = preflight.get("events")
    completion_index = preflight.get("completion_event_index")
    require(isinstance(event_records, list) and isinstance(completion_index, int), "completion event context is missing")
    prior_events = event_records[: completion_index + 1]
    matched_stdout_events: list[dict[str, Any]] = []
    for _, record in stdout:
        message = record.get("MESSAGE")
        if not isinstance(message, str) or not message.startswith("{") or not message.endswith("}"):
            continue
        try:
            value = parse_strict_json(message, f"stdout event for post-publication failure of {service}")
        except AuditFailure:
            continue
        if isinstance(value, dict) and value in prior_events:
            matched_stdout_events.append(value)
    require_complete(bool(matched_stdout_events), "failed terminal invocation has no stdout event bound to events.jsonl")
    require(
        any(record.get("event") == "train_progress" and record.get("step") == EXPECTED_TOTAL_STEPS for record in matched_stdout_events),
        "failed terminal invocation has no exact step300000 progress binding",
    )

    terminal_lines = json.dumps(report, indent=2, sort_keys=True, allow_nan=False).splitlines()
    after_completion_stdout: list[str] = []
    for _, record in stdout:
        try:
            timestamp_us = int(record["__REALTIME_TIMESTAMP"])
        except (KeyError, TypeError, ValueError) as error:
            raise AuditFailure("trainer stdout journal record has no valid timestamp") from error
        if timestamp_us >= completion_timestamp_us:
            message = record.get("MESSAGE")
            require(isinstance(message, str), "post-publication stdout is not text")
            after_completion_stdout.append(message)
    require(
        after_completion_stdout == terminal_lines[: len(after_completion_stdout)],
        "failed trainer emitted unexpected stdout after run_complete",
    )

    later_starts = [
        record
        for record in records[first_failure_index + 1 :]
        if record.get("USER_UNIT") == service and str(record.get("MESSAGE", "")).startswith(f"Started {service} ")
    ]
    require(not later_starts, "a later service invocation followed the post-publication failure")
    resources = [
        record
        for record in records[first_failure_index:]
        if record.get("MESSAGE_ID") == RESOURCE_MESSAGE_ID
        and record.get("USER_UNIT") == service
        and record.get("USER_INVOCATION_ID") == invocation
    ]
    require_complete(bool(resources), "post-publication failed invocation has no terminal resource record")

    return {
        "proof_source": "systemd_journal_postpublication_failure_after_durable_completion",
        "load_state": "journal-bound",
        "active_state": "inactive",
        "invocation_id": invocation,
        "normal_exit": False,
        "postpublication_failure_accepted": True,
        "run_complete_precedes_failure_exact": True,
        "events_stdout_binding_exact": True,
        "bound_command": expected_command,
        "start_cursor": start_record.get("__CURSOR"),
        "terminal_stdout_start_cursor": stdout[0][1].get("__CURSOR"),
        "terminal_stdout_end_cursor": stdout[-1][1].get("__CURSOR"),
        "failure_cursors": [record.get("__CURSOR") for _, record in failures],
        "failure_message_ids": sorted({str(record.get("MESSAGE_ID")) for _, record in failures}),
        "resource_cursor": resources[-1].get("__CURSOR"),
        "resource_message": resources[-1].get("MESSAGE"),
    }


def prove_service_complete(
    state: Mapping[str, str],
    service: str,
    run_dir: Path,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    validate_service_not_active(state, service)
    if state.get("LoadState") == "loaded":
        volatile_state_reset = state.get("InvocationID") == ""
        if volatile_state_reset:
            validate_loaded_post_reboot_state(state, service)
            try:
                journal_evidence = prove_collected_service_from_journal(service, run_dir, preflight)
            except (AuditFailure, IncompleteRun):
                journal_evidence = prove_postpublication_failure_from_journal(service, run_dir, preflight)
        elif state.get("ActiveState") == "failed" or state.get("Result") != "success":
            validate_loaded_failed_state(state, service)
            journal_evidence = prove_postpublication_failure_from_journal(service, run_dir, preflight)
            require_complete(
                journal_evidence.get("invocation_id") == state.get("InvocationID"),
                "failed loaded service state and post-publication journal invocation disagree",
            )
        else:
            validate_service_complete(state, service)
            journal_evidence = prove_collected_service_from_journal(service, run_dir, preflight)
            require_complete(
                journal_evidence.get("invocation_id") == state.get("InvocationID"),
                "loaded service state and terminal journal invocation disagree",
            )
        return {
            **journal_evidence,
            "proof_source": (
                (
                    "systemctl_loaded_post_reboot_and_bound_journal"
                    if not journal_evidence.get("postpublication_failure_accepted")
                    else "systemctl_loaded_post_reboot_and_postpublication_failure_journal"
                )
                if volatile_state_reset
                else (
                    "systemctl_loaded_failed_after_durable_completion"
                    if journal_evidence.get("postpublication_failure_accepted")
                    else "systemctl_loaded_unit_and_bound_journal"
                )
            ),
            "load_state": "loaded",
            "invocation_id": journal_evidence["invocation_id"],
            "systemctl_volatile_runtime_identity_reset": volatile_state_reset,
            "systemctl": dict(state),
        }
    require_complete(
        state.get("LoadState") == "not-found",
        f"unexpected service load state for {service}: {state.get('LoadState')}",
    )
    try:
        return prove_collected_service_from_journal(service, run_dir, preflight)
    except (AuditFailure, IncompleteRun):
        return prove_postpublication_failure_from_journal(service, run_dir, preflight)


@contextlib.contextmanager
def hold_completed_run_lock(run_dir: Path) -> Iterator[None]:
    lock_path = run_dir / ".run.lock"
    require_complete(lock_path.is_file() and not lock_path.is_symlink(), f"run lock is missing or a symlink: {lock_path}")
    with lock_path.open("rb") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise IncompleteRun(f"trainer still owns the run lock: {run_dir}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


VERIFIED_CHECKPOINT_EVENTS = {
    "initial_checkpoint_verified",
    "checkpoint_verified",
    "emergency_checkpoint_verified",
    "pause_checkpoint_verified",
}
RESTART_PREFLIGHT_EVENTS = {
    "inventory_verified",
    "run_opened",
    "resume_candidates_rejected",
}
FINAL_PROGRESS_KEYS = {
    "event",
    "time",
    "step",
    "total_steps",
    "loss",
    "loss_mean_recent",
    "learning_rate",
    "steps_per_second",
    "eta_hours",
    "peak_vram_bytes",
    "free_disk_gb",
}
FINAL_PROGRESS_DETERMINISTIC_KEYS = (
    "step",
    "total_steps",
    "loss",
    "loss_mean_recent",
    "learning_rate",
)


def validate_recovery_lineage(
    events: Sequence[Mapping[str, Any]],
    completion_index: int,
) -> dict[str, Any]:
    """Fail closed unless every resume and prior interruption is hash-bound."""

    require(0 <= completion_index < len(events), "completion event index is invalid")
    checkpoint_indices = [
        index
        for index, record in enumerate(events[: completion_index + 1])
        if record.get("event") in VERIFIED_CHECKPOINT_EVENTS
    ]
    resume_indices = [
        index
        for index, record in enumerate(events[:completion_index])
        if record.get("event") == "resumed"
    ]

    resumes: list[dict[str, Any]] = []
    for resume_index in resume_indices:
        record = events[resume_index]
        step = record.get("step")
        model_hash = record.get("model_state_sha256")
        require(isinstance(step, int) and 0 <= step <= EXPECTED_TOTAL_STEPS, "resumed event has an invalid step")
        require(isinstance(model_hash, str) and SHA256_PATTERN.fullmatch(model_hash), "resumed event has an invalid model hash")
        anchors = [
            index
            for index in checkpoint_indices
            if index < resume_index
            and events[index].get("step") == step
            and events[index].get("model_state_sha256") == model_hash
        ]
        require(anchors, f"resume at event {resume_index} has no prior hash-bound checkpoint")
        anchor_index = anchors[-1]
        continuations = [
            index
            for index in checkpoint_indices
            if index > resume_index
            and isinstance(events[index].get("step"), int)
            and events[index]["step"] > step
        ]
        if continuations:
            continuation_index = continuations[0]
            continuation_step = int(events[continuation_index]["step"])
        else:
            completion_step = events[completion_index].get("step")
            require(isinstance(completion_step, int), "completion event has an invalid step")
            require(
                completion_step > step
                or (step == EXPECTED_TOTAL_STEPS and completion_step == step),
                f"resume at event {resume_index} never reached a valid completion after step {step}",
            )
            continuation_index = completion_index
            continuation_step = completion_step
        resumes.append(
            {
                "event_index": resume_index,
                "step": step,
                "model_state_sha256": model_hash,
                "anchor_event_index": anchor_index,
                "anchor_event": events[anchor_index].get("event"),
                "continuation_event_index": continuation_index,
                "continuation_step": continuation_step,
            }
        )

    exception_recoveries: list[dict[str, Any]] = []
    exception_indices = [
        index
        for index, record in enumerate(events[:completion_index])
        if record.get("event") == "training_exception"
    ]
    for exception_index in exception_indices:
        exception_step = events[exception_index].get("step")
        require(isinstance(exception_step, int) and exception_step >= 0, "training exception has an invalid step")
        later_resumes = [item for item in resumes if item["event_index"] > exception_index]
        require(later_resumes, f"training exception at event {exception_index} has no later verified resume")
        recovery = later_resumes[0]
        intervening = events[exception_index + 1 : recovery["event_index"]]
        allowed_intervening = RESTART_PREFLIGHT_EVENTS | {"emergency_checkpoint_verified"}
        require(
            all(record.get("event") in allowed_intervening for record in intervening),
            f"training advanced or published an unexpected event before recovery of exception {exception_index}",
        )
        require(
            recovery["step"] <= exception_step,
            f"training exception at event {exception_index} resumed beyond its uncommitted step",
        )
        require(
            recovery["continuation_step"] > exception_step
            or (
                exception_step == EXPECTED_TOTAL_STEPS
                and recovery["continuation_step"] == exception_step
            ),
            f"training exception at event {exception_index} was not replayed through its interrupted phase",
        )
        exception_recoveries.append(
            {
                "event_index": exception_index,
                "interrupted_step": exception_step,
                "resume_event_index": recovery["event_index"],
                "resume_step": recovery["step"],
                "resume_model_state_sha256": recovery["model_state_sha256"],
                "anchor_event_index": recovery["anchor_event_index"],
                "rollback_steps": exception_step - recovery["step"],
                "continuation_event_index": recovery["continuation_event_index"],
                "continuation_step": recovery["continuation_step"],
            }
        )

    pause_recoveries: list[dict[str, Any]] = []
    pause_indices = [
        index
        for index, record in enumerate(events[:completion_index])
        if record.get("event") == "run_paused"
    ]
    for pause_index in pause_indices:
        pause_step = events[pause_index].get("step")
        require(isinstance(pause_step, int) and pause_step >= 0, "run_paused event has an invalid step")
        later_resumes = [item for item in resumes if item["event_index"] > pause_index]
        require(later_resumes, f"run_paused at event {pause_index} has no later verified resume")
        recovery = later_resumes[0]
        require(recovery["step"] == pause_step, f"run_paused at event {pause_index} resumed from a different step")
        require(
            all(
                record.get("event") in RESTART_PREFLIGHT_EVENTS
                for record in events[pause_index + 1 : recovery["event_index"]]
            ),
            f"unexpected event appeared between run_paused {pause_index} and its resume",
        )
        pause_recoveries.append(
            {
                "event_index": pause_index,
                "step": pause_step,
                "resume_event_index": recovery["event_index"],
                "anchor_event_index": recovery["anchor_event_index"],
                "continuation_event_index": recovery["continuation_event_index"],
                "continuation_step": recovery["continuation_step"],
            }
        )

    return {
        "resume_count": len(resumes),
        "resumes": resumes,
        "prior_training_exception_count": len(exception_indices),
        "training_exception_recoveries": exception_recoveries,
        "prior_pause_count": len(pause_indices),
        "pause_recoveries": pause_recoveries,
        "all_interruptions_recovered": True,
    }


def verify_final_learning_rate_endpoint(
    events: Sequence[Mapping[str, Any]],
    completion_index: int,
    provenance: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-bind the final logged/checkpointed LR to the last applied update."""

    require(0 <= completion_index < len(events), "completion event index is invalid for LR verification")
    require(events[completion_index].get("event") == "run_complete", "LR verification is not anchored to run_complete")
    require(completion_index == len(events) - 1, "LR verification requires run_complete to be the final event")
    require(provenance.get("contract_identity_sha256") == EXPECTED_CONTRACT_IDENTITY, "LR proof has the wrong contract")
    require(provenance.get("run_uuid") == EXPECTED_RUN_UUID, "LR proof has the wrong run UUID")
    require(provenance.get("trainer_sha256") == EXPECTED_TRAINER_SHA256, "LR proof is not bound to the sealed trainer")
    require(provenance.get("config_file_sha256") == EXPECTED_CONFIG_SHA256, "LR proof is not bound to the sealed config")
    require(provenance.get("experiment_sha256") == EXPECTED_EXPERIMENT_SHA256, "LR proof is not bound to the sealed experiment")
    require(checkpoint.get("step") == EXPECTED_TOTAL_STEPS, "LR proof checkpoint is not at step 300000")
    require(checkpoint.get("raw_unfinalized") is True, "LR proof checkpoint is not proven raw and unfinalized")

    config = provenance.get("config")
    require(isinstance(config, dict), "LR proof has no bound config payload")
    schedule = derive_final_learning_rate_contract(config)
    checkpoint_schedule = checkpoint.get("learning_rate_endpoint")
    require(isinstance(checkpoint_schedule, dict), "final checkpoint has no LR endpoint proof")
    for key, value in schedule.items():
        require(checkpoint_schedule.get(key) == value, f"final checkpoint LR endpoint disagrees at {key}")
    require(
        checkpoint_schedule.get("optimizer_param_group_learning_rates")
        == [schedule["last_applied_learning_rate"]],
        "final checkpoint optimizer LR is not the exact last-applied value",
    )

    bound_openings: list[int] = []
    for index, record in enumerate(events[: completion_index + 1]):
        if record.get("event") != "run_opened":
            continue
        require(record.get("run_uuid") == EXPECTED_RUN_UUID, f"run_opened {index} has the wrong run UUID")
        require(
            record.get("contract_identity_sha256") == EXPECTED_CONTRACT_IDENTITY,
            f"run_opened {index} has the wrong contract identity",
        )
        bound_openings.append(index)
    require(bound_openings, "event log has no bound run_opened event")
    completion = events[completion_index]
    require(
        completion.get("contract_identity_sha256") == EXPECTED_CONTRACT_IDENTITY,
        "run_complete is not bound to the sealed contract",
    )

    progress = [
        (index, record)
        for index, record in enumerate(events[:completion_index])
        if record.get("event") == "train_progress" and record.get("step") == EXPECTED_TOTAL_STEPS
    ]
    require(progress, "event log has no step-300000 train_progress record")
    deterministic_records: list[dict[str, Any]] = []
    progress_sessions: list[int] = []
    for index, record in progress:
        require(set(record) == FINAL_PROGRESS_KEYS, f"step-300000 train_progress {index} has unexpected fields")
        require(record.get("total_steps") == EXPECTED_TOTAL_STEPS, f"step-300000 train_progress {index} has the wrong total")
        logged_lr = _finite_float(record.get("learning_rate"), f"step-300000 train_progress LR at event {index}")
        require(
            logged_lr == schedule["last_applied_learning_rate"],
            f"step-300000 train_progress LR at event {index} is not the exact last-applied value",
        )
        _finite_float(record.get("loss"), f"step-300000 loss at event {index}")
        _finite_float(record.get("loss_mean_recent"), f"step-300000 recent loss at event {index}")
        _finite_float(record.get("steps_per_second"), f"step-300000 rate at event {index}")
        _finite_float(record.get("eta_hours"), f"step-300000 ETA at event {index}")
        _finite_float(record.get("free_disk_gb"), f"step-300000 free disk at event {index}")
        require(
            isinstance(record.get("peak_vram_bytes"), int) and not isinstance(record.get("peak_vram_bytes"), bool),
            f"step-300000 peak VRAM at event {index} is malformed",
        )
        try:
            timestamp = dt.datetime.fromisoformat(str(record.get("time")))
        except ValueError as error:
            raise AuditFailure(f"step-300000 train_progress {index} has an invalid timestamp") from error
        require(timestamp.tzinfo is not None, f"step-300000 train_progress {index} timestamp is not timezone-aware")
        openings = [opening for opening in bound_openings if opening < index]
        require(openings, f"step-300000 train_progress {index} has no preceding bound run_opened")
        opening = openings[-1]
        session_prefix = events[opening + 1 : index]
        resumes = [record for record in session_prefix if record.get("event") == "resumed"]
        initials = [record for record in session_prefix if record.get("event") == "initial_checkpoint_verified"]
        require(
            (len(resumes) == 1 and not initials) or (len(initials) == 1 and not resumes),
            f"step-300000 train_progress {index} has no unambiguous session start",
        )
        if resumes:
            resume_step = resumes[0].get("step")
            require(
                isinstance(resume_step, int) and 0 <= resume_step < EXPECTED_TOTAL_STEPS,
                f"step-300000 train_progress {index} has an invalid resume anchor",
            )
        else:
            require(initials[0].get("step") == 0, f"step-300000 train_progress {index} has a nonzero initial anchor")
        progress_sessions.append(opening)
        deterministic_records.append({key: record[key] for key in FINAL_PROGRESS_DETERMINISTIC_KEYS})
    require(
        len(set(progress_sessions)) == len(progress_sessions),
        "multiple step-300000 train_progress records came from one trainer session",
    )
    require(
        all(record == deterministic_records[0] for record in deterministic_records[1:]),
        "duplicate step-300000 train_progress records disagree on deterministic training values",
    )

    final_checkpoint_events = [
        (index, record)
        for index, record in enumerate(events[:completion_index])
        if record.get("event") in VERIFIED_CHECKPOINT_EVENTS
        and record.get("step") == EXPECTED_TOTAL_STEPS
    ]
    require(len(final_checkpoint_events) == 1, "event log does not have exactly one verified final checkpoint event")
    checkpoint_index, checkpoint_event = final_checkpoint_events[0]
    require(checkpoint_index > progress[-1][0], "verified final checkpoint does not follow the final progress record")
    all_verified_checkpoint_events = [
        (index, record)
        for index, record in enumerate(events[:completion_index])
        if record.get("event") in VERIFIED_CHECKPOINT_EVENTS
    ]
    require(all_verified_checkpoint_events, "event log has no verified checkpoint events")
    require(
        checkpoint_index == all_verified_checkpoint_events[-1][0],
        "the step-300000 checkpoint is not the last verified checkpoint event",
    )
    require(checkpoint_event.get("path") == checkpoint.get("path"), "final checkpoint event path is not authoritative")
    require(
        checkpoint_event.get("pointer") == checkpoint.get("pointer_path"),
        "final checkpoint event pointer is not bound to checkpoints/latest.json",
    )
    require(checkpoint_event.get("sha256") == checkpoint.get("sha256"), "final checkpoint event file hash disagrees")
    require(
        checkpoint_event.get("model_state_sha256") == checkpoint.get("model_state_sha256"),
        "final checkpoint event model hash disagrees",
    )

    return {
        **schedule,
        "trainer_sha256": EXPECTED_TRAINER_SHA256,
        "config_file_sha256": EXPECTED_CONFIG_SHA256,
        "experiment_sha256": EXPECTED_EXPERIMENT_SHA256,
        "contract_identity_sha256": EXPECTED_CONTRACT_IDENTITY,
        "run_uuid": EXPECTED_RUN_UUID,
        "step_300000_progress_record_count": len(progress),
        "step_300000_progress_event_indices": [index for index, _ in progress],
        "progress_deterministic_payload_sha256": canonical_sha256(deterministic_records[0]),
        "duplicate_progress_records_agree_exact": True,
        "final_checkpoint_event_index": checkpoint_index,
        "final_checkpoint_event_type": checkpoint_event["event"],
        "final_checkpoint_cross_binding_exact": True,
        "final_checkpoint_pointer_binding_exact": True,
        "final_checkpoint_is_last_verified_checkpoint_event": True,
        "raw_checkpoint_unfinalized": True,
        "theoretical_zero_endpoint_remained_unapplied": True,
    }


def preflight_completion(run_dir: Path) -> dict[str, Any]:
    receipt_path = run_dir / RECEIPT_NAME
    require(not receipt_path.exists(), f"audit receipt already exists; refusing to overwrite it: {receipt_path}")

    contract_path = run_dir / "run_contract.json"
    report_path = run_dir / "final_report.json"
    status_path = run_dir / "status.json"
    events_path = run_dir / "events.jsonl"
    for path in (contract_path, report_path, status_path, events_path):
        require_complete(path.is_file(), f"completed-run artifact is not present: {path}")

    contract = load_json_object(contract_path)
    report = load_json_object(report_path)
    status = load_json_object(status_path)
    events = load_jsonl(events_path)
    require(report_path.read_bytes() == status_path.read_bytes(), "final_report.json and status.json are not byte-identical")
    require(report == status, "final_report.json and status.json disagree")
    require(report.get("status") == "complete", "final report status is not complete")
    require(report.get("step") == EXPECTED_TOTAL_STEPS, "final report step is not 300000")
    require(report.get("total_steps") == EXPECTED_TOTAL_STEPS, "final report total_steps is not 300000")
    require(report.get("peak_vram_pass") is True, "final report peak-VRAM gate did not pass")
    peak = report.get("peak_vram_bytes")
    ceiling = report.get("maximum_peak_vram_bytes")
    require(isinstance(peak, int) and isinstance(ceiling, int) and 0 < peak <= ceiling, "final report peak-VRAM values are invalid")
    bound_config = contract.get("static", {}).get("config", {})
    try:
        expected_ceiling = int(float(bound_config["safety"]["maximum_peak_vram_gb"]) * 1e9)
    except (KeyError, TypeError, ValueError) as error:
        raise AuditFailure("run contract has no valid peak-VRAM ceiling") from error
    require(ceiling == expected_ceiling, "final report peak-VRAM ceiling differs from the bound config")

    completion_indices = [index for index, record in enumerate(events) if record.get("event") == "run_complete"]
    require_complete(len(completion_indices) == 1, "event log does not have exactly one run_complete record")
    completion_index = completion_indices[0]
    require(completion_index == len(events) - 1, "run_complete is not the final event")
    later_exceptions = [
        record for record in events[completion_index + 1 :] if record.get("event") == "training_exception"
    ]
    require(not later_exceptions, "a training exception follows run_complete")
    completion_event = events[completion_index]
    event_payload = dict(completion_event)
    event_payload.pop("event", None)
    event_payload.pop("time", None)
    require(event_payload == report, "run_complete payload disagrees with final_report.json")
    recovery_lineage = validate_recovery_lineage(events, completion_index)

    return {
        "contract": contract,
        "report": report,
        "status": status,
        "events": events,
        "completion_event": completion_event,
        "completion_event_index": completion_index,
        "completion_event_count": len(completion_indices),
        "prior_training_exception_count": recovery_lineage["prior_training_exception_count"],
        "recovery_lineage": recovery_lineage,
        "paths": {
            "contract": contract_path,
            "report": report_path,
            "status": status_path,
            "events": events_path,
        },
    }


def git_value(repo: Path, expression: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", expression],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return result.stdout.strip()


def _git_bytes(repo: Path, arguments: Sequence[str], context: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        timeout=30,
    )
    require(result.returncode == 0, f"cannot verify {context}: {result.stderr.decode('utf-8', 'replace').strip()}")
    return result.stdout


def _nul_paths(blob: bytes, context: str) -> list[str]:
    require(not blob or blob.endswith(b"\0"), f"git emitted an unterminated path inventory for {context}")
    try:
        return [item.decode("utf-8") for item in blob.split(b"\0") if item]
    except UnicodeDecodeError as error:
        raise AuditFailure(f"git emitted a non-UTF-8 path for {context}") from error


def verify_source_worktree_inventory(
    repo: Path,
    observations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    repo = repo.resolve()
    require(repo.is_dir() and not repo.is_symlink(), f"source repository is missing or a symlink: {repo}")
    tracked_status = _git_bytes(
        repo,
        ["status", "--porcelain=v1", "-z", "--untracked-files=no"],
        "tracked source worktree status",
    )
    require(tracked_status == b"", "source worktree has staged, modified, or deleted tracked files")

    head_paths = _nul_paths(
        _git_bytes(repo, ["ls-tree", "-r", "-z", "--name-only", "HEAD", "--", "hs_tasnet"], "HEAD package inventory"),
        "HEAD package inventory",
    )
    index_paths = _nul_paths(
        _git_bytes(repo, ["ls-files", "-z", "--", "hs_tasnet"], "tracked package inventory"),
        "tracked package inventory",
    )
    head_python = sorted(path for path in head_paths if path.endswith(".py"))
    index_python = sorted(path for path in index_paths if path.endswith(".py"))
    require(index_python == head_python, "tracked hs_tasnet Python inventory differs from HEAD")
    require("hs_tasnet/__init__.py" in head_python, "tracked package __init__.py is missing")
    require("hs_tasnet/trainer.py" in head_python, "tracked trainer.py imported by package __init__ is missing")

    package_dir = repo / "hs_tasnet"
    require(package_dir.is_dir() and not package_dir.is_symlink(), "on-disk hs_tasnet package is missing or a symlink")
    disk_python_paths = sorted(package_dir.rglob("*.py"))
    disk_python: list[str] = []
    for path in disk_python_paths:
        require(path.is_file() and not path.is_symlink(), f"on-disk package Python entry is not a regular file: {path}")
        resolved = path.resolve()
        require(resolved.is_relative_to(package_dir), f"on-disk package Python entry escapes hs_tasnet: {path}")
        disk_python.append(str(resolved.relative_to(repo)))
    require(disk_python == head_python, "on-disk hs_tasnet Python inventory differs from tracked HEAD inventory")

    file_hashes: dict[str, str] = {}
    for relative in head_python:
        path = repo / relative
        head_blob = _git_bytes(repo, ["cat-file", "blob", f"HEAD:{relative}"], f"HEAD bytes for {relative}")
        disk_blob = path.read_bytes()
        require(disk_blob == head_blob, f"tracked package Python file differs byte-for-byte from HEAD: {relative}")
        digest = observe_file(path, observations)
        require(digest == hashlib.sha256(head_blob).hexdigest(), f"tracked package Python hash differs from HEAD: {relative}")
        file_hashes[relative] = digest

    init_tree = ast.parse((repo / "hs_tasnet/__init__.py").read_bytes(), filename="hs_tasnet/__init__.py")
    trainer_imports = [
        node
        for node in ast.walk(init_tree)
        if isinstance(node, ast.ImportFrom) and node.module == "hs_tasnet.trainer"
    ]
    require(trainer_imports, "package __init__.py no longer imports hs_tasnet.trainer")
    require(
        {alias.name for node in trainer_imports for alias in node.names} >= {"Trainer", "MusDB18HQ"},
        "package __init__.py trainer import surface is unexpected",
    )
    return {
        "tracked_worktree_status_clean": True,
        "tracked_index_equals_head_package_python_inventory": True,
        "on_disk_equals_tracked_package_python_inventory": True,
        "tracked_package_python_files_equal_head_bytes": True,
        "package_init_imports_trainer": True,
        "package_python_file_count": len(head_python),
        "package_python_files": head_python,
        "package_python_file_sha256": file_hashes,
        "package_python_inventory_sha256": canonical_sha256(file_hashes),
        "trainer_sha256": file_hashes["hs_tasnet/trainer.py"],
    }


def verify_provenance(
    run_dir: Path,
    preflight: Mapping[str, Any],
    observations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    contract = preflight["contract"]
    report = preflight["report"]
    require(contract.get("schema_version") == 1, "run contract schema is invalid")
    static = contract.get("static")
    require(isinstance(static, dict), "run contract static payload is missing")
    identity = canonical_sha256(static)
    require(contract.get("static_identity_sha256") == identity, "run contract static identity is invalid")
    require(identity == EXPECTED_CONTRACT_IDENTITY, "run contract is not the sealed c91 production contract")
    require(report.get("contract_identity_sha256") == identity, "final report is bound to the wrong run contract")
    run_uuid = contract.get("run_uuid")
    require(isinstance(run_uuid, str), "run UUID is missing")
    try:
        uuid.UUID(run_uuid)
    except ValueError as error:
        raise AuditFailure("run UUID is malformed") from error
    require(run_uuid == EXPECTED_RUN_UUID, "run UUID is not the launched c91 production run")

    expected_bound_hashes = {
        "trainer_sha256": EXPECTED_TRAINER_SHA256,
        "config_file_sha256": EXPECTED_CONFIG_SHA256,
        "manifest_file_sha256": EXPECTED_MANIFEST_FILE_SHA256,
        "manifest_content_sha256": EXPECTED_MANIFEST_CONTENT_SHA256,
    }
    for name, expected in expected_bound_hashes.items():
        require(static.get(name) == expected, f"sealed run contract has an unexpected {name}")

    expected_paths = {
        "trainer_path": BASE_DIR / "train_production.py",
        "config_path": BASE_DIR / "full_config.json",
        "manifest_path": BASE_DIR / "manifests/combined.manifest.json",
    }
    hashes = {
        "trainer_path": "trainer_sha256",
        "config_path": "config_file_sha256",
        "manifest_path": "manifest_file_sha256",
    }
    resolved_paths: dict[str, Path] = {}
    for key, expected in expected_paths.items():
        raw = static.get(key)
        require(isinstance(raw, str), f"run contract is missing {key}")
        path = Path(raw).expanduser().resolve()
        require(path == expected.resolve(), f"run contract {key} is unexpected: {path}")
        digest = observe_file(path, observations)
        require(digest == static.get(hashes[key]), f"run contract hash mismatch for {key}")
        resolved_paths[key] = path

    config = load_json_object(resolved_paths["config_path"])
    require(config == static.get("config"), "run contract config payload differs from the config file")
    require(config.get("schedule", {}).get("total_steps") == EXPECTED_TOTAL_STEPS, "bound config is not the 300000-step production schedule")
    require(tuple(config.get("model", {}).get("sources", ())) == EXPECTED_SOURCE_NAMES, "bound source order is invalid")

    manifest = load_json_object(resolved_paths["manifest_path"])
    unhashed_manifest = dict(manifest)
    recorded_content_hash = unhashed_manifest.pop("content_sha256", None)
    manifest_content_hash = canonical_sha256(unhashed_manifest)
    require(recorded_content_hash == manifest_content_hash, "manifest content hash is invalid")
    require(manifest_content_hash == static.get("manifest_content_sha256"), "run contract manifest content hash mismatch")
    require(report.get("manifest_file_sha256") == static.get("manifest_file_sha256"), "final report manifest file hash mismatch")
    require(report.get("manifest_content_sha256") == manifest_content_hash, "final report manifest content hash mismatch")

    source = config.get("source")
    require(isinstance(source, dict), "source identity is missing from the bound config")
    require(source.get("commit") == EXPECTED_SOURCE_COMMIT, "bound source commit is not c91")
    require(source.get("tree") == EXPECTED_SOURCE_TREE, "bound source tree is not c91")
    require(
        source.get("files", {}).get("research/experiment.py") == EXPECTED_EXPERIMENT_SHA256,
        "bound c91 experiment hash is unexpected",
    )
    repo = Path(source.get("repo", "")).expanduser().resolve()
    require(repo.is_dir(), f"source repository is missing: {repo}")
    require(git_value(repo, "HEAD") == source.get("commit"), "source repository commit changed")
    require(git_value(repo, "HEAD^{tree}") == source.get("tree"), "source repository tree changed")
    source_hashes: dict[str, str] = {}
    for relative, expected_hash in source.get("files", {}).items():
        path = (repo / relative).resolve()
        require(path.is_relative_to(repo), f"source file escapes repository: {relative}")
        digest = observe_file(path, observations)
        require(digest == expected_hash, f"source file hash changed: {relative}")
        source_hashes[relative] = digest
    source_worktree_inventory = verify_source_worktree_inventory(repo, observations)

    return {
        "run_dir": str(run_dir),
        "run_uuid": run_uuid,
        "contract_identity_sha256": identity,
        "contract_file_sha256": observe_file(preflight["paths"]["contract"], observations),
        "trainer_sha256": static["trainer_sha256"],
        "config_file_sha256": static["config_file_sha256"],
        "experiment_sha256": source_hashes["research/experiment.py"],
        "manifest_file_sha256": static["manifest_file_sha256"],
        "manifest_content_sha256": manifest_content_hash,
        "source_repo": str(repo),
        "source_commit": source["commit"],
        "source_tree": source["tree"],
        "source_file_sha256": source_hashes,
        "source_worktree_inventory": source_worktree_inventory,
        "config": config,
        "source_repo_path": repo,
    }


def import_cpu_model(source_repo: Path, torch: Any) -> Any:
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA visibility was not disabled")
    require(not torch.cuda.is_available() and torch.cuda.device_count() == 0, "audit process can see a CUDA device")
    require(not torch.cuda.is_initialized(), "CUDA was initialized in the audit process")
    sys.dont_write_bytecode = True
    repo_text = str(source_repo)
    if repo_text in sys.path:
        sys.path.remove(repo_text)
    sys.path.insert(0, repo_text)
    module = importlib.import_module("hs_tasnet.hs_tasnet")
    require(Path(module.__file__).resolve() == source_repo / "hs_tasnet/hs_tasnet.py", "imported the wrong HS-TasNet module")
    package = sys.modules.get("hs_tasnet")
    trainer = sys.modules.get("hs_tasnet.trainer")
    require(package is not None and getattr(package, "__file__", None), "HS-TasNet package import is missing")
    require(trainer is not None and getattr(trainer, "__file__", None), "package __init__ did not import hs_tasnet.trainer")
    require(
        Path(package.__file__).resolve() == source_repo / "hs_tasnet/__init__.py",
        "imported the wrong HS-TasNet package initializer",
    )
    require(
        Path(trainer.__file__).resolve() == source_repo / "hs_tasnet/trainer.py",
        "package initializer imported the wrong HS-TasNet trainer module",
    )
    require(not torch.cuda.is_available() and not torch.cuda.is_initialized(), "model import initialized or exposed CUDA")
    return module.HSTasNet


def tensor_state_sha256(state: Mapping[str, Any], torch: Any) -> str:
    digest = hashlib.sha256()

    def add(blob: bytes) -> None:
        digest.update(len(blob).to_bytes(8, "big"))
        digest.update(blob)

    for name, value in state.items():
        require(isinstance(value, torch.Tensor), f"model state is not a tensor: {name}")
        cpu = value.detach().to("cpu").contiguous()
        add(name.encode("utf-8"))
        add(str(cpu.dtype).encode("ascii"))
        add(canonical_json_bytes(list(cpu.shape)))
        add(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def structured_state_sha256(value: Any, torch: Any, np: Any) -> str:
    digest = hashlib.sha256()

    def add(blob: bytes) -> None:
        digest.update(len(blob).to_bytes(8, "big"))
        digest.update(blob)

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            cpu = item.detach().to("cpu").contiguous()
            add(b"torch.Tensor")
            add(str(cpu.dtype).encode("ascii"))
            add(canonical_json_bytes(list(cpu.shape)))
            add(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            add(b"numpy.ndarray")
            add(str(array.dtype).encode("ascii"))
            add(canonical_json_bytes(list(array.shape)))
            add(array.view(np.uint8).tobytes())
        elif isinstance(item, dict):
            add(b"dict")
            ordered = sorted(item.items(), key=lambda pair: (type(pair[0]).__name__, repr(pair[0])))
            add(str(len(ordered)).encode("ascii"))
            for key, child in ordered:
                visit(key)
                visit(child)
        elif isinstance(item, list):
            add(b"list")
            add(str(len(item)).encode("ascii"))
            for child in item:
                visit(child)
        elif isinstance(item, tuple):
            add(b"tuple")
            add(str(len(item)).encode("ascii"))
            for child in item:
                visit(child)
        elif isinstance(item, bytes):
            add(b"bytes")
            add(item)
        elif isinstance(item, str):
            add(b"str")
            add(item.encode("utf-8"))
        elif item is None:
            add(b"None")
        elif isinstance(item, bool):
            add(b"bool:true" if item else b"bool:false")
        elif isinstance(item, (int, np.integer)):
            add(b"int")
            add(str(int(item)).encode("ascii"))
        elif isinstance(item, (float, np.floating)):
            add(b"float")
            add(float(item).hex().encode("ascii"))
        else:
            raise AuditFailure(f"unsupported checkpoint state type: {type(item)!r}")

    visit(value)
    return digest.hexdigest()


def verify_sha_sidecar(
    path: Path,
    observations: dict[str, dict[str, Any]],
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    require(sidecar.is_file() and not sidecar.is_symlink(), f"SHA sidecar is missing or a symlink: {sidecar}")
    fields = sidecar.read_text(encoding="ascii").split()
    require(len(fields) == 2, f"malformed SHA sidecar: {sidecar}")
    recorded_hash, recorded_name = fields
    require(bool(SHA256_PATTERN.fullmatch(recorded_hash)), f"malformed SHA-256 in sidecar: {sidecar}")
    require(recorded_name == path.name, f"SHA sidecar names the wrong artifact: {sidecar}")
    actual_hash = observe_file(path, observations)
    sidecar_hash = observe_file(sidecar, observations)
    require(recorded_hash == actual_hash, f"artifact SHA sidecar mismatch: {path}")
    if expected_sha256 is not None:
        require(actual_hash == expected_sha256, f"artifact hash differs from final report: {path}")
    return {
        "path": str(path),
        "sha256": actual_hash,
        "sidecar_path": str(sidecar),
        "sidecar_sha256": sidecar_hash,
        "size_bytes": path.stat().st_size,
    }


def instantiate_package(path: Path, HSTasNet: Any, torch: Any) -> tuple[dict[str, Any], Any, bytes, dict[str, Any]]:
    package = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(package, dict), f"model package is not a mapping: {path}")
    require(set(package) == {"model", "config"}, f"model package keys are unexpected: {path}")
    state = package["model"]
    config_blob = package["config"]
    require(isinstance(state, dict) and isinstance(config_blob, bytes), f"model package payload is malformed: {path}")
    config = pickle.loads(config_blob)
    require(isinstance(config, dict), f"pickled model config is malformed: {path}")
    model = HSTasNet(**config)
    model.load_state_dict(state, strict=True)
    model.to(torch.device("cpu"))
    model.eval()
    return package, model, config_blob, config


def model_is_raw(model: Any, config: Mapping[str, Any], torch: Any) -> bool:
    scales = model.output_source_scales.detach().cpu()
    return (
        not bool(config.get("decoder_hann_baked", False))
        and not bool(model.conv_decode.hann_window_baked)
        and torch.equal(scales, torch.full_like(scales, 0.5))
    )


def resolve_report_artifact(run_dir: Path, raw_path: Any, expected_name: str) -> Path:
    require(isinstance(raw_path, str), f"final report path is missing for {expected_name}")
    path = Path(raw_path).expanduser()
    require(path.is_absolute(), f"final report artifact path is not absolute: {raw_path}")
    require(path.resolve() == (run_dir / expected_name).resolve(), f"final report points to the wrong {expected_name}")
    require(not path.is_symlink(), f"final artifact is a symlink: {path}")
    return path


def verify_final_checkpoint(
    run_dir: Path,
    report: Mapping[str, Any],
    config: Mapping[str, Any],
    contract_identity: str,
    HSTasNet: Any,
    torch: Any,
    np: Any,
    observations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    checkpoint_dir = run_dir / "checkpoints"
    pointer_path = checkpoint_dir / "latest.json"
    pointer_hash = observe_file(pointer_path, observations)
    pointer = load_json_object(pointer_path)
    require(pointer.get("schema_version") == 1 and pointer.get("kind") == "hs_tasnet_c91_checkpoint_pointer", "checkpoint pointer schema is invalid")
    require(pointer.get("step") == EXPECTED_TOTAL_STEPS, "latest checkpoint pointer is not at step 300000")
    generation = pointer.get("generation")
    relative = pointer.get("path")
    require(isinstance(generation, str) and isinstance(relative, str), "checkpoint pointer paths are malformed")
    match = CHECKPOINT_PATTERN.fullmatch(generation)
    require(match is not None and int(match.group(1)) == EXPECTED_TOTAL_STEPS, "final checkpoint filename is malformed or has the wrong step")
    path = (run_dir / relative).resolve()
    require(path == (checkpoint_dir / generation).resolve(), "checkpoint pointer escapes or disagrees with its generation")
    file_record = verify_sha_sidecar(path, observations, expected_sha256=pointer.get("sha256"))

    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(payload, dict), "final checkpoint payload is not a mapping")
    require(payload.get("schema_version") == 1 and payload.get("kind") == "hs_tasnet_c91_raw_resume", "final checkpoint schema is invalid")
    require(payload.get("contract_identity_sha256") == contract_identity, "final checkpoint is bound to the wrong contract")
    require(payload.get("step") == EXPECTED_TOTAL_STEPS, "final checkpoint global step is not 300000")
    optimizer_meta = payload.get("optimizer_meta")
    require(isinstance(optimizer_meta, dict), "final checkpoint optimizer metadata is missing")
    require(optimizer_meta.get("autoresearch_step") == EXPECTED_TOTAL_STEPS, "final checkpoint optimizer counter is not 300000")
    expected_lr = float(config["model"]["learning_rate"])
    require(tuple(float(value) for value in optimizer_meta.get("base_lrs", ())) == (expected_lr,), "final checkpoint base learning rate is invalid")
    learning_rate_endpoint = verify_checkpoint_optimizer_learning_rate(payload, config)
    serialized_adam_topology = verify_serialized_adam_topology(payload.get("optimizer"))

    model_state = payload.get("model")
    require(isinstance(model_state, dict), "final checkpoint model state is missing")
    model_hash = tensor_state_sha256(model_state, torch)
    require(model_hash == payload.get("model_state_sha256"), "final checkpoint model-state hash is invalid")
    require(model_hash == pointer.get("model_state_sha256"), "final checkpoint model-state hash differs from the pointer")
    require(model_hash == report.get("raw_artifact", {}).get("model_state_sha256"), "final checkpoint does not cross-bind to the raw artifact")
    require(structured_state_sha256(payload.get("optimizer"), torch, np) == payload.get("optimizer_state_sha256"), "final checkpoint optimizer hash is invalid")
    require(structured_state_sha256(payload.get("scaler"), torch, np) == payload.get("scaler_state_sha256"), "final checkpoint scaler hash is invalid")
    require(structured_state_sha256(payload.get("rng_state"), torch, np) == payload.get("rng_state_sha256"), "final checkpoint RNG hash is invalid")

    model_config_blob = payload.get("model_config")
    require(isinstance(model_config_blob, bytes), "final checkpoint model config is malformed")
    model_config = pickle.loads(model_config_blob)
    require(isinstance(model_config, dict), "final checkpoint model config is malformed")
    checkpoint_model = HSTasNet(**model_config)
    checkpoint_model.load_state_dict(model_state, strict=True)
    checkpoint_model.to(torch.device("cpu"))
    require(model_is_raw(checkpoint_model, model_config, torch), "final checkpoint model is calibrated or baked")
    optimizer = torch.optim.Adam(checkpoint_model.parameters(), lr=expected_lr)
    optimizer.load_state_dict(payload["optimizer"])
    loaded_group_lrs = [
        _finite_float(group.get("lr"), f"strictly reloaded optimizer param-group LR {index}")
        for index, group in enumerate(optimizer.param_groups)
    ]
    require(
        loaded_group_lrs == learning_rate_endpoint["optimizer_param_group_learning_rates"],
        "strictly reloaded optimizer param-group learning rates changed",
    )
    adam_coverage = verify_loaded_adam_coverage(checkpoint_model, optimizer, torch)
    require(
        serialized_adam_topology["stateless_grouped_parameter_index"]
        == adam_coverage["stateless_grouped_parameter_index"],
        "serialized and strictly reloaded Adam stateless parameter identities disagree",
    )
    recent_losses = payload.get("recent_losses")
    require(isinstance(recent_losses, list) and all(math.isfinite(float(value)) for value in recent_losses), "final checkpoint recent losses are malformed")

    result = {
        **file_record,
        "pointer_path": str(pointer_path),
        "pointer_sha256": pointer_hash,
        "step": EXPECTED_TOTAL_STEPS,
        "model_state_sha256": model_hash,
        "optimizer_state_sha256": payload["optimizer_state_sha256"],
        "scaler_state_sha256": payload["scaler_state_sha256"],
        "rng_state_sha256": payload["rng_state_sha256"],
        "optimizer_autoresearch_step": optimizer_meta["autoresearch_step"],
        "learning_rate_endpoint": learning_rate_endpoint,
        "strict_reload_optimizer_param_group_learning_rates": loaded_group_lrs,
        "raw_unfinalized": True,
        "serialized_adam_topology": serialized_adam_topology,
        **adam_coverage,
        "model_config_sha256": hashlib.sha256(model_config_blob).hexdigest(),
    }
    del optimizer, checkpoint_model, model_state, payload
    gc.collect()
    return result


def current_rss_bytes() -> int:
    status = Path("/proc/self/status").read_text(encoding="ascii")
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            fields = line.split()
            require(len(fields) >= 2, "cannot parse process VmRSS")
            return int(fields[1]) * 1024
    raise AuditFailure("process VmRSS is unavailable")


def deterministic_chunk(index: int, torch: Any, *, variant: int = 0) -> Any:
    axis = torch.arange(512, dtype=torch.float32)
    frequency = 0.0013 + index * 0.000017 + variant * 0.00031
    left = torch.sin(axis * frequency + variant * 0.2)
    right = torch.cos(axis * (frequency * 1.07) - variant * 0.15)
    return torch.stack((left, right)) * 0.1


def assert_close(actual: Any, expected: Any, torch: Any, *, message: str) -> None:
    try:
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    except AssertionError as error:
        raise AuditFailure(message) from error


def verify_streaming(raw_model: Any, deployment_model: Any, gains: Sequence[float], torch: Any) -> dict[str, Any]:
    raw_model.eval()
    deployment_model.eval()
    raw_model.set_output_source_gains(torch.tensor(gains, dtype=torch.float32))
    require(not raw_model.conv_decode.hann_window_baked, "cross-mode reference decoder was unexpectedly baked")
    require(deployment_model.conv_decode.hann_window_baked, "cross-mode deployment decoder is not baked")

    unbaked_transform = raw_model.init_stateful_transform_fn(device="cpu")
    baked_transform = deployment_model.init_stateful_transform_fn(device="cpu")
    maximum_cross_mode_error = 0.0
    stream_peak = 0.0
    rss_after_warmup: int | None = None
    replay_expected: list[Any] = []
    with torch.inference_mode():
        for index in range(LONG_STREAM_CALLBACKS):
            chunk = deterministic_chunk(index, torch)
            unbaked_output = unbaked_transform(chunk)
            baked_output = baked_transform(chunk)
            require(tuple(baked_output.shape) == EXPECTED_STREAM_SHAPE, f"unexpected streaming output shape: {tuple(baked_output.shape)}")
            require(bool(torch.isfinite(unbaked_output).all()) and bool(torch.isfinite(baked_output).all()), "streaming output is non-finite")
            assert_close(baked_output, unbaked_output, torch, message=f"baked/unbaked streaming parity failed at callback {index}")
            if index < RESET_REPLAY_CALLBACKS:
                replay_expected.append(baked_output.clone())
            maximum_cross_mode_error = max(
                maximum_cross_mode_error,
                float((baked_output - unbaked_output).abs().max()),
            )
            stream_peak = max(stream_peak, float(baked_output.abs().max()))
            if index + 1 == STREAM_WARMUP_CALLBACKS:
                gc.collect()
                rss_after_warmup = current_rss_bytes()
    gc.collect()
    rss_after_stream = current_rss_bytes()
    require(rss_after_warmup is not None, "streaming RSS warmup boundary was not reached")
    rss_growth = max(0, rss_after_stream - rss_after_warmup)
    require(rss_growth <= MAX_STREAM_RSS_GROWTH_BYTES, f"streaming RSS grew by {rss_growth} bytes")
    require(rss_after_stream <= MAX_STREAM_RSS_BYTES, f"streaming process RSS is {rss_after_stream} bytes")
    require(stream_peak > 0.0 and math.isfinite(stream_peak), "streaming output is silent or invalid")

    # The first closure has consumed the full stream. Recreate state from scratch
    # and replay its prefix to prove reset semantics, not merely two parallel runs.
    del baked_transform
    reset_transform = deployment_model.init_stateful_transform_fn(device="cpu")
    with torch.inference_mode():
        for index, expected in enumerate(replay_expected):
            replayed = reset_transform(deterministic_chunk(index, torch))
            require(torch.equal(replayed, expected), f"streaming reset/replay failed at callback {index}")

    causal_a = deployment_model.init_stateful_transform_fn(device="cpu")
    causal_b = deployment_model.init_stateful_transform_fn(device="cpu")
    common_prefix_callbacks = 4
    with torch.inference_mode():
        for index in range(common_prefix_callbacks):
            chunk = deterministic_chunk(10_000 + index, torch)
            prefix_a = causal_a(chunk)
            prefix_b = causal_b(chunk)
            require(torch.equal(prefix_a, prefix_b), f"streaming common-prefix causality failed at callback {index}")
        divergent_a = causal_a(deterministic_chunk(20_000, torch, variant=0))
        divergent_b = causal_b(deterministic_chunk(20_000, torch, variant=1))
    require(bool(torch.isfinite(divergent_a).all()) and bool(torch.isfinite(divergent_b).all()), "causality suffix output is non-finite")
    divergent_suffix_difference = float((divergent_a - divergent_b).abs().max())
    require(divergent_suffix_difference > 0.0, "causality suffix inputs did not produce divergent outputs")

    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_rss_bytes = int(max_rss * 1024 if sys.platform.startswith("linux") else max_rss)
    return {
        "device": "cpu",
        "cross_mode_reference": "calibrated_unbaked_conv_transpose_vs_baked_single_frame_linear",
        "callbacks": LONG_STREAM_CALLBACKS,
        "warmup_callbacks": STREAM_WARMUP_CALLBACKS,
        "shape": list(EXPECTED_STREAM_SHAPE),
        "finite": True,
        "stream_peak": stream_peak,
        "maximum_cross_mode_absolute_error": maximum_cross_mode_error,
        "reset_determinism_exact": True,
        "reset_replay_callbacks": RESET_REPLAY_CALLBACKS,
        "common_prefix_callbacks": common_prefix_callbacks,
        "common_prefix_causality_exact": True,
        "divergent_suffix_max_difference": divergent_suffix_difference,
        "rss_after_warmup_bytes": rss_after_warmup,
        "rss_after_stream_bytes": rss_after_stream,
        "rss_growth_bytes": rss_growth,
        "rss_growth_limit_bytes": MAX_STREAM_RSS_GROWTH_BYTES,
        "rss_absolute_limit_bytes": MAX_STREAM_RSS_BYTES,
        "process_max_rss_bytes": max_rss_bytes,
    }


def verify_artifacts_and_transformation(
    run_dir: Path,
    report: Mapping[str, Any],
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    HSTasNet: Any,
    torch: Any,
    observations: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw_report = report.get("raw_artifact")
    deploy_report = report.get("deployment")
    require(isinstance(raw_report, dict) and isinstance(deploy_report, dict), "final report artifact metadata is missing")
    raw_path = resolve_report_artifact(run_dir, raw_report.get("path"), "final-training-raw.pt")
    deployment_path = resolve_report_artifact(run_dir, deploy_report.get("path"), "final-deployment.pt")
    raw_file = verify_sha_sidecar(raw_path, observations, expected_sha256=raw_report.get("sha256"))
    deployment_file = verify_sha_sidecar(deployment_path, observations, expected_sha256=deploy_report.get("sha256"))

    metadata_path = deployment_path.with_suffix(deployment_path.suffix + ".json")
    metadata_hash = observe_file(metadata_path, observations)
    deployment_metadata = load_json_object(metadata_path)
    require(deployment_metadata == deploy_report, "deployment metadata file differs from final_report.json")
    require(deploy_report.get("decoder_hann_baked") is True, "deployment report does not assert decoder baking")
    require(deploy_report.get("bake_output_parity") is True, "deployment report does not assert bake parity")
    require(deploy_report.get("strict_reload_output_parity") is True, "deployment report does not assert strict reload parity")
    require(deploy_report.get("streaming_callbacks") == 16, "final deployment did not run 16 streaming callbacks")
    require(tuple(deploy_report.get("streaming_shape", ())) == EXPECTED_STREAM_SHAPE, "reported deployment streaming shape is invalid")

    raw_package, raw_model, raw_config_blob, raw_config = instantiate_package(raw_path, HSTasNet, torch)
    deployment_package, deployment_model, deployment_config_blob, deployment_config = instantiate_package(
        deployment_path, HSTasNet, torch
    )
    raw_state_hash = tensor_state_sha256(raw_model.state_dict(), torch)
    deployment_state_hash = tensor_state_sha256(deployment_model.state_dict(), torch)
    require(raw_state_hash == raw_report.get("model_state_sha256") == checkpoint.get("model_state_sha256"), "raw artifact state is not cross-bound to the final checkpoint")
    require(deployment_state_hash == deploy_report.get("model_state_sha256"), "deployment model-state hash is invalid")
    require(hashlib.sha256(raw_config_blob).hexdigest() == checkpoint.get("model_config_sha256"), "raw artifact config differs from the final checkpoint")
    require(model_is_raw(raw_model, raw_config, torch), "final training artifact is calibrated or baked")

    gains = tuple(float(value) for value in config["finalization"]["source_gains"])
    require(gains == EXPECTED_SOURCE_GAINS, f"bound production gains are unexpected: {gains}")
    require(tuple(float(value) for value in deploy_report.get("source_gains", ())) == gains, "deployment source gains differ from bound config")
    expected_scales = torch.tensor(gains, dtype=torch.float32) * 0.5
    require(torch.equal(deployment_model.output_source_scales, expected_scales), "deployment persistent source scales are invalid")
    require(bool(deployment_config.get("decoder_hann_baked")) and deployment_model.conv_decode.hann_window_baked, "deployment artifact is not persistently baked")
    reported_scales = torch.tensor(deploy_report.get("persistent_output_source_scales", ()), dtype=torch.float32)
    require(torch.equal(reported_scales, expected_scales), "reported deployment scales are invalid")

    raw_config_without_bake = dict(raw_config)
    deployment_config_without_bake = dict(deployment_config)
    raw_bake_flag = bool(raw_config_without_bake.pop("decoder_hann_baked", False))
    deployment_bake_flag = bool(deployment_config_without_bake.pop("decoder_hann_baked", False))
    require(not raw_bake_flag and deployment_bake_flag, "raw/deployment config bake flags are invalid")
    require(raw_config_without_bake == deployment_config_without_bake, "deployment config changed beyond decoder baking")

    raw_state = raw_model.state_dict()
    deployment_state = deployment_model.state_dict()
    require(tuple(raw_state) == tuple(deployment_state), "raw/deployment state keys differ")
    changed_keys: list[str] = []
    for name, raw_value in raw_state.items():
        deployed_value = deployment_state[name]
        if raw_value.is_floating_point():
            require(bool(torch.isfinite(raw_value).all()), f"raw state contains non-finite values at {name}")
            require(bool(torch.isfinite(deployed_value).all()), f"deployment state contains non-finite values at {name}")
        if name == "output_source_scales":
            expected = expected_scales
        elif name == "conv_decode.weight":
            expected = raw_value * raw_model.conv_decode.window
        else:
            expected = raw_value
        require(torch.equal(deployed_value, expected), f"deployment state changed unexpectedly at {name}")
        if not torch.equal(deployed_value, raw_value):
            changed_keys.append(name)
    require(set(changed_keys) == {"output_source_scales", "conv_decode.weight"}, f"unexpected raw/deployment changed keys: {changed_keys}")

    raw_source_scales = [float(value) for value in raw_model.output_source_scales]
    streaming = verify_streaming(raw_model, deployment_model, gains, torch)
    require(not torch.cuda.is_available() and not torch.cuda.is_initialized(), "CUDA became visible or initialized during artifact verification")

    raw_record = {
        **raw_file,
        "model_state_sha256": raw_state_hash,
        "model_config_sha256": hashlib.sha256(raw_config_blob).hexdigest(),
        "strict_cpu_reload": True,
        "raw_source_scales": raw_source_scales,
        "decoder_hann_baked": False,
    }
    deployment_record = {
        **deployment_file,
        "metadata_path": str(metadata_path),
        "metadata_sha256": metadata_hash,
        "model_state_sha256": deployment_state_hash,
        "model_config_sha256": hashlib.sha256(deployment_config_blob).hexdigest(),
        "strict_cpu_reload": True,
        "persistent_output_source_scales": [float(value) for value in deployment_model.output_source_scales],
        "decoder_hann_baked": True,
    }
    transformation = {
        "post_optimizer_only": True,
        "source_gains": list(gains),
        "changed_state_keys": sorted(changed_keys),
        "unchanged_state_tensor_count": len(raw_state) - len(changed_keys),
        "decoder_weight_equals_raw_times_hann_exact": True,
        "all_other_persistent_tensors_equal_exact": True,
        "raw_and_deployment_config_equal_except_bake_flag": True,
    }
    del raw_package, deployment_package, raw_state, deployment_state, raw_model, deployment_model
    gc.collect()
    return raw_record, deployment_record, {"transformation": transformation, "streaming": streaming}


def find_temporary_files(run_dir: Path) -> list[str]:
    temporary: list[str] = []
    for path in run_dir.rglob("*"):
        if path.is_file() and (path.name.endswith(".tmp") or (path.name.startswith(".") and ".tmp" in path.name)):
            temporary.append(str(path.relative_to(run_dir)))
    return sorted(temporary)


def atomic_write_new_json(path: Path, value: Mapping[str, Any]) -> str:
    require(not path.exists(), f"refusing to overwrite existing receipt: {path}")
    content = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise AuditFailure(f"audit receipt appeared concurrently: {path}") from error
        os.unlink(temporary)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return hashlib.sha256(content).hexdigest()


def build_receipt(
    run_dir: Path,
    service: str,
    service_evidence: Mapping[str, Any],
    companion_service_state: Mapping[str, str] | None,
    preflight: Mapping[str, Any],
    provenance: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    raw_artifact: Mapping[str, Any],
    deployment: Mapping[str, Any],
    functional: Mapping[str, Any],
    step250_diagnostic: Mapping[str, Any],
    observations: Mapping[str, Mapping[str, Any]],
    torch: Any,
) -> dict[str, Any]:
    report = preflight["report"]
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "kind": "hs_tasnet_c91_final_audit_receipt",
        "audit_status": "pass",
        "audited_at_utc": utc_now(),
        "auditor": {
            "path": str(AUDITOR_PATH),
            "sha256": sha256_file(AUDITOR_PATH),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_cuda_initialized": torch.cuda.is_initialized(),
        },
        "service": {
            "name": service,
            **dict(service_evidence),
            "companion_trainer": (
                {"name": companion_trainer_name(service), **dict(companion_service_state)}
                if companion_service_state is not None
                else None
            ),
        },
        "run": {key: value for key, value in provenance.items() if key not in {"config", "source_repo_path"}},
        "completion": {
            "status": report["status"],
            "step": report["step"],
            "total_steps": report["total_steps"],
            "completed_at_utc": report["completed_at_utc"],
            "final_report_sha256": observations[str(preflight["paths"]["report"].resolve())]["sha256"],
            "status_sha256": observations[str(preflight["paths"]["status"].resolve())]["sha256"],
            "events_sha256": observations[str(preflight["paths"]["events"].resolve())]["sha256"],
            "completion_event_index": preflight["completion_event_index"],
            "completion_event_count": preflight["completion_event_count"],
            "prior_training_exception_count": preflight["prior_training_exception_count"],
            "recovery_lineage": preflight["recovery_lineage"],
            "no_exception_after_completion": True,
            "report_status_event_agreement": True,
            "peak_vram_bytes": report["peak_vram_bytes"],
            "maximum_peak_vram_bytes": report["maximum_peak_vram_bytes"],
            "peak_vram_pass": report["peak_vram_pass"],
        },
        "artifacts": {
            "final_checkpoint": dict(checkpoint),
            "final_training_raw": dict(raw_artifact),
            "final_deployment": dict(deployment),
        },
        "step250000_diagnostic_prerequisite": dict(step250_diagnostic),
        "functional_verification": dict(functional),
        "filesystem": {
            "temporary_files": [],
            "audited_input_file_count": len(observations),
            "audited_input_files": dict(sorted(observations.items())),
        },
    }
    receipt["audit_payload_sha256"] = canonical_sha256(receipt)
    return receipt


def run_audit(run_dir: Path, service: str) -> tuple[Path, str, dict[str, Any]]:
    run_dir = run_dir.expanduser().resolve()
    require_complete(run_dir.is_dir(), f"run directory does not exist: {run_dir}")
    initial_service_state = read_service_state(service)
    validate_service_not_active(initial_service_state, service)
    initial_companion_state = validate_companion_trainer_inactive(service)

    with hold_completed_run_lock(run_dir):
        # Re-read under the lock so final files cannot belong to a still-running trainer.
        service_state = read_service_state(service)
        validate_service_not_active(service_state, service)
        companion_service_state = validate_companion_trainer_inactive(service)
        preflight = preflight_completion(run_dir)
        service_evidence = prove_service_complete(service_state, service, run_dir, preflight)
        step250_diagnostic = verify_step250_diagnostic_terminal()
        observations: dict[str, dict[str, Any]] = {}
        for path in preflight["paths"].values():
            observe_file(path, observations)
        provenance = verify_provenance(run_dir, preflight, observations)

        import numpy as np
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            require(torch.get_num_interop_threads() == 1, "Torch interop threads are not one")
        HSTasNet = import_cpu_model(provenance["source_repo_path"], torch)
        checkpoint = verify_final_checkpoint(
            run_dir,
            preflight["report"],
            provenance["config"],
            provenance["contract_identity_sha256"],
            HSTasNet,
            torch,
            np,
            observations,
        )
        learning_rate_endpoint = verify_final_learning_rate_endpoint(
            preflight["events"],
            preflight["completion_event_index"],
            provenance,
            checkpoint,
        )
        checkpoint = {**checkpoint, "event_log_learning_rate_endpoint": learning_rate_endpoint}
        raw_artifact, deployment, functional = verify_artifacts_and_transformation(
            run_dir,
            preflight["report"],
            provenance["config"],
            checkpoint,
            HSTasNet,
            torch,
            observations,
        )
        temporary_files = find_temporary_files(run_dir)
        require(not temporary_files, f"temporary files remain under the completed run: {temporary_files}")
        require(not torch.cuda.is_available() and not torch.cuda.is_initialized(), "CUDA was exposed or initialized during the audit")
        verify_observations_stable(observations)
        final_source_observations: dict[str, dict[str, Any]] = {}
        final_source_inventory = verify_source_worktree_inventory(
            provenance["source_repo_path"],
            final_source_observations,
        )
        require(
            final_source_inventory == provenance["source_worktree_inventory"],
            "source worktree or executable Python inventory changed during the audit",
        )
        verify_observations_stable(final_source_observations)
        require(
            git_value(provenance["source_repo_path"], "HEAD") == provenance["source_commit"],
            "source repository commit changed during the audit",
        )
        require(
            git_value(provenance["source_repo_path"], "HEAD^{tree}") == provenance["source_tree"],
            "source repository tree changed during the audit",
        )

        final_service_state = read_service_state(service)
        final_service_evidence = prove_service_complete(final_service_state, service, run_dir, preflight)
        final_companion_state = validate_companion_trainer_inactive(service)
        require(
            final_service_evidence.get("invocation_id") == service_evidence.get("invocation_id"),
            "service invocation changed during the audit",
        )
        first_terminal_cursor = service_evidence.get("terminal_stdout_end_cursor")
        final_terminal_cursor = final_service_evidence.get("terminal_stdout_end_cursor")
        if first_terminal_cursor is not None and final_terminal_cursor is not None:
            require(final_terminal_cursor == first_terminal_cursor, "terminal service evidence changed during the audit")
        if companion_service_state is not None:
            require(final_companion_state is not None, "companion trainer state disappeared during the audit")
        revalidate_step250_diagnostic_terminal(step250_diagnostic)
        receipt = build_receipt(
            run_dir,
            service,
            final_service_evidence,
            final_companion_state or initial_companion_state,
            preflight,
            provenance,
            checkpoint,
            raw_artifact,
            deployment,
            functional,
            step250_diagnostic,
            observations,
            torch,
        )
        receipt_path = run_dir / RECEIPT_NAME
        receipt_file_sha256 = atomic_write_new_json(receipt_path, receipt)
        return receipt_path, receipt_file_sha256, receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt_path, receipt_hash, receipt = run_audit(args.run_dir, args.service)
    except IncompleteRun as error:
        print(json.dumps({"status": "incomplete", "receipt_written": False, "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    except BaseException as error:
        print(json.dumps({"status": "failed", "receipt_written": False, "error": repr(error)}, sort_keys=True), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "pass",
                "receipt_written": True,
                "receipt_path": str(receipt_path),
                "receipt_file_sha256": receipt_hash,
                "audit_payload_sha256": receipt["audit_payload_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
