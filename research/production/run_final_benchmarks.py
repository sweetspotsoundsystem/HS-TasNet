#!/usr/bin/env python3
"""Crash-safe final-deployment benchmark driver for the c91 production run.

This is deliberately separate from the sealed trainer.  It runs only after the
completion watchdog has validated the final CPU audit and uses the frozen
autoresearch benchmark validator rather than maintaining a second timing
schema here.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
import statistics
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence


BASE_DIR = Path(__file__).resolve().parent
RUN_DIR = BASE_DIR / "runs/c91-full-best200-v1-seed60"
SOURCE_COMMIT = "e5b62db805dcd5cd91225843c1cb07c12a453a86"
SOURCE_TREE = "79dab041ad89d852aa5b8d2233195d42e1ec6441"
RUN_CONFIG_SHA256 = "76cc662c65c6f2152d313f4152c6f2e7c85b9d17f46e179f5791a69dc35026fe"
BENCHMARK_SHA256 = "391a4be642cbcb80476120aa7266b80c63ee2fd01c413b12e2d855232bc93187"
ORCHESTRATOR_SHA256 = "c84f5b4e54d06cdf125ee855acd5b729c707786216a196cef2be3068999324fe"
EXPECTED_PARAMETER_COUNT = 30_127_210
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STAGE_NAMES = ("benchmark_1", "benchmark_2", "benchmark_3", "cpu_diagnostic")
RESULT_TOP_LEVEL_KEYS = {
    "schema_version",
    "checkpoint",
    "local_package_path",
    "requested_device",
    "device",
    "device_identity",
    "dtype",
    "runtime",
    "runtime_profile_pass",
    "execution_backend",
    "precision",
    "model",
    "correctness",
    "prefix_causality",
    "impulse_alignment",
    "algorithmic_latency_samples",
    "algorithmic_latency_ms",
    "latency_pass",
    "timing",
    "memory",
    "memory_growth_mb",
    "peak_vram_mb",
    "determinism_pass",
    "threading_pass",
    "constraint_pass",
}


class FinalBenchmarkError(RuntimeError):
    """Base class for fail-closed runner errors."""


class Contradiction(FinalBenchmarkError):
    """Persistent evidence is malformed or contradicts the frozen contract."""


class NotReady(FinalBenchmarkError):
    """The final audited publication is not available yet."""


class Busy(FinalBenchmarkError):
    """Another cooperative process owns the production run lock."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Contradiction(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_utc_timestamp(value: Any, label: str) -> str:
    require(isinstance(value, str), f"{label} is not a string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise Contradiction(f"{label} is not ISO-8601") from error
    require(parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed), f"{label} is not UTC")
    require(value.endswith("+00:00") and parsed.isoformat() == value, f"{label} is not canonical UTC")
    return value


def _reject_constant(token: str) -> None:
    raise Contradiction(f"non-finite JSON constant: {token}")


def _without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(path: Path) -> tuple[dict[str, Any], bytes]:
    require(path.is_file() and not path.is_symlink(), f"missing/symlink JSON: {path}")
    content = path.read_bytes()
    try:
        value = json.loads(
            content,
            object_pairs_hook=_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Contradiction(f"malformed JSON: {path}") from error
    require(isinstance(value, dict), f"JSON is not an object: {path}")
    return value, content


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        require(stat.S_ISREG(os.fstat(descriptor).st_mode), f"cannot fsync non-regular file: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_replace(path: Path, content: bytes) -> None:
    temporary = path.with_name(
        f".finalbench-atomic-{path.name}-{os.getpid()}-{uuid.uuid4().hex}.tmp"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def pair_state(path: Path) -> str:
    sidecar = sidecar_path(path)
    data = path.exists() or path.is_symlink()
    seal = sidecar.exists() or sidecar.is_symlink()
    if data and seal:
        return "pair"
    if data:
        return "json_only"
    if seal:
        return "sidecar_only"
    return "absent"


def verify_sidecar(path: Path) -> str:
    sidecar = sidecar_path(path)
    require(path.is_file() and not path.is_symlink(), f"invalid sealed JSON: {path}")
    require(sidecar.is_file() and not sidecar.is_symlink(), f"invalid sidecar: {sidecar}")
    digest = sha256_file(path)
    expected = f"{digest}  {path.name}\n".encode("ascii")
    require(sidecar.read_bytes() == expected, f"sidecar mismatch: {sidecar}")
    return digest


def seal_existing_json(path: Path) -> str:
    digest = sha256_file(path)
    atomic_replace(sidecar_path(path), f"{digest}  {path.name}\n".encode("ascii"))
    return digest


OWN_ATOMIC_TEMP_RE = re.compile(
    r"^\.finalbench-atomic-(?P<target>.+)-(?P<pid>[1-9][0-9]*)-"
    r"(?P<nonce>[0-9a-f]{32})\.tmp$"
)


def cleanup_own_atomic_temps(
    *directories: Path,
    proc_root: Path = Path("/proc"),
) -> list[str]:
    require(proc_root.is_dir() and not proc_root.is_symlink(), f"cannot inspect process table: {proc_root}")
    removed: list[str] = []
    for directory in directories:
        if not directory.exists():
            continue
        require(directory.is_dir() and not directory.is_symlink(), f"invalid atomic-temp directory: {directory}")
        stale: list[Path] = []
        for path in directory.iterdir():
            match = OWN_ATOMIC_TEMP_RE.fullmatch(path.name)
            if match is None:
                continue
            owner_pid = int(match.group("pid"))
            require(owner_pid != os.getpid(), f"runner atomic temp is owned by current process: {path}")
            owner_path = proc_root / str(owner_pid)
            require(
                not (owner_path.exists() or owner_path.is_symlink()),
                f"runner atomic temp owner is still live: {path} (pid={owner_pid})",
            )
            require(path.is_file() and not path.is_symlink(), f"invalid runner atomic temp: {path}")
            stale.append(path)
        for path in stale:
            path.unlink()
            removed.append(str(path))
        if stale:
            fsync_directory(directory)
    return sorted(removed)


def load_sealed_document(
    path: Path,
    validator: Callable[[Mapping[str, Any]], None],
) -> tuple[dict[str, Any], str] | None:
    state = pair_state(path)
    require(state != "sidecar_only", f"orphan sealed-document sidecar: {path}")
    if state == "absent":
        return None
    value, _ = strict_json(path)
    validator(value)
    digest = seal_existing_json(path) if state == "json_only" else verify_sidecar(path)
    return value, digest


def publish_sealed_document(
    path: Path,
    value: Mapping[str, Any],
    validator: Callable[[Mapping[str, Any]], None],
) -> tuple[dict[str, Any], str]:
    require(pair_state(path) == "absent", f"refusing to replace sealed document: {path}")
    validator(value)
    atomic_replace(path, canonical_json_bytes(value))
    validator(strict_json(path)[0])
    digest = seal_existing_json(path)
    return dict(value), digest


def _git(repo: Path, *arguments: str) -> str:
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": "/nonexistent",
        "LC_ALL": "C",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_ATTR_NOSYSTEM": "1",
    }
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    require(completed.returncode == 0, f"git {' '.join(arguments)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"{name}_{uuid.uuid4().hex}", path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


@dataclass(frozen=True)
class Stage:
    name: str
    device: str
    args: Mapping[str, Any]
    timeout_seconds: float
    canonical: Path
    staging: Path
    launch: Path
    execution: Path


@dataclass(frozen=True)
class Context:
    run_dir: Path
    output_dir: Path
    receipt_path: Path
    deployment_path: Path
    deployment_sha256: str
    deployment_model_state_sha256: str
    final_report_sha256: str
    audit_receipt_sha256: str
    audit_payload_sha256: str
    contract_identity_sha256: str
    source_repo: Path
    source_commit: str
    source_tree: str
    run_config_path: Path
    run_config_sha256: str
    benchmark_script: Path
    benchmark_sha256: str
    orchestrator_path: Path
    orchestrator_sha256: str
    watchdog_path: Path
    watchdog_sha256: str
    runner_path: Path
    runner_sha256: str
    full_config_path: Path
    full_config_sha256: str
    frozen_files: Mapping[str, str]
    run_evidence_files: Mapping[str, Mapping[str, str]]
    trainer_state_evidence: Mapping[str, Mapping[str, str]]
    benchmark_args: Mapping[str, Any]
    benchmark_timeout_seconds: float
    cpu_args: Mapping[str, Any]
    cpu_timeout_seconds: float
    validator: Any = field(repr=False, compare=False)
    trainer_services: tuple[str, ...] = ()
    enforce_trainer_inactive: bool = False
    enforce_source_identity: bool = False
    proc_root: Path = Path("/proc")


def _stage_args(config: Mapping[str, Any], name: str) -> tuple[Mapping[str, Any], float]:
    raw = config.get(name)
    require(isinstance(raw, dict), f"run_config.{name} is malformed")
    args = raw.get("args")
    timeout = raw.get("timeout_seconds")
    require(isinstance(args, dict), f"run_config.{name}.args is malformed")
    require(
        isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and timeout > 0,
        f"run_config.{name}.timeout_seconds is invalid",
    )
    require(
        not {"checkpoint", "json-output", "report-only"}.intersection(args),
        f"run_config.{name} contains orchestrator-owned arguments",
    )
    return args, float(timeout)


def _validate_protocol(config: Mapping[str, Any], repo: Path) -> tuple[Mapping[str, Any], float, Mapping[str, Any], float]:
    require(config.get("schema_version") == 1, "run_config schema changed")
    require(config.get("primary_benchmark_repetitions") == 3, "primary repetitions must be exactly 3")
    frozen = config.get("frozen_files")
    require(
        isinstance(frozen, dict) and frozen.get("research/benchmark_streaming.py") == BENCHMARK_SHA256,
        "run_config benchmark identity changed",
    )
    for name in ("benchmark", "cpu_diagnostic"):
        raw = config.get(name)
        require(isinstance(raw, dict), f"run_config.{name} is malformed")
        require(raw.get("script") == "research/benchmark_streaming.py", f"run_config.{name}.script changed")
    gpu, gpu_timeout = _stage_args(config, "benchmark")
    cpu, cpu_timeout = _stage_args(config, "cpu_diagnostic")
    require(gpu_timeout == cpu_timeout, "GPU/CPU benchmark timeouts differ")
    require(gpu.get("device") == "cuda:0" and cpu.get("device") == "cpu", "benchmark devices changed")
    require(gpu.get("callbacks") == 10_000 and cpu.get("callbacks") == 10_000, "callback count changed")
    require(not {"expected-cuda-device-name", "expected-cuda-device-uuid"}.intersection(cpu), "CPU protocol contains GPU identity")
    normalized_gpu = {
        key: value
        for key, value in gpu.items()
        if key != "device" and key not in {"expected-cuda-device-name", "expected-cuda-device-uuid"}
    }
    require(normalized_gpu == {key: value for key, value in cpu.items() if key != "device"}, "GPU/CPU protocols differ")
    require((repo / str(config["benchmark"]["script"])).resolve() == (repo / "research/benchmark_streaming.py").resolve(), "benchmark script escaped source")
    return gpu, gpu_timeout, cpu, cpu_timeout


def verify_frozen_files(repo: Path, frozen: Mapping[str, str]) -> None:
    require(isinstance(frozen, Mapping) and frozen, "run_config frozen_files is empty/malformed")
    for relative, expected in frozen.items():
        require(isinstance(relative, str) and relative and Path(relative).as_posix() == relative, f"invalid frozen path: {relative!r}")
        require(isinstance(expected, str) and SHA256_RE.fullmatch(expected) is not None, f"invalid frozen hash: {relative}")
        path = (repo / relative).resolve()
        require(path.is_relative_to(repo), f"frozen path escaped source: {relative}")
        require(path.is_file() and not path.is_symlink(), f"frozen file is missing/symlink: {relative}")
        require(sha256_file(path) == expected, f"frozen file hash changed: {relative}")


def verify_source_worktree(repo: Path) -> None:
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    require(not status, f"source worktree changed: {status}")
    ignored_text = _git(repo, "ls-files", "--others", "--ignored", "--exclude-standard")
    importable_suffixes = (".py", ".pyo", ".so", ".pyd", ".dll", ".dylib", ".zip", ".egg", ".whl", ".pth")
    forbidden = [
        path
        for path in ignored_text.splitlines()
        if path.startswith("hs_tasnet/") and not ("/__pycache__/" in path and path.endswith(".pyc"))
        or path.lower().endswith(importable_suffixes)
    ]
    require(not forbidden, "ignored importable source payloads are forbidden: " + ", ".join(forbidden))


def read_trainer_services(services: Sequence[str]) -> dict[str, dict[str, str]]:
    evidence: dict[str, dict[str, str]] = {}
    for service in services:
        completed = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                service,
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
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        require(completed.returncode == 0, f"cannot inspect trainer service {service}: {completed.stderr.strip()}")
        state = dict(line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line)
        require(state.get("LoadState") in {"loaded", "not-found"}, f"trainer service load state is unexpected: {service} ({state})")
        require(state.get("ActiveState") not in {"active", "activating", "reloading", "deactivating"}, f"trainer service is not inactive: {service} ({state})")
        require(
            (state.get("ActiveState"), state.get("SubState"))
            in {("inactive", "dead"), ("failed", "failed"), ("failed", "dead")},
            f"trainer service is not terminal dead/failed: {service} ({state})",
        )
        require(state.get("MainPID") in {"", "0"}, f"trainer service still has a PID: {service} ({state})")
        invocation = state.get("InvocationID", "")
        require(invocation == "" or re.fullmatch(r"[0-9a-f]{32}", invocation) is not None, f"trainer invocation is malformed: {service}")
        evidence[service] = {
            "LoadState": state["LoadState"],
            "ActiveState": state["ActiveState"],
            "SubState": state["SubState"],
            "MainPID": state.get("MainPID", ""),
            "InvocationID": invocation,
        }
    return evidence


def verify_trainer_services_inactive(context: Context) -> None:
    if not context.enforce_trainer_inactive:
        return
    observed = read_trainer_services(context.trainer_services)
    for service, bound in context.trainer_state_evidence.items():
        current = observed.get(service)
        require(isinstance(current, Mapping), f"trainer service disappeared: {service}")
        bound_invocation = bound.get("InvocationID", "")
        require(current.get("InvocationID") == bound_invocation, f"trainer invocation changed: {service}")


def build_context(run_dir: Path = RUN_DIR, output_dir: Path | None = None) -> Context:
    run_dir = run_dir.expanduser().resolve()
    watchdog_path = BASE_DIR / "completion_watchdog.py"
    watchdog = load_module(watchdog_path, "hs_tasnet_completion_watchdog")
    state = watchdog.publication_state(run_dir)
    if state.get("status") != "audited_complete":
        raise NotReady(f"publication is {state.get('status')}, not audited_complete")

    report, report_bytes = strict_json(run_dir / "final_report.json")
    audit, audit_bytes = strict_json(run_dir / "final_audit_receipt.json")
    deployment = report.get("deployment")
    require(isinstance(deployment, dict), "final report deployment is malformed")
    deployment_path = Path(str(deployment.get("path", ""))).expanduser().resolve()
    require(deployment_path == (run_dir / "final-deployment.pt").resolve(), "deployment path changed")
    deployment_sha = str(deployment.get("sha256", ""))
    model_sha = str(deployment.get("model_state_sha256", ""))
    require(SHA256_RE.fullmatch(deployment_sha) is not None, "deployment hash is invalid")
    require(SHA256_RE.fullmatch(model_sha) is not None, "deployment model-state hash is invalid")
    require(deployment_path.is_file() and not deployment_path.is_symlink(), "deployment is missing/symlink")
    require(sha256_file(deployment_path) == deployment_sha, "deployment hash mismatch")
    require(sidecar_path(deployment_path).read_bytes() == f"{deployment_sha}  {deployment_path.name}\n".encode("ascii"), "deployment sidecar mismatch")
    metadata, _ = strict_json(deployment_path.with_suffix(deployment_path.suffix + ".json"))
    require(metadata == deployment, "deployment metadata differs from final report")
    audit_deployment = audit.get("artifacts", {}).get("final_deployment")
    require(isinstance(audit_deployment, dict), "audit deployment evidence is malformed")
    require(audit_deployment.get("sha256") == deployment_sha, "audit deployment hash changed")
    require(audit_deployment.get("model_state_sha256") == model_sha, "audit deployment model hash changed")
    report_sha = hashlib.sha256(report_bytes).hexdigest()
    audit_sha = hashlib.sha256(audit_bytes).hexdigest()
    require(state.get("receipt_sha256") == audit_sha, "watchdog/audit receipt hash mismatch")
    require(state.get("publication", {}).get("final_report_sha256") == report_sha, "watchdog/report hash mismatch")

    full_config_path = BASE_DIR / "full_config.json"
    full_config, full_config_bytes = strict_json(full_config_path)
    source = full_config.get("source")
    require(isinstance(source, dict), "production source binding is malformed")
    repo = Path(str(source.get("repo", ""))).expanduser().resolve()
    require(source.get("commit") == SOURCE_COMMIT and source.get("tree") == SOURCE_TREE, "production source identity changed")
    require(repo.is_dir() and not repo.is_symlink(), "source repository is missing/symlink")
    require(_git(repo, "rev-parse", "HEAD") == SOURCE_COMMIT, "source HEAD changed")
    require(_git(repo, "rev-parse", "HEAD^{tree}") == SOURCE_TREE, "source tree changed")
    run_config_path = repo / "research/run_config.json"
    benchmark_script = repo / "research/benchmark_streaming.py"
    orchestrator_path = repo / "research/run_experiment.py"
    for path, expected, label in (
        (run_config_path, RUN_CONFIG_SHA256, "run_config"),
        (benchmark_script, BENCHMARK_SHA256, "benchmark"),
        (orchestrator_path, ORCHESTRATOR_SHA256, "orchestrator"),
    ):
        require(path.is_file() and not path.is_symlink(), f"frozen {label} is missing/symlink")
        require(sha256_file(path) == expected, f"frozen {label} hash changed")
    run_config, _ = strict_json(run_config_path)
    gpu_args, gpu_timeout, cpu_args, cpu_timeout = _validate_protocol(run_config, repo)
    frozen = run_config.get("frozen_files")
    require(isinstance(frozen, dict), "run_config frozen_files is malformed")
    verify_frozen_files(repo, frozen)
    require(not (repo / "research/__init__.py").exists(), "research/__init__.py would alter frozen imports")
    verify_source_worktree(repo)
    require(not any(name == "research" or name.startswith("research.") for name in sys.modules), "research modules were imported before source verification")
    old_pycache_prefix = sys.pycache_prefix
    old_dont_write = sys.dont_write_bytecode
    try:
        sys.pycache_prefix = "/nonexistent/hs-tasnet-final-benchmark-pycache"
        sys.dont_write_bytecode = True
        validator = load_module(orchestrator_path, "hs_tasnet_frozen_run_experiment")
    finally:
        sys.pycache_prefix = old_pycache_prefix
        sys.dont_write_bytecode = old_dont_write
    try:
        validator._verify_frozen_files(frozen)
    except Exception as error:
        raise Contradiction(f"frozen orchestrator rejected source files: {error}") from error

    chosen_output = (output_dir or (run_dir / "final-benchmarks")).expanduser().resolve()
    require(chosen_output.parent == run_dir, "final benchmark output must be a direct run-directory child")
    audit_payload = str(audit.get("audit_payload_sha256", ""))
    require(SHA256_RE.fullmatch(audit_payload) is not None, "audit payload hash is invalid")
    contract_identity = str(audit.get("run", {}).get("contract_identity_sha256", ""))
    require(SHA256_RE.fullmatch(contract_identity) is not None, "audit contract identity is invalid")
    evidence_paths = {
        "final_report": run_dir / "final_report.json",
        "status": run_dir / "status.json",
        "events": run_dir / "events.jsonl",
        "final_audit_receipt": run_dir / "final_audit_receipt.json",
        "deployment_metadata": deployment_path.with_suffix(deployment_path.suffix + ".json"),
        "deployment_sidecar": sidecar_path(deployment_path),
        "run_contract": run_dir / "run_contract.json",
        "checkpoint_pointer": run_dir / "checkpoints/latest.json",
    }
    run_evidence_files: dict[str, dict[str, str]] = {}
    for name, path in evidence_paths.items():
        require(path.is_file() and not path.is_symlink(), f"bound run evidence is missing/symlink: {name}")
        run_evidence_files[name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    require(run_evidence_files["final_report"]["sha256"] == report_sha, "run evidence report hash changed")
    require(run_evidence_files["final_audit_receipt"]["sha256"] == audit_sha, "run evidence audit hash changed")
    require(state.get("publication", {}).get("events_sha256") == run_evidence_files["events"]["sha256"], "watchdog/events hash mismatch")
    completion = audit.get("completion")
    require(isinstance(completion, dict), "audit completion evidence is malformed")
    require(completion.get("status_sha256") == run_evidence_files["status"]["sha256"], "audit/status hash mismatch")
    require(completion.get("events_sha256") == run_evidence_files["events"]["sha256"], "audit/events hash mismatch")
    trainer_services = (watchdog.PRIMARY_SERVICE, watchdog.AUTORESTART_SERVICE)
    trainer_state_evidence = read_trainer_services(trainer_services)
    runner_path = Path(__file__).resolve()
    context = Context(
        run_dir=run_dir,
        output_dir=chosen_output,
        receipt_path=run_dir / "final-benchmark-receipt.json",
        deployment_path=deployment_path,
        deployment_sha256=deployment_sha,
        deployment_model_state_sha256=model_sha,
        final_report_sha256=report_sha,
        audit_receipt_sha256=audit_sha,
        audit_payload_sha256=audit_payload,
        contract_identity_sha256=contract_identity,
        source_repo=repo,
        source_commit=SOURCE_COMMIT,
        source_tree=SOURCE_TREE,
        run_config_path=run_config_path,
        run_config_sha256=RUN_CONFIG_SHA256,
        benchmark_script=benchmark_script,
        benchmark_sha256=BENCHMARK_SHA256,
        orchestrator_path=orchestrator_path,
        orchestrator_sha256=ORCHESTRATOR_SHA256,
        watchdog_path=watchdog_path,
        watchdog_sha256=sha256_file(watchdog_path),
        runner_path=runner_path,
        runner_sha256=sha256_file(runner_path),
        full_config_path=full_config_path,
        full_config_sha256=hashlib.sha256(full_config_bytes).hexdigest(),
        frozen_files=dict(frozen),
        run_evidence_files=run_evidence_files,
        trainer_state_evidence=trainer_state_evidence,
        benchmark_args=gpu_args,
        benchmark_timeout_seconds=gpu_timeout,
        cpu_args=cpu_args,
        cpu_timeout_seconds=cpu_timeout,
        validator=validator,
        trainer_services=trainer_services,
        enforce_trainer_inactive=True,
        enforce_source_identity=True,
    )
    verify_trainer_services_inactive(context)
    return context


def stages(context: Context) -> tuple[Stage, ...]:
    staging_dir = context.output_dir / ".staging"
    result: list[Stage] = []
    for index in range(1, 4):
        name = f"benchmark_{index}"
        result.append(Stage(
            name,
            "cuda:0",
            context.benchmark_args,
            context.benchmark_timeout_seconds,
            context.output_dir / f"{name}.json",
            staging_dir / f"{name}.json",
            context.output_dir / f"{name}.launch.json",
            context.output_dir / f"{name}.execution.json",
        ))
    name = "cpu_diagnostic"
    result.append(Stage(
        name,
        "cpu",
        context.cpu_args,
        context.cpu_timeout_seconds,
        context.output_dir / f"{name}.json",
        staging_dir / f"{name}.json",
        context.output_dir / f"{name}.launch.json",
        context.output_dir / f"{name}.execution.json",
    ))
    return tuple(result)


def args_to_cli(arguments: Mapping[str, Any]) -> list[str]:
    command: list[str] = []
    for name, value in arguments.items():
        flag = f"--{name}"
        if value is None or value is False:
            continue
        command.append(flag)
        if value is True:
            continue
        if isinstance(value, list):
            command.extend(str(item) for item in value)
        else:
            command.append(str(value))
    return command


def command_for(context: Context, stage: Stage) -> list[str]:
    return [
        sys.executable,
        "-P",
        "-B",
        str(context.benchmark_script),
        *args_to_cli(stage.args),
        "--checkpoint",
        str(context.deployment_path),
        "--json-output",
        str(stage.staging),
        "--report-only",
    ]


def _nested(value: Mapping[str, Any], path: Sequence[str], label: str) -> Any:
    current: Any = value
    for component in path:
        require(isinstance(current, Mapping) and component in current, f"{label}.{'.'.join(path)} is missing")
        current = current[component]
    return current


def require_exact_keys(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} is not an object")
    require(set(value) == keys, f"{label} fields changed: {sorted(set(value) ^ keys)}")
    return value


def require_exact_value(actual: Any, expected: Any, label: str) -> None:
    require(type(actual) is type(expected), f"{label} type changed")
    if isinstance(expected, dict):
        require(set(actual) == set(expected), f"{label} fields changed")
        for key, expected_value in expected.items():
            require_exact_value(actual[key], expected_value, f"{label}.{key}")
        return
    if isinstance(expected, list):
        require(len(actual) == len(expected), f"{label} length changed")
        for index, (actual_value, expected_value) in enumerate(zip(actual, expected, strict=True)):
            require_exact_value(actual_value, expected_value, f"{label}[{index}]")
        return
    require(actual == expected, f"{label} changed")


CPU_NON_TIMING_GATES = (
    ("model", "deployment_config_pass"),
    ("device_identity", "identity_pass"),
    ("device_identity", "native_pre_editable_import", "pass"),
    ("determinism_pass",),
    ("runtime_profile_pass",),
    ("execution_backend", "pass"),
    ("precision", "pass"),
    ("model", "fp32_state_pass"),
    ("model", "eager_backend_pass"),
    ("threading_pass",),
    ("correctness", "pass"),
    ("correctness", "silence", "pass"),
    ("correctness", "full_scale", "pass"),
    ("prefix_causality", "pass"),
    ("impulse_alignment", "pass"),
    ("latency_pass",),
    ("timing", "shape_pass"),
    ("timing", "finite_pass"),
    ("timing", "bounded_peak_pass"),
    ("memory", "pass"),
)

CPU_CONSTRUCTOR_CONFIG_KEYS = [
    "decoder_hann_baked",
    "dim",
    "n_fft",
    "norm_before_mask_estimate",
    "num_basis",
    "num_sources",
    "overlap_len",
    "residual_source_softmax",
    "rnn_klass",
    "sample_rate",
    "segment_len",
    "small",
    "spec_branch_use_phase",
    "stereo",
    "torch_compile",
    "use_branch_rnns",
    "use_gru",
]
CONSTRUCTOR_EVIDENCE_KEYS = {
    "source", "raw_type", "raw_bytes", "decode_error", "decoded_type",
    "keys", "torch_compile_present", "torch_compile_value",
    "torch_compile_value_type", "torch_compile_exact_false", "pass",
}
EAGER_PHASE_KEYS = {
    "phase", "backend", "saved_constructor_torch_compile_exact_false",
    "model_config_blob_matches_checkpoint", "constructor_config",
    "module_count", "compile_indicator_violations", "known_limitation", "pass",
}
PRECISION_PHASE_KEYS = {
    "phase", "expected_dtype", "expected_device", "parameter_tensor_count",
    "parameter_element_count", "buffer_tensor_count", "buffer_element_count",
    "floating_parameter_tensor_count", "floating_buffer_tensor_count",
    "floating_tensor_count", "floating_element_count",
    "floating_dtype_tensor_counts", "floating_device_tensor_counts",
    "dtype_violations", "device_violations", "pass",
}
MEMORY_KEYS = {
    "rss_source", "rss_start_mb", "rss_end_mb", "rss_end_growth_mb",
    "rss_sampled_peak_mb", "rss_sampled_peak_growth_mb", "rss_samples",
    "rss_trend", "ru_maxrss_start_mb", "ru_maxrss_end_mb",
    "max_rss_growth_mb", "rss_absolute_growth_pass", "rss_pass",
    "cuda_start", "cuda_end", "cuda_allocated_growth_mb",
    "cuda_reserved_growth_mb", "max_cuda_allocated_growth_mb",
    "max_cuda_reserved_growth_mb", "cuda_allocated_growth_pass",
    "cuda_reserved_growth_pass", "cuda_pass", "pass",
}
RSS_SAMPLE_KEYS = {"callback", "rss_mb"}
RSS_TREND_KEYS = {
    "sample_count", "quartile_sample_count", "first_quartile_median_mb",
    "last_quartile_median_mb", "quartile_median_growth_signed_mb",
    "quartile_median_growth_mb",
    "least_squares_slope_mb_per_callback_signed",
    "least_squares_slope_mb_per_callback",
    "least_squares_growth_mb_per_10k", "observed_callback_span",
    "least_squares_growth_over_observed_span_mb", "method",
    "sample_interval_callbacks", "extrapolate_callbacks",
    "minimum_sample_count", "sample_count_pass",
    "max_quartile_median_growth_mb", "quartile_median_growth_pass",
    "max_least_squares_growth_mb_per_10k", "least_squares_growth_pass",
    "pass",
}
INTEROP_CONTROL_EVIDENCE = (
    "effective count verified; PyTorch set_num_interop_threads is one-shot "
    "and was frozen before editable import"
)
EAGER_KNOWN_LIMITATION = (
    "standard Dynamo/OptimizedModule/JIT indicators are rejected, but PyTorch "
    "exposes no proof that arbitrary model code never invokes a compiler "
    "transiently or disguises a custom compiled backend"
)


def validate_result_structure(context: Context, stage: Stage, result: Mapping[str, Any]) -> None:
    require(set(result) == RESULT_TOP_LEVEL_KEYS, f"{stage.name} top-level result fields changed")
    require(type(result.get("schema_version")) is int and result["schema_version"] == 1, f"{stage.name} schema changed")
    require(result.get("checkpoint") == str(context.deployment_path), f"{stage.name} checkpoint path changed")
    require(result.get("local_package_path") == str((context.source_repo / "hs_tasnet/__init__.py").resolve()), f"{stage.name} loaded an unexpected package")
    require(result.get("requested_device") == stage.device and result.get("device") == stage.device, f"{stage.name} device changed")
    callback_count = result.get("timing", {}).get("callback_count")
    require(type(callback_count) is int and callback_count == 10_000, f"{stage.name} callback count changed")
    parameter_count = result.get("model", {}).get("parameter_count")
    require(type(parameter_count) is int and parameter_count == EXPECTED_PARAMETER_COUNT, f"{stage.name} parameter count changed")
    require(isinstance(result.get("constraint_pass"), bool), f"{stage.name} constraint_pass is invalid")


def _expected_cpu_control_state() -> dict[str, Any]:
    return {
        "affinity": {"cpus": [0, 2, 4, 6], "supported": True},
        "cpu_threads": 4,
        "cublas_workspace_config": None,
        "cuda_device_order": None,
        "cuda_device_order_present": False,
        "cuda_matmul_allow_tf32": False,
        "cuda_visible_devices": "",
        "cuda_visible_devices_present": True,
        "cudnn_allow_tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms_enabled": True,
        "deterministic_algorithms_warn_only": False,
        "float32_matmul_precision": "highest",
        "interop_threads": 1,
    }


def validate_constructor_evidence(value: Any, *, source: str, label: str) -> None:
    constructor = require_exact_keys(value, CONSTRUCTOR_EVIDENCE_KEYS, label)
    require_exact_value(constructor, {
        "source": source,
        "raw_type": "bytes",
        "raw_bytes": True,
        "decode_error": None,
        "decoded_type": "dict",
        "keys": CPU_CONSTRUCTOR_CONFIG_KEYS,
        "torch_compile_present": True,
        "torch_compile_value": False,
        "torch_compile_value_type": "bool",
        "torch_compile_exact_false": True,
        "pass": True,
    }, label)


def _finite_float(value: Any, label: str) -> float:
    require(type(value) is float and math.isfinite(value), f"{label} is not a finite float")
    return value


def validate_cpu_rss_memory(
    context: Context,
    result: Mapping[str, Any],
    memory: Mapping[str, Any],
) -> None:
    require(memory.get("rss_source") == "/proc/self/statm", "CPU RSS source changed")
    interval = context.cpu_args["memory-sample-interval"]
    callbacks = context.cpu_args["callbacks"]
    minimum_samples = context.cpu_args["memory-trend-min-samples"]
    require(type(interval) is int and interval == 1_000, "CPU RSS sample interval changed")
    require(type(callbacks) is int and callbacks == 10_000, "CPU RSS callback count changed")
    require(type(minimum_samples) is int and minimum_samples == 8, "CPU RSS minimum samples changed")
    samples = memory.get("rss_samples")
    require(isinstance(samples, list), "CPU RSS samples are malformed")
    expected_callbacks = list(range(0, callbacks + 1, interval))
    require(len(samples) == len(expected_callbacks), "CPU RSS sample count changed")
    rss_values: list[float] = []
    for index, (sample, expected_callback) in enumerate(zip(samples, expected_callbacks, strict=True)):
        sample = require_exact_keys(sample, RSS_SAMPLE_KEYS, f"CPU RSS sample {index}")
        require(
            type(sample.get("callback")) is int and sample["callback"] == expected_callback,
            f"CPU RSS sample callback changed: {index}",
        )
        rss = _finite_float(sample.get("rss_mb"), f"CPU RSS sample {index}.rss_mb")
        require(rss >= 0.0, f"CPU RSS sample is negative: {index}")
        rss_values.append(rss)

    rss_start = _finite_float(memory.get("rss_start_mb"), "CPU RSS start")
    rss_end = _finite_float(memory.get("rss_end_mb"), "CPU RSS end")
    rss_end_growth = _finite_float(memory.get("rss_end_growth_mb"), "CPU RSS end growth")
    rss_peak = _finite_float(memory.get("rss_sampled_peak_mb"), "CPU RSS sampled peak")
    rss_peak_growth = _finite_float(memory.get("rss_sampled_peak_growth_mb"), "CPU RSS sampled peak growth")
    require(rss_start == rss_values[0] and rss_end == rss_values[-1], "CPU RSS endpoints do not bind samples")
    require(rss_end_growth == rss_end - rss_start, "CPU RSS end growth is inconsistent")
    require(rss_peak == max(rss_values) and rss_peak_growth == rss_peak - rss_start, "CPU RSS peak evidence is inconsistent")
    ru_start = _finite_float(memory.get("ru_maxrss_start_mb"), "CPU ru_maxrss start")
    ru_end = _finite_float(memory.get("ru_maxrss_end_mb"), "CPU ru_maxrss end")
    require(ru_start >= 0.0 and ru_end >= ru_start, "CPU ru_maxrss evidence is inconsistent")

    max_rss_growth = context.cpu_args["max-rss-growth-mb"]
    max_quartile_growth = context.cpu_args["max-rss-quartile-growth-mb"]
    max_slope_growth = context.cpu_args["max-rss-slope-growth-mb-per-10k"]
    for value, expected, label in (
        (memory.get("max_rss_growth_mb"), max_rss_growth, "max RSS growth"),
    ):
        require_exact_value(value, expected, f"CPU {label}")

    sample_count = len(samples)
    quartile_count = max(1, math.ceil(sample_count / 4.0))
    first_median = float(statistics.median(rss_values[:quartile_count]))
    last_median = float(statistics.median(rss_values[-quartile_count:]))
    quartile_signed = last_median - first_median
    quartile_growth = max(0.0, quartile_signed)
    callback_values = expected_callbacks
    mean_x = math.fsum(callback_values) / sample_count
    mean_y = math.fsum(rss_values) / sample_count
    denominator = math.fsum((value - mean_x) ** 2 for value in callback_values)
    require(denominator > 0.0, "CPU RSS samples have no callback span")
    signed_slope = math.fsum(
        (callback - mean_x) * (rss - mean_y)
        for callback, rss in zip(callback_values, rss_values, strict=True)
    ) / denominator
    slope = max(0.0, signed_slope)
    extrapolate = 10_000
    span = callback_values[-1] - callback_values[0]
    sample_count_pass = sample_count >= minimum_samples
    quartile_pass = quartile_growth <= max_quartile_growth
    slope_pass = slope * extrapolate <= max_slope_growth
    trend_pass = sample_count_pass and quartile_pass and slope_pass
    expected_trend = {
        "sample_count": sample_count,
        "quartile_sample_count": quartile_count,
        "first_quartile_median_mb": first_median,
        "last_quartile_median_mb": last_median,
        "quartile_median_growth_signed_mb": quartile_signed,
        "quartile_median_growth_mb": quartile_growth,
        "least_squares_slope_mb_per_callback_signed": signed_slope,
        "least_squares_slope_mb_per_callback": slope,
        "least_squares_growth_mb_per_10k": slope * extrapolate,
        "observed_callback_span": span,
        "least_squares_growth_over_observed_span_mb": slope * span,
        "method": (
            "positive last-minus-first quartile RSS median and nonnegative "
            "least-squares slope projected to 10000 callbacks"
        ),
        "sample_interval_callbacks": interval,
        "extrapolate_callbacks": extrapolate,
        "minimum_sample_count": minimum_samples,
        "sample_count_pass": sample_count_pass,
        "max_quartile_median_growth_mb": max_quartile_growth,
        "quartile_median_growth_pass": quartile_pass,
        "max_least_squares_growth_mb_per_10k": max_slope_growth,
        "least_squares_growth_pass": slope_pass,
        "pass": trend_pass,
    }
    trend = require_exact_keys(memory.get("rss_trend"), RSS_TREND_KEYS, "CPU RSS trend")
    require_exact_value(trend, expected_trend, "CPU RSS trend")
    absolute_pass = rss_end_growth <= max_rss_growth
    require(memory.get("rss_absolute_growth_pass") is absolute_pass, "CPU RSS absolute-growth pass changed")
    require(memory.get("rss_pass") is (absolute_pass and trend_pass), "CPU RSS pass changed")
    require_exact_value(result.get("memory_growth_mb"), rss_end_growth, "CPU top-level RSS growth")


def validate_cpu_metadata(context: Context, result: Mapping[str, Any]) -> None:
    identity_keys = {
        "requested", "requested_torch_device", "resolved", "type",
        "requested_index", "resolved_visible_ordinal", "visible_device_count",
        "expected_name", "expected_uuid_raw", "expected_uuid_canonical",
        "resolved_name", "resolved_uuid_raw", "resolved_uuid_canonical",
        "identity_pass", "cuda_environment", "native_pre_editable_import",
    }
    identity = require_exact_keys(result.get("device_identity"), identity_keys, "CPU device_identity")
    expected_identity = {
        "requested": "cpu",
        "requested_torch_device": "cpu",
        "resolved": "cpu",
        "type": "cpu",
        "requested_index": None,
        "resolved_visible_ordinal": None,
        "visible_device_count": None,
        "expected_name": None,
        "expected_uuid_raw": None,
        "expected_uuid_canonical": None,
        "resolved_name": None,
        "resolved_uuid_raw": None,
        "resolved_uuid_canonical": None,
        "identity_pass": True,
        "native_pre_editable_import": {"applicable": False, "pass": True},
    }
    expected_cuda_environment = {
        "cuda_visible_devices_present": True,
        "cuda_visible_devices": "",
        "cuda_visible_device_tokens": [],
        "cuda_device_order_present": False,
        "cuda_device_order": None,
        "resolved_visible_ordinal": None,
        "selected_cuda_visible_devices_token": None,
        "mapping_source": "CUDA_VISIBLE_DEVICES",
    }
    require_exact_value(
        identity,
        {**expected_identity, "cuda_environment": expected_cuda_environment},
        "CPU device identity",
    )

    runtime_keys = {
        "platform", "python", "torch", "numpy", "cpu_model", "affinity",
        "cpu_threads_requested", "cpu_threads_effective",
        "interop_threads_requested", "interop_threads_effective",
        "interop_thread_error", "determinism", "initial_control_state",
        "profile_boundaries",
    }
    runtime = require_exact_keys(result.get("runtime"), runtime_keys, "CPU runtime")
    expected_versions = {
        "platform": context.validator.EXPECTED_PLATFORM,
        "python": context.validator.EXPECTED_PYTHON_VERSION,
        "torch": context.validator.EXPECTED_TORCH_VERSION,
        "numpy": context.validator.EXPECTED_NUMPY_VERSION,
        "cpu_model": context.validator.EXPECTED_TRAIN_CPU_MODEL,
        "cpu_threads_requested": 4,
        "cpu_threads_effective": 4,
        "interop_threads_requested": 1,
        "interop_threads_effective": 1,
        "interop_thread_error": None,
    }
    for key, expected in expected_versions.items():
        require_exact_value(runtime.get(key), expected, f"CPU frozen runtime.{key}")
    expected_affinity = {
        "requested": "0,2,4,6",
        "mode": "explicit",
        "supported": True,
        "success": True,
        "cpus": [0, 2, 4, 6],
        "error": None,
    }
    require_exact_value(runtime.get("affinity"), expected_affinity, "CPU affinity")
    determinism_keys = {
        "cuda_requested", "deterministic_algorithms_enabled",
        "deterministic_algorithms_warn_only", "cudnn_deterministic",
        "cudnn_benchmark", "cuda_matmul_allow_tf32", "cudnn_allow_tf32",
        "float32_matmul_precision", "torch_cuda_version", "cudnn_version",
        "cublas_workspace", "pass",
    }
    determinism = require_exact_keys(runtime.get("determinism"), determinism_keys, "CPU determinism")
    expected_determinism = {
        "cuda_requested": False,
        "deterministic_algorithms_enabled": True,
        "deterministic_algorithms_warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "torch_cuda_version": context.validator.EXPECTED_TORCH_CUDA_VERSION,
        "cudnn_version": context.validator.EXPECTED_CUDNN_VERSION,
        "pass": True,
    }
    expected_cublas = {
        "required": False,
        "expected": None,
        "value_before": None,
        "value_after": None,
        "set_by_benchmark": False,
        "torch_already_imported_before_configuration": False,
        "pass": True,
    }
    require_exact_value(
        determinism,
        {**expected_determinism, "cublas_workspace": expected_cublas},
        "CPU determinism",
    )
    expected_control = _expected_cpu_control_state()
    require_exact_value(runtime.get("initial_control_state"), expected_control, "CPU initial runtime controls")
    boundaries = require_exact_keys(runtime.get("profile_boundaries"), {"pre_probes", "pre_timing", "post_timing"}, "CPU runtime boundaries")
    match_keys = {
        "cublas_workspace", "cuda_visible_devices", "cuda_device_order",
        "deterministic_algorithms", "cudnn_deterministic", "cudnn_benchmark",
        "cuda_matmul_tf32", "cudnn_tf32", "float32_matmul_precision",
        "cpu_threads", "interop_threads", "cpu_affinity",
    }
    boundary_keys = {
        "phase", "before_matches", "after_matches", "mutation_detected",
        "reassertion_errors", "observed_before_reassert", "observed_after_reassert",
        "native_device_identity", "interop_thread_control", "pass",
    }
    for phase in ("pre_probes", "pre_timing", "post_timing"):
        boundary = require_exact_keys(boundaries[phase], boundary_keys, f"CPU boundary {phase}")
        require(boundary.get("phase") == phase and boundary.get("pass") is True, f"CPU boundary failed: {phase}")
        require(boundary.get("mutation_detected") is False and boundary.get("reassertion_errors") == [], f"CPU boundary mutated: {phase}")
        for key in ("before_matches", "after_matches"):
            matches = require_exact_keys(boundary.get(key), match_keys, f"CPU boundary {phase}.{key}")
            require(all(value is True for value in matches.values()), f"CPU boundary matches failed: {phase}.{key}")
        require_exact_value(boundary.get("observed_before_reassert"), expected_control, f"CPU pre-reassert controls {phase}")
        require_exact_value(boundary.get("observed_after_reassert"), expected_control, f"CPU post-reassert controls {phase}")
        require_exact_value(boundary.get("native_device_identity"), {"applicable": False, "pass": True}, f"CPU native device record {phase}")
        require(boundary.get("interop_thread_control") == INTEROP_CONTROL_EVIDENCE, f"CPU interop control evidence changed: {phase}")

    backend = require_exact_keys(result.get("execution_backend"), {"name", "checks", "pass"}, "CPU execution backend")
    require(backend.get("name") == "pytorch_eager" and backend.get("pass") is True, "CPU backend is not frozen eager")
    backend_checks = require_exact_keys(backend.get("checks"), {"checkpoint_constructor_config", "pre_probes", "pre_timing", "post_timing"}, "CPU backend checks")
    validate_constructor_evidence(
        backend_checks["checkpoint_constructor_config"],
        source="checkpoint package config",
        label="CPU checkpoint constructor",
    )
    for phase in ("pre_probes", "pre_timing", "post_timing"):
        check = require_exact_keys(backend_checks[phase], EAGER_PHASE_KEYS, f"CPU eager check {phase}")
        require(check.get("phase") == phase and check.get("backend") == "pytorch_eager" and check.get("pass") is True, f"CPU eager check failed: {phase}")
        require(
            check.get("compile_indicator_violations") == []
            and check.get("model_config_blob_matches_checkpoint") is True
            and check.get("saved_constructor_torch_compile_exact_false") is True
            and type(check.get("module_count")) is int
            and check.get("module_count") == 19
            and check.get("known_limitation") == EAGER_KNOWN_LIMITATION,
            f"CPU eager evidence changed: {phase}",
        )
        validate_constructor_evidence(
            check.get("constructor_config"),
            source=f"model._config at {phase}",
            label=f"CPU phase constructor {phase}",
        )

    precision = require_exact_keys(result.get("precision"), {"name", "checks", "pass"}, "CPU precision")
    require(precision.get("name") == "fp32" and precision.get("pass") is True, "CPU precision is not FP32")
    precision_checks = require_exact_keys(precision.get("checks"), {"pre_probes", "pre_timing", "post_timing"}, "CPU precision checks")
    for phase in ("pre_probes", "pre_timing", "post_timing"):
        check = require_exact_keys(precision_checks[phase], PRECISION_PHASE_KEYS, f"CPU FP32 check {phase}")
        expected_precision = {
            "phase": phase,
            "expected_dtype": "torch.float32",
            "expected_device": "cpu",
            "parameter_tensor_count": 22,
            "parameter_element_count": EXPECTED_PARAMETER_COUNT,
            "buffer_tensor_count": 4,
            "buffer_element_count": 3_076,
            "floating_parameter_tensor_count": 22,
            "floating_buffer_tensor_count": 4,
            "floating_tensor_count": 26,
            "floating_element_count": 30_130_286,
            "floating_dtype_tensor_counts": {"torch.float32": 26},
            "floating_device_tensor_counts": {"cpu": 26},
            "dtype_violations": [],
            "device_violations": [],
            "pass": True,
        }
        require_exact_value(check, expected_precision, f"CPU FP32 state {phase}")

    expected_model = {
        "stereo": True,
        "audio_channels": 2,
        "num_sources": 4,
        "sample_rate": 44_100,
        "segment_len": 1024,
        "overlap_len": 512,
        "hop_length": 512,
        "n_fft": 1024,
        "batch_size": 1,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "num_params_m": EXPECTED_PARAMETER_COUNT / 1e6,
        "deployment_config_pass": True,
        "fp32_state_pass": True,
        "eager_backend_pass": True,
    }
    require_exact_value(result.get("model"), expected_model, "CPU deployment model geometry/profile")
    require(result.get("dtype") == "torch.float32", "CPU top-level dtype changed")
    require(result.get("runtime_profile_pass") is True and result.get("determinism_pass") is True and result.get("threading_pass") is True, "CPU runtime hard profile failed")
    memory = require_exact_keys(result.get("memory"), MEMORY_KEYS, "CPU memory evidence")
    validate_cpu_rss_memory(context, result, memory)
    require(memory.get("cuda_start") is None and memory.get("cuda_end") is None, "CPU diagnostic unexpectedly used CUDA memory")
    for field in ("cuda_allocated_growth_mb", "cuda_reserved_growth_mb"):
        require(type(memory.get(field)) is float and memory[field] == 0.0, f"CPU CUDA growth changed: {field}")
    require_exact_value(
        memory.get("max_cuda_allocated_growth_mb"),
        context.cpu_args["max-cuda-allocated-growth-mb"],
        "CPU max CUDA allocated growth",
    )
    require_exact_value(
        memory.get("max_cuda_reserved_growth_mb"),
        context.cpu_args["max-cuda-reserved-growth-mb"],
        "CPU max CUDA reserved growth",
    )
    require(
        memory.get("cuda_allocated_growth_pass") is True
        and memory.get("cuda_reserved_growth_pass") is True
        and memory.get("cuda_pass") is True,
        "CPU CUDA memory pass evidence changed",
    )
    require(type(memory.get("rss_pass")) is bool and type(memory.get("pass")) is bool, "CPU RSS/memory pass flags are invalid")
    require(memory.get("pass") is memory.get("rss_pass"), "CPU memory pass is inconsistent with hidden CUDA")
    require(type(result.get("peak_vram_mb")) is float and result["peak_vram_mb"] == 0.0, "CPU diagnostic reported CUDA VRAM")


def acceptance_pass(context: Context, stage: Stage, result: Mapping[str, Any]) -> bool:
    validate_result_structure(context, stage, result)
    if stage.device == "cuda:0":
        try:
            context.validator._validate_primary_benchmark_metadata(result)
        except Exception as error:
            raise Contradiction(f"{stage.name} failed frozen validation: {error}") from error
        gates = tuple(context.validator.AGGREGATED_HARD_BOOLEAN_PATHS)
        values = [_nested(result, path, stage.name) for path in gates]
        require(all(isinstance(value, bool) for value in values), f"{stage.name} has a non-boolean hard gate")
        return result.get("constraint_pass") is True and all(values)

    try:
        validate_cpu_metadata(context, result)
        context.validator._validate_timing_evidence(result.get("timing", {}))
    except Exception as error:
        raise Contradiction(f"CPU diagnostic failed frozen timing-evidence validation: {error}") from error
    values = [_nested(result, path, stage.name) for path in CPU_NON_TIMING_GATES]
    require(all(isinstance(value, bool) for value in values), "CPU diagnostic has a non-boolean hard gate")
    return all(values)


def verify_deployment(context: Context) -> None:
    require(context.deployment_path.is_file() and not context.deployment_path.is_symlink(), "deployment disappeared/became a symlink")
    require(sha256_file(context.deployment_path) == context.deployment_sha256, "deployment changed during final benchmarks")
    expected = f"{context.deployment_sha256}  {context.deployment_path.name}\n".encode("ascii")
    sidecar = sidecar_path(context.deployment_path)
    require(sidecar.is_file() and not sidecar.is_symlink(), "deployment sidecar disappeared/became a symlink")
    require(sidecar.read_bytes() == expected, "deployment sidecar changed during final benchmarks")


def verify_runtime_bindings(context: Context) -> None:
    verify_deployment(context)
    for path, expected, label in (
        (context.runner_path, context.runner_sha256, "runner"),
        (context.watchdog_path, context.watchdog_sha256, "watchdog"),
        (context.full_config_path, context.full_config_sha256, "full config"),
        (context.run_dir / "final_report.json", context.final_report_sha256, "final report"),
        (context.run_dir / "final_audit_receipt.json", context.audit_receipt_sha256, "final audit receipt"),
        (context.run_config_path, context.run_config_sha256, "run config"),
        (context.benchmark_script, context.benchmark_sha256, "benchmark script"),
        (context.orchestrator_path, context.orchestrator_sha256, "frozen orchestrator"),
    ):
        require(path.is_file() and not path.is_symlink(), f"bound {label} is missing/symlink")
        require(sha256_file(path) == expected, f"bound {label} changed during final benchmarks")
    for name, evidence in context.run_evidence_files.items():
        require(set(evidence) == {"path", "sha256"}, f"bound run evidence schema changed: {name}")
        path = Path(evidence["path"])
        require(path.is_file() and not path.is_symlink(), f"bound run evidence is missing/symlink: {name}")
        require(sha256_file(path) == evidence["sha256"], f"bound run evidence changed during final benchmarks: {name}")
    if context.enforce_source_identity:
        require(_git(context.source_repo, "rev-parse", "HEAD") == context.source_commit, "source HEAD changed during final benchmarks")
        require(_git(context.source_repo, "rev-parse", "HEAD^{tree}") == context.source_tree, "source tree changed during final benchmarks")
        verify_frozen_files(context.source_repo, context.frozen_files)
        require(not (context.source_repo / "research/__init__.py").exists(), "research/__init__.py appeared during final benchmarks")
        verify_source_worktree(context.source_repo)
    verify_trainer_services_inactive(context)


def child_environment(context: Context, stage: Stage) -> dict[str, str]:
    environment = dict(os.environ)
    for key in list(environment):
        if key.startswith("GIT_") or key in {"PYTHONHOME", "PYTHONSTARTUP"}:
            environment.pop(key, None)
    environment["PYTHONPATH"] = str(context.source_repo)
    environment["PYTHONSAFEPATH"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPYCACHEPREFIX"] = "/nonexistent/hs-tasnet-final-benchmark-pycache"
    environment.pop("CUDA_DEVICE_ORDER", None)
    environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
    if stage.device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    else:
        environment.pop("CUDA_VISIBLE_DEVICES", None)
    return environment


def exact_child_pids(command: Sequence[str], proc_root: Path = Path("/proc")) -> list[int]:
    expected = [str(value) for value in command]
    matches: list[int] = []
    require(proc_root.is_dir() and not proc_root.is_symlink(), f"cannot inspect process table: {proc_root}")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            fields = entry.joinpath("cmdline").read_bytes().split(b"\0")
            observed = [field.decode("utf-8") for field in fields if field]
        except (OSError, UnicodeError):
            continue
        if len(observed) != len(expected):
            continue
        first_matches = observed[0] == expected[0]
        if not first_matches:
            try:
                first_matches = Path(observed[0]).resolve() == Path(expected[0]).resolve()
            except OSError:
                first_matches = False
        if first_matches and observed[1:] == expected[1:]:
            matches.append(int(entry.name))
    return sorted(matches)


LAUNCH_KEYS = {
    "schema_version", "kind", "stage", "stage_index", "launch_id",
    "launched_at_utc", "launcher_pid", "session_identity_sha256",
    "session_file_sha256", "command", "command_sha256", "staging_path",
    "launch_payload_sha256",
}


def build_launch(
    context: Context,
    stage: Stage,
    stage_index: int,
    session: Mapping[str, Any],
    session_sha256: str,
) -> dict[str, Any]:
    command = command_for(context, stage)
    launch: dict[str, Any] = {
        "schema_version": 1,
        "kind": "hs_tasnet_c91_final_benchmark_launch",
        "stage": stage.name,
        "stage_index": stage_index,
        "launch_id": uuid.uuid4().hex,
        "launched_at_utc": utc_now(),
        "launcher_pid": os.getpid(),
        "session_identity_sha256": session["session_identity_sha256"],
        "session_file_sha256": session_sha256,
        "command": command,
        "command_sha256": canonical_sha256(command),
        "staging_path": str(stage.staging),
    }
    launch["launch_payload_sha256"] = canonical_sha256(launch)
    return launch


def validate_launch(
    context: Context,
    stage: Stage,
    stage_index: int,
    session: Mapping[str, Any],
    session_sha256: str,
    launch: Mapping[str, Any],
) -> None:
    require(set(launch) == LAUNCH_KEYS, f"{stage.name} launch fields changed")
    require(type(launch.get("schema_version")) is int and launch["schema_version"] == 1 and launch.get("kind") == "hs_tasnet_c91_final_benchmark_launch", f"{stage.name} launch schema/kind changed")
    require(launch.get("stage") == stage.name and type(launch.get("stage_index")) is int and launch["stage_index"] == stage_index, f"{stage.name} launch ordering changed")
    launch_id = launch.get("launch_id")
    require(isinstance(launch_id, str) and re.fullmatch(r"[0-9a-f]{32}", launch_id) is not None, f"{stage.name} launch ID is invalid")
    launched = validate_utc_timestamp(launch.get("launched_at_utc"), f"{stage.name}.launched_at_utc")
    require(datetime.fromisoformat(launched) >= datetime.fromisoformat(str(session["created_at_utc"])), f"{stage.name} launch predates session")
    require(isinstance(launch.get("launcher_pid"), int) and not isinstance(launch.get("launcher_pid"), bool) and launch["launcher_pid"] > 0, f"{stage.name} launcher PID is invalid")
    require(launch.get("session_identity_sha256") == session["session_identity_sha256"], f"{stage.name} launch session identity changed")
    require(launch.get("session_file_sha256") == session_sha256, f"{stage.name} launch session file changed")
    command = command_for(context, stage)
    require(launch.get("command") == command and launch.get("command_sha256") == canonical_sha256(command), f"{stage.name} launch command changed")
    require(launch.get("staging_path") == str(stage.staging), f"{stage.name} launch staging path changed")
    payload = launch.get("launch_payload_sha256")
    require(isinstance(payload, str) and SHA256_RE.fullmatch(payload) is not None, f"{stage.name} launch payload hash is invalid")
    unhashed = dict(launch)
    unhashed.pop("launch_payload_sha256")
    require(canonical_sha256(unhashed) == payload, f"{stage.name} launch payload hash mismatch")


def ensure_launch(
    context: Context,
    stage: Stage,
    stage_index: int,
    session: Mapping[str, Any],
    session_sha256: str,
) -> tuple[dict[str, Any], str, bool]:
    validator = lambda value: validate_launch(
        context, stage, stage_index, session, session_sha256, value
    )
    existing = load_sealed_document(stage.launch, validator)
    if existing is not None:
        return existing[0], existing[1], False
    value, digest = publish_sealed_document(
        stage.launch,
        build_launch(context, stage, stage_index, session, session_sha256),
        validator,
    )
    return value, digest, True


def load_launch(
    context: Context,
    stage: Stage,
    stage_index: int,
    session: Mapping[str, Any],
    session_sha256: str,
) -> tuple[dict[str, Any], str] | None:
    validator = lambda value: validate_launch(
        context, stage, stage_index, session, session_sha256, value
    )
    return load_sealed_document(stage.launch, validator)


def classify_result_file(
    context: Context,
    stage: Stage,
    path: Path,
) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid result candidate: {path}")
    try:
        json.loads(path.read_bytes())
    except (UnicodeError, json.JSONDecodeError):
        return {
            "state": "partial_malformed",
            "value": None,
            "acceptance_pass": None,
            "validation_error": "syntactically_malformed_json",
        }
    try:
        value, _ = strict_json(path)
        accepted = acceptance_pass(context, stage, value)
    except Exception:
        return {
            "state": "contract_invalid",
            "value": None,
            "acceptance_pass": None,
            "validation_error": "scientific_contract_invalid",
        }
    return {
        "state": "valid",
        "value": value,
        "acceptance_pass": accepted,
        "validation_error": None,
    }


def persistent_result_classification(classification: Mapping[str, Any]) -> dict[str, Any]:
    if classification["state"] != "partial_malformed":
        return dict(classification)
    return {
        "state": "contract_invalid",
        "value": None,
        "acceptance_pass": None,
        "validation_error": "syntactically_malformed_json",
    }


def promote_result_candidate(candidate: Path, canonical: Path) -> str:
    require(candidate.is_file() and not candidate.is_symlink(), f"invalid result candidate: {candidate}")
    fsync_file(candidate)
    if candidate != canonical:
        source_directory = candidate.parent
        os.replace(candidate, canonical)
        fsync_directory(source_directory)
        if canonical.parent != source_directory:
            fsync_directory(canonical.parent)
    else:
        fsync_directory(canonical.parent)
    return seal_existing_json(canonical)


def reconcile_result_candidate(
    context: Context,
    stage: Stage,
    *,
    allow_partial_tmp_cleanup: bool,
) -> dict[str, Any]:
    staging_tmp = stage.staging.with_name(stage.staging.name + ".tmp")
    live_pids = exact_child_pids(command_for(context, stage), context.proc_root)
    if live_pids:
        raise Busy(f"exact benchmark child is still live for {stage.name}: {live_pids}")
    canonical_state = pair_state(stage.canonical)
    staging_state = pair_state(stage.staging)
    tmp_state = pair_state(staging_tmp)
    require(canonical_state != "sidecar_only", f"orphan canonical result sidecar: {stage.canonical}")
    require(staging_state in {"absent", "json_only"}, f"invalid staging result seal: {stage.staging}")
    require(tmp_state in {"absent", "json_only"}, f"invalid temporary result seal: {staging_tmp}")
    representations = [
        name
        for name, state in (
            ("canonical", canonical_state),
            ("staging", staging_state),
            ("temporary", tmp_state),
        )
        if state != "absent"
    ]
    require(len(representations) <= 1, f"ambiguous result representations for {stage.name}: {representations}")
    if not representations:
        return {
            "state": "absent",
            "value": None,
            "acceptance_pass": None,
            "validation_error": None,
            "path": None,
            "sha256": None,
            "retryable_partial": False,
        }

    representation = representations[0]
    candidate = {
        "canonical": stage.canonical,
        "staging": stage.staging,
        "temporary": staging_tmp,
    }[representation]
    classification = classify_result_file(context, stage, candidate)
    if representation == "temporary" and classification["state"] == "partial_malformed":
        require(allow_partial_tmp_cleanup, f"cannot clean unproven live temporary output: {candidate}")
        require(candidate.is_file() and not candidate.is_symlink(), f"invalid partial benchmark temporary: {candidate}")
        candidate.unlink()
        fsync_directory(candidate.parent)
        return {
            "state": "absent",
            "value": None,
            "acceptance_pass": None,
            "validation_error": None,
            "path": None,
            "sha256": None,
            "retryable_partial": True,
        }

    classification = persistent_result_classification(classification)

    if representation == "canonical" and canonical_state == "pair":
        digest = verify_sidecar(stage.canonical)
    else:
        digest = promote_result_candidate(candidate, stage.canonical)
    return {
        **classification,
        "path": str(stage.canonical),
        "sha256": digest,
        "retryable_partial": False,
    }


EXECUTION_KEYS = {
    "schema_version", "kind", "stage", "stage_index", "recorded_at_utc",
    "session_identity_sha256", "session_file_sha256", "launch_file_sha256",
    "process", "binding_check", "result", "execution_pass",
    "failure_reasons", "execution_payload_sha256",
}
PROCESS_KEYS = {"outcome", "returncode", "timed_out", "outcome_known"}
BINDING_CHECK_KEYS = {"precheck_pass", "postcheck_pass", "postcheck_error"}
EXECUTION_RESULT_KEYS = {
    "state", "path", "sha256", "scientific_validation_pass",
    "acceptance_pass", "validation_error",
}


def execution_failure_reasons(
    process: Mapping[str, Any],
    binding_check: Mapping[str, Any],
    result: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    outcome = process.get("outcome")
    if outcome == "unknown_parent_restart":
        reasons.append("process_outcome_unknown")
    elif outcome == "timeout":
        reasons.append("process_timeout")
    elif outcome == "launch_error":
        reasons.append("process_launch_error")
    elif outcome == "exited" and process.get("returncode") != 0:
        reasons.append("process_nonzero_exit")
    if binding_check.get("postcheck_pass") is not True:
        reasons.append("postcheck_binding_violation")
    if result.get("state") == "contract_invalid":
        reasons.append("scientific_contract_invalid")
    elif result.get("state") == "absent":
        reasons.append("scientific_result_absent")
    return reasons


def build_execution(
    context: Context,
    stage: Stage,
    stage_index: int,
    session: Mapping[str, Any],
    session_sha256: str,
    launch_sha256: str,
    *,
    process_outcome: str,
    returncode: int | None,
    postcheck_pass: bool,
    postcheck_error: str | None,
    result_classification: Mapping[str, Any],
) -> dict[str, Any]:
    process = {
        "outcome": process_outcome,
        "returncode": returncode,
        "timed_out": process_outcome == "timeout",
        "outcome_known": process_outcome != "unknown_parent_restart",
    }
    binding_check = {
        "precheck_pass": True,
        "postcheck_pass": postcheck_pass,
        "postcheck_error": postcheck_error,
    }
    result = {
        "state": result_classification["state"],
        "path": result_classification["path"],
        "sha256": result_classification["sha256"],
        "scientific_validation_pass": result_classification["state"] == "valid",
        "acceptance_pass": result_classification["acceptance_pass"],
        "validation_error": result_classification["validation_error"],
    }
    reasons = execution_failure_reasons(process, binding_check, result)
    execution_pass = bool(
        process_outcome == "exited"
        and returncode == 0
        and postcheck_pass
        and result["scientific_validation_pass"]
    )
    execution: dict[str, Any] = {
        "schema_version": 1,
        "kind": "hs_tasnet_c91_final_benchmark_execution",
        "stage": stage.name,
        "stage_index": stage_index,
        "recorded_at_utc": utc_now(),
        "session_identity_sha256": session["session_identity_sha256"],
        "session_file_sha256": session_sha256,
        "launch_file_sha256": launch_sha256,
        "process": process,
        "binding_check": binding_check,
        "result": result,
        "execution_pass": execution_pass,
        "failure_reasons": reasons,
    }
    execution["execution_payload_sha256"] = canonical_sha256(execution)
    return execution


def validate_execution(
    context: Context,
    stage: Stage,
    stage_index: int,
    session: Mapping[str, Any],
    session_sha256: str,
    launch: Mapping[str, Any],
    launch_sha256: str,
    execution: Mapping[str, Any],
) -> None:
    require(set(execution) == EXECUTION_KEYS, f"{stage.name} execution fields changed")
    require(type(execution.get("schema_version")) is int and execution["schema_version"] == 1 and execution.get("kind") == "hs_tasnet_c91_final_benchmark_execution", f"{stage.name} execution schema/kind changed")
    require(execution.get("stage") == stage.name and type(execution.get("stage_index")) is int and execution["stage_index"] == stage_index, f"{stage.name} execution ordering changed")
    recorded = validate_utc_timestamp(execution.get("recorded_at_utc"), f"{stage.name}.recorded_at_utc")
    require(datetime.fromisoformat(recorded) >= datetime.fromisoformat(str(launch["launched_at_utc"])), f"{stage.name} execution predates launch")
    require(execution.get("session_identity_sha256") == session["session_identity_sha256"], f"{stage.name} execution session identity changed")
    require(execution.get("session_file_sha256") == session_sha256, f"{stage.name} execution session file changed")
    require(execution.get("launch_file_sha256") == launch_sha256, f"{stage.name} execution launch file changed")
    process = require_exact_keys(execution.get("process"), PROCESS_KEYS, f"{stage.name} process outcome")
    outcome = process.get("outcome")
    require(outcome in {"exited", "timeout", "launch_error", "unknown_parent_restart"}, f"{stage.name} process outcome is invalid")
    returncode = process.get("returncode")
    if outcome == "exited":
        require(isinstance(returncode, int) and not isinstance(returncode, bool), f"{stage.name} return code is invalid")
    else:
        require(returncode is None, f"{stage.name} non-exit outcome has a return code")
    require(process.get("timed_out") is (outcome == "timeout"), f"{stage.name} timeout flag changed")
    require(process.get("outcome_known") is (outcome != "unknown_parent_restart"), f"{stage.name} outcome-known flag changed")
    binding_check = require_exact_keys(execution.get("binding_check"), BINDING_CHECK_KEYS, f"{stage.name} binding check")
    require(binding_check.get("precheck_pass") is True and isinstance(binding_check.get("postcheck_pass"), bool), f"{stage.name} binding check flags are invalid")
    post_error = binding_check.get("postcheck_error")
    require((post_error is None) is (binding_check.get("postcheck_pass") is True), f"{stage.name} postcheck error/pass disagree")
    require(post_error is None or (isinstance(post_error, str) and post_error), f"{stage.name} postcheck error is invalid")
    result = require_exact_keys(execution.get("result"), EXECUTION_RESULT_KEYS, f"{stage.name} execution result")
    state = result.get("state")
    require(state in {"valid", "contract_invalid", "absent"}, f"{stage.name} result state is invalid")
    canonical_state = pair_state(stage.canonical)
    if state == "absent":
        require(canonical_state == "absent", f"{stage.name} absent execution has a durable result")
        require(result.get("path") is None and result.get("sha256") is None, f"{stage.name} absent result binding changed")
        require(result.get("scientific_validation_pass") is False and result.get("acceptance_pass") is None, f"{stage.name} absent scientific flags changed")
        require(result.get("validation_error") is None, f"{stage.name} absent result has an error code")
    else:
        require(canonical_state == "pair", f"{stage.name} execution result pair is incomplete")
        digest = verify_sidecar(stage.canonical)
        require(result.get("path") == str(stage.canonical) and result.get("sha256") == digest, f"{stage.name} execution result binding changed")
        classification = persistent_result_classification(
            classify_result_file(context, stage, stage.canonical)
        )
        require(classification["state"] == state, f"{stage.name} scientific classification changed")
        require(result.get("scientific_validation_pass") is (state == "valid"), f"{stage.name} scientific pass flag changed")
        require(result.get("acceptance_pass") is classification["acceptance_pass"], f"{stage.name} acceptance classification changed")
        require(result.get("validation_error") == classification["validation_error"], f"{stage.name} validation error changed")
    expected_reasons = execution_failure_reasons(process, binding_check, result)
    require(execution.get("failure_reasons") == expected_reasons, f"{stage.name} execution failure reasons changed")
    expected_pass = bool(
        outcome == "exited"
        and returncode == 0
        and binding_check["postcheck_pass"]
        and result["scientific_validation_pass"]
    )
    require(execution.get("execution_pass") is expected_pass, f"{stage.name} execution pass changed")
    payload = execution.get("execution_payload_sha256")
    require(isinstance(payload, str) and SHA256_RE.fullmatch(payload) is not None, f"{stage.name} execution payload hash is invalid")
    unhashed = dict(execution)
    unhashed.pop("execution_payload_sha256")
    require(canonical_sha256(unhashed) == payload, f"{stage.name} execution payload hash mismatch")


def load_execution(
    context: Context,
    stage: Stage,
    stage_index: int,
    session: Mapping[str, Any],
    session_sha256: str,
    launch: Mapping[str, Any],
    launch_sha256: str,
) -> tuple[dict[str, Any], str] | None:
    validator = lambda value: validate_execution(
        context, stage, stage_index, session, session_sha256,
        launch, launch_sha256, value,
    )
    return load_sealed_document(stage.execution, validator)


def publish_execution(
    context: Context,
    stage: Stage,
    stage_index: int,
    session: Mapping[str, Any],
    session_sha256: str,
    launch: Mapping[str, Any],
    launch_sha256: str,
    execution: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    validator = lambda value: validate_execution(
        context, stage, stage_index, session, session_sha256,
        launch, launch_sha256, value,
    )
    return publish_sealed_document(stage.execution, execution, validator)


def bindings(context: Context) -> dict[str, Any]:
    return {
        "run": {
            "path": str(context.run_dir),
            "contract_identity_sha256": context.contract_identity_sha256,
            "final_report_sha256": context.final_report_sha256,
            "audit_receipt_sha256": context.audit_receipt_sha256,
            "audit_payload_sha256": context.audit_payload_sha256,
            "evidence_files": {name: dict(value) for name, value in context.run_evidence_files.items()},
            "trainer_services": {name: dict(value) for name, value in context.trainer_state_evidence.items()},
        },
        "deployment": {
            "path": str(context.deployment_path),
            "sha256": context.deployment_sha256,
            "model_state_sha256": context.deployment_model_state_sha256,
        },
        "source": {
            "repo": str(context.source_repo),
            "commit": context.source_commit,
            "tree": context.source_tree,
            "run_config_path": str(context.run_config_path),
            "run_config_sha256": context.run_config_sha256,
            "benchmark_script_path": str(context.benchmark_script),
            "benchmark_script_sha256": context.benchmark_sha256,
            "orchestrator_path": str(context.orchestrator_path),
            "orchestrator_sha256": context.orchestrator_sha256,
            "frozen_files_sha256": canonical_sha256(dict(context.frozen_files)),
        },
        "automation": {
            "runner_path": str(context.runner_path),
            "runner_sha256": context.runner_sha256,
            "watchdog_path": str(context.watchdog_path),
            "watchdog_sha256": context.watchdog_sha256,
            "full_config_path": str(context.full_config_path),
            "full_config_sha256": context.full_config_sha256,
        },
        "protocol": {
            "primary_repetitions": 3,
            "benchmark_args": dict(context.benchmark_args),
            "benchmark_timeout_seconds": context.benchmark_timeout_seconds,
            "cpu_diagnostic_args": dict(context.cpu_args),
            "cpu_diagnostic_timeout_seconds": context.cpu_timeout_seconds,
            "commands": {stage.name: command_for(context, stage) for stage in stages(context)},
        },
    }


SESSION_KEYS = {
    "schema_version",
    "kind",
    "created_at_utc",
    "session_identity_sha256",
    "bindings",
    "session_payload_sha256",
}


def session_path(context: Context) -> Path:
    return context.output_dir / "session.json"


def build_session(context: Context) -> dict[str, Any]:
    bound = bindings(context)
    session: dict[str, Any] = {
        "schema_version": 1,
        "kind": "hs_tasnet_c91_final_benchmark_session",
        "created_at_utc": utc_now(),
        "session_identity_sha256": canonical_sha256(bound),
        "bindings": bound,
    }
    session["session_payload_sha256"] = canonical_sha256(session)
    return session


def validate_session(context: Context, session: Mapping[str, Any]) -> None:
    require(set(session) == SESSION_KEYS, "final benchmark session fields changed")
    require(type(session.get("schema_version")) is int and session["schema_version"] == 1 and session.get("kind") == "hs_tasnet_c91_final_benchmark_session", "final benchmark session schema/kind changed")
    validate_utc_timestamp(session.get("created_at_utc"), "session.created_at_utc")
    require_exact_value(
        session.get("bindings"), bindings(context), "final benchmark session bindings"
    )
    identity = session.get("session_identity_sha256")
    require(isinstance(identity, str) and SHA256_RE.fullmatch(identity) is not None, "session identity hash is invalid")
    require(identity == canonical_sha256(session["bindings"]), "session identity hash mismatch")
    payload = session.get("session_payload_sha256")
    require(isinstance(payload, str) and SHA256_RE.fullmatch(payload) is not None, "session payload hash is invalid")
    unhashed = dict(session)
    unhashed.pop("session_payload_sha256")
    require(canonical_sha256(unhashed) == payload, "session payload hash mismatch")


def ensure_session(context: Context) -> tuple[dict[str, Any], str]:
    validator = lambda value: validate_session(context, value)
    existing = load_sealed_document(session_path(context), validator)
    if existing is not None:
        return existing
    return publish_sealed_document(session_path(context), build_session(context), validator)


def _path_or_sidecar_exists(path: Path) -> bool:
    return any(
        candidate.exists() or candidate.is_symlink()
        for candidate in (path, sidecar_path(path))
    )


def stage_nonlaunch_evidence_present(stage: Stage) -> bool:
    temporary = stage.staging.with_name(stage.staging.name + ".tmp")
    return any(
        _path_or_sidecar_exists(path)
        for path in (stage.execution, stage.canonical, stage.staging, temporary)
    )


def stage_evidence_present(stage: Stage) -> bool:
    return _path_or_sidecar_exists(stage.launch) or stage_nonlaunch_evidence_present(stage)


def require_session_precedes_stage_evidence(context: Context) -> None:
    state = pair_state(session_path(context))
    require(state != "sidecar_only", "orphan final benchmark session sidecar")
    if state != "absent":
        return
    live = {
        stage.name: pids
        for stage in stages(context)
        if (pids := exact_child_pids(command_for(context, stage), context.proc_root))
    }
    if live:
        raise Busy(f"exact benchmark child predates sealed session: {live}")
    evidence = [stage.name for stage in stages(context) if stage_evidence_present(stage)]
    if _path_or_sidecar_exists(context.receipt_path):
        evidence.append("receipt")
    require(not evidence, f"benchmark evidence predates sealed session: {evidence}")


def validate_stage_prefix(
    context: Context,
    session: Mapping[str, Any],
    session_sha256: str,
) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    prefix_closed = False
    previous_terminal_time = datetime.fromisoformat(str(session["created_at_utc"]))
    for stage_index, stage in enumerate(stages(context), start=1):
        live_pids = exact_child_pids(command_for(context, stage), context.proc_root)
        launch_evidence = load_launch(
            context, stage, stage_index, session, session_sha256
        )
        if launch_evidence is None:
            if live_pids:
                raise Busy(f"unbound exact benchmark child is live for {stage.name}: {live_pids}")
            require(
                not stage_nonlaunch_evidence_present(stage),
                f"{stage.name} evidence exists without a sealed launch",
            )
            prefix_closed = True
            continue
        require(not prefix_closed, f"out-of-order launch evidence: {stage.name}")
        launch, launch_sha256 = launch_evidence
        launch_time = datetime.fromisoformat(str(launch["launched_at_utc"]))
        require(
            launch_time >= previous_terminal_time,
            f"{stage.name} launch predates the preceding terminal record",
        )
        execution_evidence = load_execution(
            context,
            stage,
            stage_index,
            session,
            session_sha256,
            launch,
            launch_sha256,
        )
        if execution_evidence is None:
            prefix_closed = True
            continue
        if live_pids:
            raise Busy(f"exact benchmark child remains live after {stage.name} execution: {live_pids}")
        execution, execution_sha256 = execution_evidence
        require(
            pair_state(stage.staging) == "absent"
            and pair_state(stage.staging.with_name(stage.staging.name + ".tmp")) == "absent",
            f"{stage.name} has staging evidence after terminal execution",
        )
        execution_time = datetime.fromisoformat(str(execution["recorded_at_utc"]))
        require(
            execution_time >= previous_terminal_time,
            f"{stage.name} execution predates the preceding terminal record",
        )
        result_value = None
        if execution["result"]["state"] != "absent":
            result_value = persistent_result_classification(
                classify_result_file(context, stage, stage.canonical)
            )["value"]
        completed[stage.name] = {
            "result": result_value,
            "result_sha256": execution["result"]["sha256"],
            "execution": execution,
            "execution_sha256": execution_sha256,
        }
        previous_terminal_time = execution_time
        if execution["execution_pass"] is not True:
            prefix_closed = True
    if pair_state(context.receipt_path) != "absent":
        require(
            len(completed) == len(STAGE_NAMES)
            and all(record["execution"]["execution_pass"] is True for record in completed.values()),
            "receipt exists before a complete execution-pass prefix",
        )
    return completed


def raise_terminal_execution(stage: Stage, execution: Mapping[str, Any]) -> None:
    reasons = execution["failure_reasons"]
    message = f"{stage.name} has terminal failed execution evidence: {reasons}"
    if (
        execution["result"]["state"] == "contract_invalid"
        or execution["binding_check"]["postcheck_pass"] is not True
    ):
        raise Contradiction(message)
    raise FinalBenchmarkError(message)


def _record_from_execution(
    context: Context,
    stage: Stage,
    execution: Mapping[str, Any],
    execution_sha256: str,
) -> dict[str, Any]:
    classification = persistent_result_classification(
        classify_result_file(context, stage, stage.canonical)
    )
    require(classification["state"] == "valid", f"{stage.name} execution lacks a valid result")
    return {
        "result": classification["value"],
        "result_sha256": execution["result"]["sha256"],
        "execution": dict(execution),
        "execution_sha256": execution_sha256,
    }


def process_stage(
    context: Context,
    stage: Stage,
    stage_index: int,
    session: Mapping[str, Any],
    session_sha256: str,
    run_subprocess: Callable[..., Any],
) -> dict[str, Any]:
    command = command_for(context, stage)
    launch_evidence = load_launch(
        context, stage, stage_index, session, session_sha256
    )
    if launch_evidence is None:
        require(
            not stage_nonlaunch_evidence_present(stage),
            f"{stage.name} evidence exists without a sealed launch",
        )
        live_pids = exact_child_pids(command, context.proc_root)
        if live_pids:
            raise Busy(f"unbound exact benchmark child is live for {stage.name}: {live_pids}")
        verify_runtime_bindings(context)
        launch, launch_sha256, launch_created = ensure_launch(
            context, stage, stage_index, session, session_sha256
        )
    else:
        launch, launch_sha256 = launch_evidence
        launch_created = False

    existing_execution = load_execution(
        context,
        stage,
        stage_index,
        session,
        session_sha256,
        launch,
        launch_sha256,
    )
    if existing_execution is not None:
        execution, execution_sha256 = existing_execution
        if execution["execution_pass"] is not True:
            raise_terminal_execution(stage, execution)
        return _record_from_execution(
            context, stage, execution, execution_sha256
        )

    live_pids = exact_child_pids(command, context.proc_root)
    if live_pids:
        raise Busy(f"exact benchmark child is still live for {stage.name}: {live_pids}")
    verify_runtime_bindings(context)
    recovered = reconcile_result_candidate(
        context, stage, allow_partial_tmp_cleanup=True
    )
    if recovered["state"] != "absent" or (
        not launch_created and recovered["retryable_partial"] is not True
    ):
        execution = build_execution(
            context,
            stage,
            stage_index,
            session,
            session_sha256,
            launch_sha256,
            process_outcome="unknown_parent_restart",
            returncode=None,
            postcheck_pass=True,
            postcheck_error=None,
            result_classification=recovered,
        )
        execution, _ = publish_execution(
            context,
            stage,
            stage_index,
            session,
            session_sha256,
            launch,
            launch_sha256,
            execution,
        )
        raise_terminal_execution(stage, execution)

    live_pids = exact_child_pids(command, context.proc_root)
    if live_pids:
        raise Busy(f"exact benchmark child appeared for {stage.name}: {live_pids}")
    verify_runtime_bindings(context)
    process_outcome = "launch_error"
    returncode: int | None = None
    try:
        completed = run_subprocess(
            command,
            cwd=context.source_repo,
            env=child_environment(context, stage),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=stage.timeout_seconds,
        )
        observed_returncode = getattr(completed, "returncode", None)
        if isinstance(observed_returncode, int) and not isinstance(observed_returncode, bool):
            process_outcome = "exited"
            returncode = observed_returncode
    except subprocess.TimeoutExpired:
        process_outcome = "timeout"
    except OSError:
        process_outcome = "launch_error"

    try:
        verify_runtime_bindings(context)
    except Exception as error:
        postcheck_pass = False
        postcheck_error = f"{type(error).__name__}: {error}"
    else:
        postcheck_pass = True
        postcheck_error = None

    result_classification = reconcile_result_candidate(
        context, stage, allow_partial_tmp_cleanup=True
    )
    execution = build_execution(
        context,
        stage,
        stage_index,
        session,
        session_sha256,
        launch_sha256,
        process_outcome=process_outcome,
        returncode=returncode,
        postcheck_pass=postcheck_pass,
        postcheck_error=postcheck_error,
        result_classification=result_classification,
    )
    execution, execution_sha256 = publish_execution(
        context,
        stage,
        stage_index,
        session,
        session_sha256,
        launch,
        launch_sha256,
        execution,
    )
    if execution["execution_pass"] is not True:
        raise_terminal_execution(stage, execution)
    return _record_from_execution(
        context, stage, execution, execution_sha256
    )


def _finite_number(value: Any, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    number = float(value)
    require(math.isfinite(number), f"{label} is non-finite")
    return number


def primary_aggregate(
    context: Context,
    completed: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    names = STAGE_NAMES[:3]
    require(all(name in completed for name in names), "cannot aggregate fewer than three GPU repeats")
    results = [completed[name]["result"] for name in names]
    float_timing_fields = (
        "p50_ms",
        "mean_ms",
        "p95_ms",
        "p99_ms",
        "p99_9_ms",
        "max_ms",
        "real_time_factor",
    )
    timing_medians = {
        field: float(statistics.median([
            _finite_number(result.get("timing", {}).get(field), f"{name}.timing.{field}")
            for name, result in zip(names, results, strict=True)
        ]))
        for field in float_timing_fields
    }
    misses: list[int] = []
    for name, result in zip(names, results, strict=True):
        value = result.get("timing", {}).get("deadline_misses")
        require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"{name}.timing.deadline_misses is not a nonnegative integer",
        )
        misses.append(value)
    deadline_median = statistics.median(misses)
    require(isinstance(deadline_median, int), "three-repeat deadline median is not an integer")
    timing_medians["deadline_misses"] = deadline_median
    gate_values: dict[str, list[bool]] = {}
    gate_all: dict[str, bool] = {}
    for path in context.validator.AGGREGATED_HARD_BOOLEAN_PATHS:
        dotted = ".".join(path)
        values = [_nested(result, path, name) for name, result in zip(names, results, strict=True)]
        require(all(isinstance(value, bool) for value in values), f"aggregate gate is not boolean: {dotted}")
        gate_values[dotted] = list(values)
        gate_all[dotted] = all(values)
    constraints = [result.get("constraint_pass") for result in results]
    require(all(isinstance(value, bool) for value in constraints), "aggregate constraint values are invalid")
    memory_median = float(statistics.median([
        _finite_number(result.get("memory_growth_mb"), f"{name}.memory_growth_mb")
        for name, result in zip(names, results, strict=True)
    ]))
    return {
        "schema_version": 1,
        "method": "componentwise timing medians; logical AND for every hard gate",
        "replicate_count": 3,
        "timing_medians": timing_medians,
        "memory_growth_mb_median": memory_median,
        "constraint_passes": list(constraints),
        "all_constraint_pass": all(constraints),
        "hard_gate_values": gate_values,
        "hard_gate_all": gate_all,
    }


AGGREGATE_KEYS = {
    "schema_version", "method", "replicate_count", "timing_medians",
    "memory_growth_mb_median", "constraint_passes", "all_constraint_pass",
    "hard_gate_values", "hard_gate_all",
}
TIMING_MEDIAN_KEYS = {
    "p50_ms", "mean_ms", "p95_ms", "p99_ms", "p99_9_ms", "max_ms",
    "real_time_factor", "deadline_misses",
}


def validate_primary_aggregate_shape(context: Context, aggregate: Mapping[str, Any]) -> None:
    require_exact_keys(aggregate, AGGREGATE_KEYS, "primary aggregate")
    require(type(aggregate.get("schema_version")) is int and aggregate["schema_version"] == 1, "primary aggregate schema changed")
    require(
        aggregate.get("method") == "componentwise timing medians; logical AND for every hard gate",
        "primary aggregate method changed",
    )
    require(
        isinstance(aggregate.get("replicate_count"), int)
        and not isinstance(aggregate.get("replicate_count"), bool)
        and aggregate["replicate_count"] == 3,
        "primary aggregate replicate count changed",
    )
    timing = require_exact_keys(aggregate.get("timing_medians"), TIMING_MEDIAN_KEYS, "timing medians")
    for field in TIMING_MEDIAN_KEYS - {"deadline_misses"}:
        require(isinstance(timing[field], float) and math.isfinite(timing[field]), f"aggregate {field} is not a finite float")
    require(
        isinstance(timing["deadline_misses"], int)
        and not isinstance(timing["deadline_misses"], bool)
        and timing["deadline_misses"] >= 0,
        "aggregate deadline_misses is not a nonnegative integer",
    )
    require(
        isinstance(aggregate.get("memory_growth_mb_median"), float)
        and math.isfinite(aggregate["memory_growth_mb_median"]),
        "aggregate memory median is not a finite float",
    )
    constraints = aggregate.get("constraint_passes")
    require(
        isinstance(constraints, list)
        and len(constraints) == 3
        and all(isinstance(value, bool) for value in constraints),
        "aggregate constraint passes are malformed",
    )
    require(
        isinstance(aggregate.get("all_constraint_pass"), bool)
        and aggregate["all_constraint_pass"] is all(constraints),
        "aggregate constraint AND changed",
    )
    expected_paths = {".".join(path) for path in context.validator.AGGREGATED_HARD_BOOLEAN_PATHS}
    gate_values = require_exact_keys(aggregate.get("hard_gate_values"), expected_paths, "aggregate hard-gate values")
    gate_all = require_exact_keys(aggregate.get("hard_gate_all"), expected_paths, "aggregate hard-gate AND")
    for path in expected_paths:
        values = gate_values[path]
        require(
            isinstance(values, list)
            and len(values) == 3
            and all(isinstance(value, bool) for value in values),
            f"aggregate hard-gate values are malformed: {path}",
        )
        require(
            isinstance(gate_all[path], bool) and gate_all[path] is all(values),
            f"aggregate hard-gate AND changed: {path}",
        )


RESULT_RECORD_KEYS = {
    "result_path", "result_sha256", "execution_path", "execution_sha256",
    "execution_pass", "device", "constraint_pass", "acceptance_pass",
}
RECEIPT_SESSION_KEYS = {"path", "sha256", "identity_sha256"}
RECEIPT_OUTCOME_KEYS = {"status", "failed_stages", "failure_reason"}
RECEIPT_KEYS = {
    "schema_version", "kind", "status", "completed_at_utc", "session",
    "results", "primary_aggregate", "outcome", "receipt_payload_sha256",
}


def result_records(context: Context, completed: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    stage_map = {stage.name: stage for stage in stages(context)}
    for name in STAGE_NAMES:
        require(name in completed, f"missing completed stage record: {name}")
        record = completed[name]
        result = record["result"]
        execution = record["execution"]
        require(execution["execution_pass"] is True, f"receipt cannot bind a failed execution: {name}")
        records[name] = {
            "result_path": str(stage_map[name].canonical),
            "result_sha256": record["result_sha256"],
            "execution_path": str(stage_map[name].execution),
            "execution_sha256": record["execution_sha256"],
            "execution_pass": True,
            "device": stage_map[name].device,
            "constraint_pass": result.get("constraint_pass"),
            "acceptance_pass": acceptance_pass(context, stage_map[name], result),
        }
    return records


def build_receipt(
    context: Context,
    session: Mapping[str, Any],
    session_sha256: str,
    completed: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    records = result_records(context, completed)
    failed_stages = [name for name in STAGE_NAMES if records[name]["acceptance_pass"] is not True]
    status_value = "fail" if failed_stages else "pass"
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "kind": "hs_tasnet_c91_final_benchmark_receipt",
        "status": status_value,
        "completed_at_utc": utc_now(),
        "session": {
            "path": str(session_path(context)),
            "sha256": session_sha256,
            "identity_sha256": session["session_identity_sha256"],
        },
        "results": records,
        "primary_aggregate": primary_aggregate(context, completed),
        "outcome": {
            "status": status_value,
            "failed_stages": list(failed_stages),
            "failure_reason": "acceptance_gate_failed" if failed_stages else None,
        },
    }
    receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
    return receipt


def validate_receipt(
    context: Context,
    session: Mapping[str, Any],
    session_sha256: str,
    receipt: Mapping[str, Any],
) -> None:
    require(set(receipt) == RECEIPT_KEYS, "final benchmark receipt fields changed")
    require(type(receipt.get("schema_version")) is int and receipt["schema_version"] == 1 and receipt.get("kind") == "hs_tasnet_c91_final_benchmark_receipt", "final benchmark receipt schema/kind changed")
    completed_at = validate_utc_timestamp(receipt.get("completed_at_utc"), "receipt.completed_at_utc")
    digest = receipt.get("receipt_payload_sha256")
    require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None, "receipt payload hash is invalid")
    unhashed = dict(receipt)
    unhashed.pop("receipt_payload_sha256", None)
    require(canonical_sha256(unhashed) == digest, "receipt payload hash mismatch")
    receipt_session = require_exact_keys(receipt.get("session"), RECEIPT_SESSION_KEYS, "receipt session")
    require(receipt_session == {
        "path": str(session_path(context)),
        "sha256": session_sha256,
        "identity_sha256": session["session_identity_sha256"],
    }, "receipt session binding changed")
    status_value = receipt.get("status")
    require(status_value in {"pass", "fail"}, "receipt status is invalid")
    raw_results = receipt.get("results")
    require(isinstance(raw_results, dict) and set(raw_results) == set(STAGE_NAMES), "receipt results are malformed")
    stage_map = {stage.name: stage for stage in stages(context)}
    passes: dict[str, bool] = {}
    completed: dict[str, dict[str, Any]] = {}
    latest_execution = datetime.fromisoformat(str(session["created_at_utc"]))
    for stage_index, name in enumerate(STAGE_NAMES, start=1):
        stage = stage_map[name]
        launch_evidence = load_launch(context, stage, stage_index, session, session_sha256)
        require(launch_evidence is not None, f"receipt launch is missing: {name}")
        launch, launch_sha256 = launch_evidence
        execution_evidence = load_execution(
            context, stage, stage_index, session, session_sha256, launch, launch_sha256
        )
        require(execution_evidence is not None, f"receipt execution is missing: {name}")
        execution, execution_sha256 = execution_evidence
        require(execution["execution_pass"] is True, f"receipt binds failed execution: {name}")
        record = _record_from_execution(context, stage, execution, execution_sha256)
        completed[name] = record
        result = record["result"]
        passed = acceptance_pass(context, stage, result)
        passes[name] = passed
        raw_record = require_exact_keys(raw_results[name], RESULT_RECORD_KEYS, f"receipt result {name}")
        require(
            type(raw_record.get("result_path")) is str
            and type(raw_record.get("execution_path")) is str
            and type(raw_record.get("result_sha256")) is str
            and SHA256_RE.fullmatch(raw_record["result_sha256"]) is not None
            and type(raw_record.get("execution_sha256")) is str
            and SHA256_RE.fullmatch(raw_record["execution_sha256"]) is not None
            and raw_record.get("execution_pass") is True
            and type(raw_record.get("device")) is str
            and type(raw_record.get("constraint_pass")) is bool
            and type(raw_record.get("acceptance_pass")) is bool,
            f"receipt result types changed: {name}",
        )
        require(raw_record == {
            "result_path": str(stage.canonical),
            "result_sha256": record["result_sha256"],
            "execution_path": str(stage.execution),
            "execution_sha256": execution_sha256,
            "execution_pass": True,
            "device": stage.device,
            "constraint_pass": result.get("constraint_pass"),
            "acceptance_pass": passed,
        }, f"receipt result binding changed: {name}")
        latest_execution = max(latest_execution, datetime.fromisoformat(str(execution["recorded_at_utc"])))
    require(datetime.fromisoformat(completed_at) >= latest_execution, "receipt predates stage execution")
    aggregate = require_exact_keys(receipt.get("primary_aggregate"), AGGREGATE_KEYS, "receipt primary aggregate")
    validate_primary_aggregate_shape(context, aggregate)
    expected_aggregate = primary_aggregate(context, completed)
    require(canonical_json_bytes(aggregate) == canonical_json_bytes(expected_aggregate), "receipt primary aggregate changed")
    outcome = require_exact_keys(receipt.get("outcome"), RECEIPT_OUTCOME_KEYS, "receipt outcome")
    require(outcome.get("status") == status_value, "receipt outcome changed")
    failed_stages = [name for name in STAGE_NAMES if not passes[name]]
    if status_value == "pass":
        require(not failed_stages, "passing receipt has failed stages")
        require(outcome.get("failed_stages") == [] and outcome.get("failure_reason") is None, "passing receipt records a failure")
    else:
        require(failed_stages and outcome.get("failed_stages") == failed_stages, "failure receipt stages changed")
        require(outcome.get("failure_reason") == "acceptance_gate_failed", "failure receipt reason changed")


def load_or_reconcile_receipt(
    context: Context,
    session: Mapping[str, Any],
    session_sha256: str,
) -> dict[str, Any] | None:
    verify_runtime_bindings(context)
    validator = lambda value: validate_receipt(
        context, session, session_sha256, value
    )
    evidence = load_sealed_document(context.receipt_path, validator)
    return evidence[0] if evidence is not None else None


def publish_receipt(
    context: Context,
    session: Mapping[str, Any],
    session_sha256: str,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    verify_runtime_bindings(context)
    validator = lambda value: validate_receipt(
        context, session, session_sha256, value
    )
    value, _ = publish_sealed_document(context.receipt_path, receipt, validator)
    return value


def run_pipeline(
    context: Context,
    *,
    run_subprocess: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    verify_runtime_bindings(context)
    output_directory_existed = context.output_dir.exists() or context.output_dir.is_symlink()
    context.output_dir.mkdir(mode=0o700, parents=False, exist_ok=True)
    require(context.output_dir.is_dir() and not context.output_dir.is_symlink(), "invalid final benchmark output directory")
    if not output_directory_existed:
        fsync_directory(context.run_dir)
    staging_dir = context.output_dir / ".staging"
    staging_directory_existed = staging_dir.exists() or staging_dir.is_symlink()
    staging_dir.mkdir(mode=0o700, exist_ok=True)
    require(staging_dir.is_dir() and not staging_dir.is_symlink(), "invalid final benchmark staging directory")
    if not staging_directory_existed:
        fsync_directory(context.output_dir)
    cleanup_own_atomic_temps(
        context.run_dir, context.output_dir, staging_dir, proc_root=context.proc_root
    )
    require_session_precedes_stage_evidence(context)
    session, session_sha256 = ensure_session(context)
    completed = validate_stage_prefix(context, session, session_sha256)
    existing_receipt = load_or_reconcile_receipt(context, session, session_sha256)
    if existing_receipt is not None:
        return existing_receipt

    for stage_index, stage in enumerate(stages(context), start=1):
        if stage.name not in completed:
            completed[stage.name] = process_stage(
                context,
                stage,
                stage_index,
                session,
                session_sha256,
                run_subprocess,
            )
        elif completed[stage.name]["execution"]["execution_pass"] is not True:
            raise_terminal_execution(stage, completed[stage.name]["execution"])
    verify_runtime_bindings(context)
    validate_stage_prefix(context, session, session_sha256)
    receipt = build_receipt(context, session, session_sha256, completed)
    return publish_receipt(context, session, session_sha256, receipt)


@contextlib.contextmanager
def exclusive_run_lock(run_dir: Path) -> Iterator[None]:
    lock_path = run_dir / ".run.lock"
    require(lock_path.is_file() and not lock_path.is_symlink(), f"missing/symlink run lock: {lock_path}")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags)
    try:
        require(stat.S_ISREG(os.fstat(descriptor).st_mode), f"run lock is not regular: {lock_path}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise Busy("production run lock is owned by another process") from error
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    try:
        with exclusive_run_lock(run_dir):
            context = build_context(run_dir, args.output_dir)
            receipt = run_pipeline(context)
    except NotReady as error:
        print(f"not_ready: {error}", file=sys.stderr)
        return 2
    except Busy as error:
        print(f"busy: {error}", file=sys.stderr)
        return 2
    except FinalBenchmarkError as error:
        print(f"final_benchmark_error: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"receipt": str(context.receipt_path), "status": receipt["status"]}, sort_keys=True))
    return 0 if receipt["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
