from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import pytest


RUNNER_PATH = Path(__file__).resolve().parents[1] / "run_final_benchmarks.py"
CPU_FIXTURE_PATH = Path(
    "/home/axel/autoresearch/codex/HS-TasNet/research/runs/"
    "jul12r-other-deranged-c71-r1/cpu_diagnostic.json"
)
SPEC = importlib.util.spec_from_file_location("hs_tasnet_final_benchmarks", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)
CPU_FIXTURE = json.loads(CPU_FIXTURE_PATH.read_text(encoding="utf-8"))


class FakeFrozenValidator:
    EXPECTED_PLATFORM = "Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39"
    EXPECTED_PYTHON_VERSION = "3.12.10"
    EXPECTED_TORCH_VERSION = "2.8.0+cu128"
    EXPECTED_NUMPY_VERSION = "2.2.6"
    EXPECTED_TRAIN_CPU_MODEL = "AMD Ryzen 5 5500"
    EXPECTED_TORCH_CUDA_VERSION = "12.8"
    EXPECTED_CUDNN_VERSION = 91002
    AGGREGATED_HARD_BOOLEAN_PATHS = (
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
        ("timing", "mean_headroom_pass"),
        ("timing", "shape_pass"),
        ("timing", "finite_pass"),
        ("timing", "bounded_peak_pass"),
        ("timing", "pass"),
        ("memory", "pass"),
    )

    @staticmethod
    def _validate_primary_benchmark_metadata(value: Mapping[str, Any]) -> None:
        assert value["timing"]["callback_count"] == 10_000

    @staticmethod
    def _validate_timing_evidence(value: Mapping[str, Any]) -> None:
        assert value["callback_count"] == 10_000


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(runner.canonical_json_bytes(value))


def make_context(tmp_path: Path) -> runner.Context:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    deployment = run_dir / "final-deployment.pt"
    deployment.write_bytes(b"deployment")
    deployment_sha = sha(deployment)
    runner.sidecar_path(deployment).write_text(
        f"{deployment_sha}  {deployment.name}\n", encoding="ascii"
    )

    source = tmp_path / "source"
    (source / "hs_tasnet").mkdir(parents=True)
    (source / "hs_tasnet/__init__.py").write_text("", encoding="utf-8")
    (source / "research").mkdir()
    for name in ("run_config.json", "benchmark_streaming.py", "run_experiment.py"):
        (source / "research" / name).write_text("{}\n", encoding="utf-8")

    evidence_paths = {
        "final_report": run_dir / "final_report.json",
        "status": run_dir / "status.json",
        "events": run_dir / "events.jsonl",
        "final_audit_receipt": run_dir / "final_audit_receipt.json",
        "deployment_metadata": deployment.with_suffix(".pt.json"),
        "deployment_sidecar": runner.sidecar_path(deployment),
        "run_contract": run_dir / "run_contract.json",
        "checkpoint_pointer": run_dir / "checkpoints/latest.json",
    }
    for name, path in evidence_paths.items():
        if name == "deployment_sidecar":
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{{\"evidence\":\"{name}\"}}\n", encoding="utf-8")
    run_evidence = {
        name: {"path": str(path.resolve()), "sha256": sha(path)}
        for name, path in evidence_paths.items()
    }

    full_config = tmp_path / "full_config.json"
    full_config.write_text("{}\n", encoding="utf-8")
    automation = tmp_path / "runner.py"
    automation.write_text("# frozen runner\n", encoding="utf-8")
    watchdog = tmp_path / "watchdog.py"
    watchdog.write_text("# frozen watchdog\n", encoding="utf-8")
    benchmark_args = {
        "callbacks": 10_000,
        "device": "cuda:0",
        "expected-cuda-device-name": "NVIDIA GeForce RTX 5090",
        "expected-cuda-device-uuid": "GPU-78229470-2cc7-fa8a-0ff1-69923eedffb2",
        "cpu-affinity": "0,2,4,6",
        "cpu-threads": 4,
        "interop-threads": 1,
        "max-cuda-allocated-growth-mb": 64.0,
        "max-cuda-reserved-growth-mb": 64.0,
        "max-rss-growth-mb": 128.0,
        "max-rss-quartile-growth-mb": 8.0,
        "max-rss-slope-growth-mb-per-10k": 8.0,
        "memory-sample-interval": 1_000,
        "memory-trend-min-samples": 8,
    }
    cpu_args = {
        key: value
        for key, value in benchmark_args.items()
        if not key.startswith("expected-cuda")
    }
    cpu_args["device"] = "cpu"
    return runner.Context(
        run_dir=run_dir,
        output_dir=run_dir / "final-benchmarks",
        receipt_path=run_dir / "final-benchmark-receipt.json",
        deployment_path=deployment,
        deployment_sha256=deployment_sha,
        deployment_model_state_sha256="b" * 64,
        final_report_sha256=run_evidence["final_report"]["sha256"],
        audit_receipt_sha256=run_evidence["final_audit_receipt"]["sha256"],
        audit_payload_sha256="e" * 64,
        contract_identity_sha256="f" * 64,
        source_repo=source,
        source_commit=runner.SOURCE_COMMIT,
        source_tree=runner.SOURCE_TREE,
        run_config_path=source / "research/run_config.json",
        run_config_sha256=sha(source / "research/run_config.json"),
        benchmark_script=source / "research/benchmark_streaming.py",
        benchmark_sha256=sha(source / "research/benchmark_streaming.py"),
        orchestrator_path=source / "research/run_experiment.py",
        orchestrator_sha256=sha(source / "research/run_experiment.py"),
        watchdog_path=watchdog,
        watchdog_sha256=sha(watchdog),
        runner_path=automation,
        runner_sha256=sha(automation),
        full_config_path=full_config,
        full_config_sha256=sha(full_config),
        frozen_files={},
        run_evidence_files=run_evidence,
        trainer_state_evidence={},
        benchmark_args=benchmark_args,
        benchmark_timeout_seconds=1200.0,
        cpu_args=cpu_args,
        cpu_timeout_seconds=1200.0,
        validator=FakeFrozenValidator,
        proc_root=proc_root,
    )


def gpu_result(context: runner.Context) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "checkpoint": str(context.deployment_path),
        "local_package_path": str((context.source_repo / "hs_tasnet/__init__.py").resolve()),
        "requested_device": "cuda:0",
        "device": "cuda:0",
        "device_identity": {
            "identity_pass": True,
            "native_pre_editable_import": {"pass": True},
        },
        "dtype": "torch.float32",
        "runtime": {},
        "runtime_profile_pass": True,
        "execution_backend": {"pass": True},
        "precision": {"pass": True},
        "model": {
            "parameter_count": runner.EXPECTED_PARAMETER_COUNT,
            "deployment_config_pass": True,
            "fp32_state_pass": True,
            "eager_backend_pass": True,
        },
        "correctness": {
            "pass": True,
            "silence": {"pass": True},
            "full_scale": {"pass": True},
        },
        "prefix_causality": {"pass": True},
        "impulse_alignment": {"pass": True},
        "algorithmic_latency_samples": 512,
        "algorithmic_latency_ms": 512 / 44.1,
        "latency_pass": True,
        "timing": {
            "callback_count": 10_000,
            "p50_ms": 1.0,
            "mean_ms": 1.0,
            "p95_ms": 1.0,
            "p99_ms": 1.0,
            "p99_9_ms": 1.0,
            "max_ms": 1.0,
            "real_time_factor": 0.1,
            "deadline_misses": 0,
            "mean_headroom_pass": True,
            "shape_pass": True,
            "finite_pass": True,
            "bounded_peak_pass": True,
            "pass": True,
        },
        "memory": {"pass": True},
        "memory_growth_mb": 1.0,
        "peak_vram_mb": 1.0,
        "determinism_pass": True,
        "threading_pass": True,
        "constraint_pass": True,
    }


def cpu_result(context: runner.Context) -> dict[str, Any]:
    value = copy.deepcopy(CPU_FIXTURE)
    value["checkpoint"] = str(context.deployment_path)
    value["local_package_path"] = str((context.source_repo / "hs_tasnet/__init__.py").resolve())
    cuda_environment = value["device_identity"]["cuda_environment"]
    cuda_environment.update({
        "cuda_visible_devices_present": True,
        "cuda_visible_devices": "",
        "mapping_source": "CUDA_VISIBLE_DEVICES",
    })
    control_records = [value["runtime"]["initial_control_state"]]
    for boundary in value["runtime"]["profile_boundaries"].values():
        control_records.extend([
            boundary["observed_before_reassert"],
            boundary["observed_after_reassert"],
        ])
    for control in control_records:
        control["cuda_visible_devices_present"] = True
        control["cuda_visible_devices"] = ""
    constructor = value["execution_backend"]["checks"]["checkpoint_constructor_config"]
    constructor["keys"] = list(runner.CPU_CONSTRUCTOR_CONFIG_KEYS)
    for phase in ("pre_probes", "pre_timing", "post_timing"):
        phase_constructor = value["execution_backend"]["checks"][phase]["constructor_config"]
        phase_constructor["keys"] = list(runner.CPU_CONSTRUCTOR_CONFIG_KEYS)
        precision = value["precision"]["checks"][phase]
        precision.update({
            "parameter_tensor_count": 22,
            "parameter_element_count": runner.EXPECTED_PARAMETER_COUNT,
            "buffer_tensor_count": 4,
            "buffer_element_count": 3_076,
            "floating_parameter_tensor_count": 22,
            "floating_buffer_tensor_count": 4,
            "floating_tensor_count": 26,
            "floating_element_count": 30_130_286,
            "floating_dtype_tensor_counts": {"torch.float32": 26},
            "floating_device_tensor_counts": {"cpu": 26},
        })
    return value


def apply_overrides(value: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    for dotted, replacement in overrides.items():
        cursor: Any = value
        parts = dotted.split(".")
        for part in parts[:-1]:
            cursor = cursor[part]
        cursor[parts[-1]] = replacement


class FakeSubprocess:
    def __init__(
        self,
        context: runner.Context,
        behaviours: Mapping[str, Mapping[str, Any]] | None = None,
        assertion: Callable[[list[str]], None] | None = None,
    ) -> None:
        self.context = context
        self.behaviours = behaviours or {}
        self.assertion = assertion
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        if self.assertion is not None:
            self.assertion(command)
        output = Path(command[command.index("--json-output") + 1])
        stage_name = output.stem
        behaviour = self.behaviours.get(stage_name, {})
        value = cpu_result(self.context) if stage_name == "cpu_diagnostic" else gpu_result(self.context)
        if "value" in behaviour:
            value = copy.deepcopy(behaviour["value"])
        apply_overrides(value, behaviour.get("overrides", {}))
        destination: Path | None = output
        write_mode = behaviour.get("write", "staging")
        if write_mode == "tmp":
            destination = output.with_name(output.name + ".tmp")
        elif write_mode == "none":
            destination = None
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if write_mode == "partial_tmp":
                destination = output.with_name(output.name + ".tmp")
                destination.write_text("{partial", encoding="utf-8")
            else:
                write_json(destination, value)
        callback = behaviour.get("after_write")
        if callback is not None:
            callback()
        outcome = behaviour.get("outcome", "exit")
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs.get("timeout", 0))
        if outcome == "oserror":
            raise OSError("synthetic launch failure")
        return subprocess.CompletedProcess(
            command,
            int(behaviour.get("returncode", 0)),
            stdout="",
            stderr=str(behaviour.get("stderr", "")),
        )


def prepare_session(context: runner.Context) -> tuple[dict[str, Any], str]:
    context.output_dir.mkdir()
    (context.output_dir / ".staging").mkdir()
    return runner.ensure_session(context)


def prepare_launch(
    context: runner.Context,
    stage_index: int,
    session: Mapping[str, Any],
    session_sha: str,
) -> tuple[runner.Stage, dict[str, Any], str]:
    stage = runner.stages(context)[stage_index - 1]
    launch, launch_sha, _ = runner.ensure_launch(
        context, stage, stage_index, session, session_sha
    )
    return stage, launch, launch_sha


def read_sealed(path: Path) -> dict[str, Any]:
    assert runner.pair_state(path) == "pair"
    runner.verify_sidecar(path)
    return runner.strict_json(path)[0]


def rewrite_sealed(path: Path, value: Mapping[str, Any]) -> None:
    write_json(path, value)
    runner.seal_existing_json(path)


def create_live_exact_child(context: runner.Context, stage: runner.Stage, pid: int = 4242) -> Path:
    process = context.proc_root / str(pid)
    process.mkdir()
    command = runner.command_for(context, stage)
    process.joinpath("cmdline").write_bytes(b"\0".join(item.encode() for item in command) + b"\0")
    return process


def test_commands_environment_and_session_exist_before_every_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = make_context(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    def assert_sealed(command: list[str]) -> None:
        session = read_sealed(runner.session_path(context))
        assert session["bindings"] == runner.bindings(context)
        output = Path(command[command.index("--json-output") + 1])
        launch = context.output_dir / f"{output.stem}.launch.json"
        assert runner.pair_state(launch) == "pair"

    child = FakeSubprocess(context, assertion=assert_sealed)
    receipt = runner.run_pipeline(context, run_subprocess=child)
    assert receipt["status"] == "pass"
    assert len(child.commands) == 4
    assert list(receipt["results"]) == list(runner.STAGE_NAMES)
    for command in child.commands:
        assert command[:4] == [sys.executable, "-P", "-B", str(context.benchmark_script)]
        assert command.count("--report-only") == 1
    gpu_env = runner.child_environment(context, runner.stages(context)[0])
    cpu_env = runner.child_environment(context, runner.stages(context)[-1])
    assert "CUDA_VISIBLE_DEVICES" not in gpu_env
    assert "CUDA_DEVICE_ORDER" not in gpu_env and "CUBLAS_WORKSPACE_CONFIG" not in gpu_env
    assert cpu_env["CUDA_VISIBLE_DEVICES"] == ""
    assert "CUDA_DEVICE_ORDER" not in cpu_env and "CUBLAS_WORKSPACE_CONFIG" not in cpu_env


def test_receipt_reuse_and_json_only_reconciliation_never_reruns(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    first_child = FakeSubprocess(context)
    first = runner.run_pipeline(context, run_subprocess=first_child)
    runner.sidecar_path(context.receipt_path).unlink()
    never = FakeSubprocess(context)
    second = runner.run_pipeline(context, run_subprocess=never)
    assert second == first
    assert not never.commands
    assert runner.pair_state(context.receipt_path) == "pair"


def test_ordinary_scientific_miss_continues_all_stages(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    child = FakeSubprocess(context, {
        "benchmark_1": {"overrides": {"timing.pass": False, "constraint_pass": False}}
    })
    receipt = runner.run_pipeline(context, run_subprocess=child)
    assert len(child.commands) == 4
    assert receipt["status"] == "fail"
    assert receipt["outcome"] == {
        "status": "fail",
        "failed_stages": ["benchmark_1"],
        "failure_reason": "acceptance_gate_failed",
    }
    assert read_sealed(runner.stages(context)[0].execution)["execution_pass"] is True


def test_cpu_timing_thresholds_are_observational(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    child = FakeSubprocess(context, {
        "cpu_diagnostic": {"overrides": {
            "timing.mean_headroom_pass": False,
            "timing.pass": False,
            "constraint_pass": False,
        }}
    })
    receipt = runner.run_pipeline(context, run_subprocess=child)
    assert receipt["status"] == "pass"
    assert receipt["results"]["cpu_diagnostic"]["constraint_pass"] is False
    assert receipt["results"]["cpu_diagnostic"]["acceptance_pass"] is True


@pytest.mark.parametrize("mode", ["nonzero", "timeout"])
def test_valid_nonzero_and_timeout_results_are_preserved_and_never_rerun(
    tmp_path: Path, mode: str
) -> None:
    context = make_context(tmp_path)
    behaviour = {"returncode": 9} if mode == "nonzero" else {"outcome": "timeout"}
    first_child = FakeSubprocess(context, {"benchmark_1": behaviour})
    with pytest.raises(runner.FinalBenchmarkError, match="terminal failed execution"):
        runner.run_pipeline(context, run_subprocess=first_child)
    stage = runner.stages(context)[0]
    assert runner.pair_state(stage.canonical) == "pair"
    execution = read_sealed(stage.execution)
    assert execution["result"]["state"] == "valid"
    assert execution["execution_pass"] is False
    assert execution["process"]["outcome"] == ("exited" if mode == "nonzero" else "timeout")
    never = FakeSubprocess(context)
    with pytest.raises(runner.FinalBenchmarkError, match="terminal failed execution"):
        runner.run_pipeline(context, run_subprocess=never)
    assert not never.commands


def test_unknown_parent_restart_with_valid_result_is_sealed_and_not_cherry_picked(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    session, session_sha = prepare_session(context)
    stage, _, _ = prepare_launch(context, 1, session, session_sha)
    write_json(stage.staging, gpu_result(context))
    never = FakeSubprocess(context)
    with pytest.raises(runner.FinalBenchmarkError, match="process_outcome_unknown"):
        runner.run_pipeline(context, run_subprocess=never)
    assert not never.commands
    execution = read_sealed(stage.execution)
    assert execution["process"]["outcome"] == "unknown_parent_restart"
    assert execution["result"]["state"] == "valid"
    assert execution["execution_pass"] is False
    second = FakeSubprocess(context)
    with pytest.raises(runner.FinalBenchmarkError):
        runner.run_pipeline(context, run_subprocess=second)
    assert not second.commands


def test_existing_launch_without_result_is_terminal_unknown_and_never_rerun(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    session, session_sha = prepare_session(context)
    stage, _, _ = prepare_launch(context, 1, session, session_sha)
    never = FakeSubprocess(context)
    with pytest.raises(runner.FinalBenchmarkError, match="process_outcome_unknown"):
        runner.run_pipeline(context, run_subprocess=never)
    assert not never.commands
    execution = read_sealed(stage.execution)
    assert execution["process"]["outcome"] == "unknown_parent_restart"
    assert execution["result"]["state"] == "absent"
    assert execution["execution_pass"] is False
    second = FakeSubprocess(context)
    with pytest.raises(runner.FinalBenchmarkError, match="terminal failed execution"):
        runner.run_pipeline(context, run_subprocess=second)
    assert not second.commands


def test_complete_contract_invalid_tmp_is_promoted_sealed_and_permanent(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    bad = gpu_result(context)
    bad["checkpoint"] = str(tmp_path / "wrong.pt")
    first = FakeSubprocess(context, {"benchmark_1": {"write": "tmp", "value": bad}})
    with pytest.raises(runner.Contradiction, match="scientific_contract_invalid"):
        runner.run_pipeline(context, run_subprocess=first)
    stage = runner.stages(context)[0]
    assert runner.pair_state(stage.canonical) == "pair"
    assert read_sealed(stage.canonical)["checkpoint"] == str(tmp_path / "wrong.pt")
    execution = read_sealed(stage.execution)
    assert execution["result"]["state"] == "contract_invalid"
    assert execution["execution_pass"] is False
    never = FakeSubprocess(context)
    with pytest.raises(runner.Contradiction, match="terminal failed execution"):
        runner.run_pipeline(context, run_subprocess=never)
    assert not never.commands


def test_partial_native_tmp_retries_only_after_exact_child_is_dead(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    session, session_sha = prepare_session(context)
    stage, _, _ = prepare_launch(context, 1, session, session_sha)
    temporary = stage.staging.with_name(stage.staging.name + ".tmp")
    temporary.write_text("{partial", encoding="utf-8")
    live = create_live_exact_child(context, stage)
    child = FakeSubprocess(context)
    with pytest.raises(runner.Busy, match="still live"):
        runner.run_pipeline(context, run_subprocess=child)
    assert temporary.read_text(encoding="utf-8") == "{partial"
    assert not child.commands
    live.joinpath("cmdline").unlink()
    live.rmdir()
    receipt = runner.run_pipeline(context, run_subprocess=child)
    assert receipt["status"] == "pass"
    assert len(child.commands) == 4
    assert not temporary.exists()


def test_unbound_live_exact_child_blocks_launch(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    context.output_dir.mkdir()
    (context.output_dir / ".staging").mkdir()
    stage = runner.stages(context)[0]
    create_live_exact_child(context, stage)
    child = FakeSubprocess(context)
    with pytest.raises(runner.Busy, match="predates sealed session"):
        runner.run_pipeline(context, run_subprocess=child)
    assert runner.pair_state(runner.session_path(context)) == "absent"
    assert runner.pair_state(stage.launch) == "absent"
    assert not child.commands


def test_post_child_binding_violation_remains_failed_after_file_restored(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    status = Path(context.run_evidence_files["status"]["path"])
    original = status.read_bytes()

    def mutate() -> None:
        status.write_bytes(b"mutated\n")

    first = FakeSubprocess(context, {"benchmark_1": {"after_write": mutate}})
    with pytest.raises(runner.Contradiction, match="postcheck_binding_violation"):
        runner.run_pipeline(context, run_subprocess=first)
    stage = runner.stages(context)[0]
    execution = read_sealed(stage.execution)
    assert execution["binding_check"]["postcheck_pass"] is False
    assert execution["result"]["state"] == "valid"
    status.write_bytes(original)
    never = FakeSubprocess(context)
    with pytest.raises(runner.Contradiction, match="terminal failed execution"):
        runner.run_pipeline(context, run_subprocess=never)
    assert not never.commands


def test_first_empty_stage_closes_prefix_against_later_launch(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    session, session_sha = prepare_session(context)
    prepare_launch(context, 2, session, session_sha)
    child = FakeSubprocess(context)
    with pytest.raises(runner.Contradiction, match="out-of-order launch"):
        runner.run_pipeline(context, run_subprocess=child)
    assert not child.commands


def test_result_without_session_or_launch_fails_closed(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    stage = runner.stages(context)[0]
    stage.canonical.parent.mkdir()
    write_json(stage.canonical, gpu_result(context))
    with pytest.raises(runner.Contradiction, match="predates sealed session"):
        runner.run_pipeline(context, run_subprocess=FakeSubprocess(context))

    context = make_context(tmp_path / "second")
    prepare_session(context)
    stage = runner.stages(context)[0]
    write_json(stage.canonical, gpu_result(context))
    with pytest.raises(runner.Contradiction, match="without a sealed launch"):
        runner.run_pipeline(context, run_subprocess=FakeSubprocess(context))


def test_candidate_is_fsynced_before_promotion_and_both_directories_are_synced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = make_context(tmp_path)
    fsynced_files: list[Path] = []
    fsynced_directories: list[Path] = []
    real_file = runner.fsync_file
    real_directory = runner.fsync_directory

    def record_file(path: Path) -> None:
        fsynced_files.append(path)
        real_file(path)

    def record_directory(path: Path) -> None:
        fsynced_directories.append(path)
        real_directory(path)

    monkeypatch.setattr(runner, "fsync_file", record_file)
    monkeypatch.setattr(runner, "fsync_directory", record_directory)
    runner.run_pipeline(context, run_subprocess=FakeSubprocess(context))
    first = runner.stages(context)[0]
    assert first.staging in fsynced_files
    assert first.staging.parent in fsynced_directories
    assert first.canonical.parent in fsynced_directories
    assert runner.pair_state(first.canonical) == "pair"


def test_new_output_and_staging_directories_are_synced_before_first_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = make_context(tmp_path)
    events: list[tuple[str, Path | str]] = []
    real_directory = runner.fsync_directory

    def record_directory(path: Path) -> None:
        events.append(("fsync", path))
        real_directory(path)

    def record_child(command: list[str]) -> None:
        events.append(("child", Path(command[command.index("--json-output") + 1]).stem))

    monkeypatch.setattr(runner, "fsync_directory", record_directory)
    runner.run_pipeline(
        context,
        run_subprocess=FakeSubprocess(context, assertion=record_child),
    )
    first_child = next(index for index, event in enumerate(events) if event[0] == "child")
    run_sync = events.index(("fsync", context.run_dir))
    output_sync = events.index(("fsync", context.output_dir))
    assert run_sync < first_child
    assert output_sync < first_child


def test_atomic_temp_cleanup_rejects_live_owner_and_removes_only_stale(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    live_pid = 7001
    stale_pid = 7002
    (context.proc_root / str(live_pid)).mkdir()
    live = context.run_dir / f".finalbench-atomic-receipt.json-{live_pid}-{'a' * 32}.tmp"
    stale = context.run_dir / f".finalbench-atomic-receipt.json-{stale_pid}-{'b' * 32}.tmp"
    live.write_text("live", encoding="utf-8")
    stale.write_text("stale", encoding="utf-8")
    with pytest.raises(runner.Contradiction, match="owner is still live"):
        runner.cleanup_own_atomic_temps(context.run_dir, proc_root=context.proc_root)
    assert live.exists()
    (context.proc_root / str(live_pid)).rmdir()
    removed = runner.cleanup_own_atomic_temps(context.run_dir, proc_root=context.proc_root)
    assert removed == sorted([str(live), str(stale)])
    assert not live.exists() and not stale.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["runtime"]["determinism"].update({"extra": True}),
        lambda value: value["execution_backend"]["checks"]["pre_probes"].update({"extra": True}),
        lambda value: value["execution_backend"]["checks"]["checkpoint_constructor_config"].update({"raw_type": "str"}),
        lambda value: value["runtime"]["profile_boundaries"]["pre_timing"].update({"interop_thread_control": "changed"}),
        lambda value: value["precision"]["checks"]["post_timing"].update({"buffer_tensor_count": 5}),
        lambda value: value["precision"]["checks"]["post_timing"].update({"floating_element_count": 30_130_287}),
        lambda value: value["precision"]["checks"]["post_timing"].update({"parameter_tensor_count": 22.0}),
        lambda value: value["runtime"].update({"cpu_threads_effective": 4.0}),
        lambda value: value["model"].update({"sample_rate": 44_100.0}),
        lambda value: value["memory"].update({"cuda_allocated_growth_mb": 1.0}),
        lambda value: value["memory"].update({"rss_source": "fabricated"}),
        lambda value: value["memory"].update({"max_rss_growth_mb": 129.0}),
        lambda value: value["memory"]["rss_samples"][1].update({"callback": 1_001}),
        lambda value: value["memory"]["rss_trend"].update({"sample_count": 10}),
        lambda value: value["memory"].update({"extra": True}),
        lambda value: value["device_identity"]["cuda_environment"].update({"cuda_visible_devices_present": False}),
    ],
)
def test_real_shaped_cpu_metadata_mutations_are_contract_invalid(
    tmp_path: Path, mutation: Callable[[dict[str, Any]], None]
) -> None:
    context = make_context(tmp_path)
    value = cpu_result(context)
    mutation(value)
    with pytest.raises(runner.Contradiction):
        runner.validate_cpu_metadata(context, value)


def test_cpu_result_fixture_is_strictly_accepted(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    stage = runner.stages(context)[-1]
    value = cpu_result(context)
    runner.validate_cpu_metadata(context, value)
    assert runner.acceptance_pass(context, stage, value) is True


@pytest.mark.parametrize("document", ["session", "launch", "execution", "receipt"])
def test_extra_document_keys_fail_closed(tmp_path: Path, document: str) -> None:
    context = make_context(tmp_path)
    if document in {"session", "launch"}:
        session, session_sha = prepare_session(context)
        path = runner.session_path(context)
        if document == "launch":
            stage, _, _ = prepare_launch(context, 1, session, session_sha)
            path = stage.launch
    else:
        runner.run_pipeline(context, run_subprocess=FakeSubprocess(context))
        path = context.receipt_path
        if document == "execution":
            path = runner.stages(context)[0].execution
    value = read_sealed(path)
    value["extra"] = True
    rewrite_sealed(path, value)
    with pytest.raises(runner.Contradiction, match="fields changed"):
        runner.run_pipeline(context, run_subprocess=FakeSubprocess(context))


@pytest.mark.parametrize("document", ["session", "launch", "execution", "receipt"])
def test_noncanonical_document_timestamps_fail_closed(tmp_path: Path, document: str) -> None:
    context = make_context(tmp_path)
    if document in {"session", "launch"}:
        session, session_sha = prepare_session(context)
        path = runner.session_path(context)
        timestamp = "created_at_utc"
        payload = "session_payload_sha256"
        if document == "launch":
            stage, _, _ = prepare_launch(context, 1, session, session_sha)
            path = stage.launch
            timestamp = "launched_at_utc"
            payload = "launch_payload_sha256"
    else:
        runner.run_pipeline(context, run_subprocess=FakeSubprocess(context))
        path = context.receipt_path
        timestamp = "completed_at_utc"
        payload = "receipt_payload_sha256"
        if document == "execution":
            path = runner.stages(context)[0].execution
            timestamp = "recorded_at_utc"
            payload = "execution_payload_sha256"
    value = read_sealed(path)
    value[timestamp] = str(value[timestamp]).replace("+00:00", "Z")
    value.pop(payload)
    value[payload] = runner.canonical_sha256(value)
    rewrite_sealed(path, value)
    with pytest.raises(runner.Contradiction, match="canonical UTC"):
        runner.run_pipeline(context, run_subprocess=FakeSubprocess(context))


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("protocol", "primary_repetitions"), 3.0),
        (("protocol", "benchmark_timeout_seconds"), 1200),
        (("protocol", "benchmark_args", "callbacks"), 10_000.0),
    ],
)
def test_nested_session_binding_numeric_types_are_exact(
    tmp_path: Path, path: tuple[str, ...], replacement: Any
) -> None:
    context = make_context(tmp_path)
    prepare_session(context)
    session_file = runner.session_path(context)
    session = read_sealed(session_file)
    cursor = session["bindings"]
    for component in path[:-1]:
        cursor = cursor[component]
    assert cursor[path[-1]] == replacement
    assert type(cursor[path[-1]]) is not type(replacement)
    cursor[path[-1]] = replacement
    session["session_identity_sha256"] = runner.canonical_sha256(session["bindings"])
    session.pop("session_payload_sha256")
    session["session_payload_sha256"] = runner.canonical_sha256(session)
    rewrite_sealed(session_file, session)
    with pytest.raises(runner.Contradiction, match="type changed"):
        runner.run_pipeline(context, run_subprocess=FakeSubprocess(context))


def test_aggregate_deadline_median_is_integer_and_hard_gates_are_exact_and(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    child = FakeSubprocess(context, {
        "benchmark_1": {"overrides": {"timing.deadline_misses": 1}},
        "benchmark_2": {"overrides": {"timing.deadline_misses": 9, "timing.pass": False}},
        "benchmark_3": {"overrides": {"timing.deadline_misses": 4}},
    })
    receipt = runner.run_pipeline(context, run_subprocess=child)
    aggregate = receipt["primary_aggregate"]
    assert aggregate["timing_medians"]["deadline_misses"] == 4
    assert type(aggregate["timing_medians"]["deadline_misses"]) is int
    assert aggregate["hard_gate_values"]["timing.pass"] == [True, False, True]
    assert aggregate["hard_gate_all"]["timing.pass"] is False
    mutated = copy.deepcopy(receipt)
    mutated["primary_aggregate"]["timing_medians"]["deadline_misses"] = 4.0
    mutated.pop("receipt_payload_sha256")
    mutated["receipt_payload_sha256"] = runner.canonical_sha256(mutated)
    rewrite_sealed(context.receipt_path, mutated)
    with pytest.raises(runner.Contradiction, match="deadline_misses"):
        runner.run_pipeline(context, run_subprocess=FakeSubprocess(context))


def test_receipt_outcome_schema_is_exact(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    receipt = runner.run_pipeline(context, run_subprocess=FakeSubprocess(context))
    receipt["outcome"]["extra"] = True
    receipt.pop("receipt_payload_sha256")
    receipt["receipt_payload_sha256"] = runner.canonical_sha256(receipt)
    rewrite_sealed(context.receipt_path, receipt)
    with pytest.raises(runner.Contradiction, match="receipt outcome fields changed"):
        runner.run_pipeline(context, run_subprocess=FakeSubprocess(context))


def test_bound_run_evidence_mutation_fails_before_child(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    events = Path(context.run_evidence_files["events"]["path"])
    events.write_text("mutated\n", encoding="utf-8")
    child = FakeSubprocess(context)
    with pytest.raises(runner.Contradiction, match="bound run evidence changed"):
        runner.run_pipeline(context, run_subprocess=child)
    assert not child.commands


@pytest.mark.parametrize(
    ("active", "sub", "pid", "accepted"),
    [
        ("inactive", "dead", "0", True),
        ("failed", "failed", "0", True),
        ("failed", "dead", "0", True),
        ("active", "running", "12", False),
        ("activating", "start", "0", False),
        ("inactive", "dead", "12", False),
    ],
)
def test_trainer_service_terminal_state_compatibility(
    monkeypatch: pytest.MonkeyPatch,
    active: str,
    sub: str,
    pid: str,
    accepted: bool,
) -> None:
    stdout = (
        "LoadState=loaded\n"
        f"ActiveState={active}\n"
        f"SubState={sub}\n"
        f"MainPID={pid}\n"
        f"InvocationID={'a' * 32}\n"
    )

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args[0], 0, stdout=stdout, stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    if accepted:
        assert runner.read_trainer_services(["trainer.service"])["trainer.service"]["InvocationID"] == "a" * 32
    else:
        with pytest.raises(runner.Contradiction):
            runner.read_trainer_services(["trainer.service"])


def test_trainer_invocation_binding_is_rechecked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = make_context(tmp_path)
    context = runner.Context(
        **{
            **context.__dict__,
            "trainer_services": ("trainer.service",),
            "trainer_state_evidence": {"trainer.service": {"InvocationID": "a" * 32}},
            "enforce_trainer_inactive": True,
        }
    )
    monkeypatch.setattr(runner, "read_trainer_services", lambda services: {
        "trainer.service": {
            "LoadState": "failed", "ActiveState": "failed", "SubState": "dead",
            "MainPID": "0", "InvocationID": "a" * 32,
        }
    })
    runner.verify_trainer_services_inactive(context)
    monkeypatch.setattr(runner, "read_trainer_services", lambda services: {
        "trainer.service": {
            "LoadState": "inactive", "ActiveState": "inactive", "SubState": "dead",
            "MainPID": "0", "InvocationID": "b" * 32,
        }
    })
    with pytest.raises(runner.Contradiction, match="invocation changed"):
        runner.verify_trainer_services_inactive(context)


def test_empty_trainer_invocation_cannot_change_to_nonempty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = make_context(tmp_path)
    context = runner.Context(
        **{
            **context.__dict__,
            "trainer_services": ("trainer.service",),
            "trainer_state_evidence": {"trainer.service": {"InvocationID": ""}},
            "enforce_trainer_inactive": True,
        }
    )
    monkeypatch.setattr(runner, "read_trainer_services", lambda services: {
        "trainer.service": {
            "LoadState": "failed", "ActiveState": "failed", "SubState": "dead",
            "MainPID": "0", "InvocationID": "c" * 32,
        }
    })
    with pytest.raises(runner.Contradiction, match="invocation changed"):
        runner.verify_trainer_services_inactive(context)


def test_load_module_registers_dataclass_module(tmp_path: Path) -> None:
    module_path = tmp_path / "with_dataclass.py"
    module_path.write_text(
        "from dataclasses import dataclass\n@dataclass\nclass Value:\n    item: int\n",
        encoding="utf-8",
    )
    loaded = runner.load_module(module_path, "dataclass_probe")
    assert loaded.Value(7).item == 7
