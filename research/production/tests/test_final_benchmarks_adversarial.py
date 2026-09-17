from __future__ import annotations

import copy
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import pytest


# Reuse only the synthetic scientific-result builders from the primary suite.  The
# assertions below are an independent, adversarial state-machine layer.
FIXTURES_PATH = Path(__file__).with_name("test_final_benchmarks.py")
FIXTURES_SPEC = importlib.util.spec_from_file_location(
    "_hs_tasnet_final_benchmark_test_fixtures", FIXTURES_PATH
)
assert FIXTURES_SPEC is not None and FIXTURES_SPEC.loader is not None
fixtures = importlib.util.module_from_spec(FIXTURES_SPEC)
sys.modules[FIXTURES_SPEC.name] = fixtures
FIXTURES_SPEC.loader.exec_module(fixtures)

runner = fixtures.runner


def _sealed_bytes(path: Path) -> tuple[bytes, bytes]:
    assert runner.pair_state(path) == "pair"
    runner.verify_sidecar(path)
    return path.read_bytes(), runner.sidecar_path(path).read_bytes()


def _write_shape(path: Path, shape: str, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if shape in {"json_only", "pair"}:
        fixtures.write_json(path, value)
    if shape in {"sidecar_only", "pair"}:
        if shape == "pair":
            runner.seal_existing_json(path)
        else:
            runner.sidecar_path(path).write_text(
                f"{'0' * 64}  {path.name}\n", encoding="ascii"
            )


def _remove_fake_process(process: Path) -> None:
    process.joinpath("cmdline").unlink()
    process.rmdir()


def _rehash_document(value: dict[str, Any], payload_field: str) -> None:
    value.pop(payload_field, None)
    value[payload_field] = runner.canonical_sha256(value)


def _document_fixture(
    tmp_path: Path, document: str
) -> tuple[Any, Path, str, str]:
    context = fixtures.make_context(tmp_path)
    if document == "session":
        fixtures.prepare_session(context)
        return context, runner.session_path(context), "created_at_utc", "session_payload_sha256"
    if document == "launch":
        session, session_sha = fixtures.prepare_session(context)
        stage, _, _ = fixtures.prepare_launch(context, 1, session, session_sha)
        return context, stage.launch, "launched_at_utc", "launch_payload_sha256"

    runner.run_pipeline(
        context, run_subprocess=fixtures.FakeSubprocess(context)
    )
    if document == "execution":
        return (
            context,
            runner.stages(context)[0].execution,
            "recorded_at_utc",
            "execution_payload_sha256",
        )
    assert document == "receipt"
    return context, context.receipt_path, "completed_at_utc", "receipt_payload_sha256"


def _mutate_path(value: dict[str, Any], path: tuple[str, ...], mode: str) -> None:
    cursor: dict[str, Any] = value
    for component in path:
        nested = cursor[component]
        assert isinstance(nested, dict)
        cursor = nested
    if mode == "extra":
        cursor["adversarial_extra"] = True
    else:
        assert mode.startswith("missing:")
        cursor.pop(mode.split(":", 1)[1])


def test_session_and_launch_are_durably_sealed_before_each_child(tmp_path: Path) -> None:
    context = fixtures.make_context(tmp_path)
    observations: list[str] = []

    def assert_prelaunch_evidence(command: list[str]) -> None:
        session_path = runner.session_path(context)
        session = fixtures.read_sealed(session_path)
        assert session["bindings"] == runner.bindings(context)
        assert session["session_identity_sha256"] == runner.canonical_sha256(
            session["bindings"]
        )
        output = Path(command[command.index("--json-output") + 1])
        stage_name = output.stem
        stage = next(stage for stage in runner.stages(context) if stage.name == stage_name)
        launch = fixtures.read_sealed(stage.launch)
        assert launch["session_file_sha256"] == runner.sha256_file(session_path)
        assert launch["command"] == command
        observations.append(stage_name)

    child = fixtures.FakeSubprocess(context, assertion=assert_prelaunch_evidence)
    receipt = runner.run_pipeline(context, run_subprocess=child)
    assert receipt["status"] == "pass"
    assert observations == list(runner.STAGE_NAMES)


OUT_OF_ORDER_ARTIFACTS = (
    "launch",
    "execution",
    "canonical",
    "staging",
    "temporary",
    "receipt",
)


@pytest.mark.parametrize("artifact", OUT_OF_ORDER_ARTIFACTS)
@pytest.mark.parametrize("shape", ("json_only", "sidecar_only", "pair"))
def test_every_later_stage_evidence_representation_closes_the_prefix(
    tmp_path: Path, artifact: str, shape: str
) -> None:
    context = fixtures.make_context(tmp_path)
    session, session_sha = fixtures.prepare_session(context)
    later = runner.stages(context)[1]
    if artifact == "launch":
        path = later.launch
        value = runner.build_launch(context, later, 2, session, session_sha)
    elif artifact == "execution":
        path = later.execution
        value = {"synthetic": "later execution"}
    elif artifact == "canonical":
        path = later.canonical
        value = fixtures.gpu_result(context)
    elif artifact == "staging":
        path = later.staging
        value = fixtures.gpu_result(context)
    elif artifact == "temporary":
        path = later.staging.with_name(later.staging.name + ".tmp")
        value = fixtures.gpu_result(context)
    else:
        path = context.receipt_path
        value = {"synthetic": "premature receipt"}
    _write_shape(path, shape, value)

    child = fixtures.FakeSubprocess(context)
    with pytest.raises(runner.Contradiction):
        runner.run_pipeline(context, run_subprocess=child)
    assert child.commands == []
    assert runner.pair_state(runner.stages(context)[0].launch) == "absent"


@pytest.mark.parametrize(
    ("case", "expected_outcome", "expected_reason"),
    (
        ("nonzero", "exited", "process_nonzero_exit"),
        ("timeout", "timeout", "process_timeout"),
        ("recovered", "unknown_parent_restart", "process_outcome_unknown"),
    ),
)
def test_valid_result_with_untrusted_exit_is_sealed_terminal_and_never_rerun(
    tmp_path: Path,
    case: str,
    expected_outcome: str,
    expected_reason: str,
) -> None:
    context = fixtures.make_context(tmp_path)
    stage = runner.stages(context)[0]
    if case == "recovered":
        session, session_sha = fixtures.prepare_session(context)
        fixtures.prepare_launch(context, 1, session, session_sha)
        fixtures.write_json(stage.staging, fixtures.gpu_result(context))
        first_child = fixtures.FakeSubprocess(context)
    else:
        behaviour: dict[str, Any]
        if case == "nonzero":
            behaviour = {"returncode": 23}
        else:
            behaviour = {"outcome": "timeout"}
        first_child = fixtures.FakeSubprocess(
            context, {"benchmark_1": behaviour}
        )

    with pytest.raises(runner.FinalBenchmarkError, match="terminal failed execution"):
        runner.run_pipeline(context, run_subprocess=first_child)
    if case == "recovered":
        assert first_child.commands == []
    else:
        assert len(first_child.commands) == 1

    canonical_before = _sealed_bytes(stage.canonical)
    execution_before = _sealed_bytes(stage.execution)
    execution = fixtures.read_sealed(stage.execution)
    assert execution["process"]["outcome"] == expected_outcome
    assert expected_reason in execution["failure_reasons"]
    assert execution["result"]["state"] == "valid"
    assert execution["execution_pass"] is False

    for _ in range(2):
        never = fixtures.FakeSubprocess(context)
        with pytest.raises(runner.FinalBenchmarkError, match="terminal failed execution"):
            runner.run_pipeline(context, run_subprocess=never)
        assert never.commands == []
        assert _sealed_bytes(stage.canonical) == canonical_before
        assert _sealed_bytes(stage.execution) == execution_before


def test_post_child_binding_mutation_is_terminal_even_after_byte_restoration(
    tmp_path: Path,
) -> None:
    context = fixtures.make_context(tmp_path)
    bound_path = Path(context.run_evidence_files["events"]["path"])
    original = bound_path.read_bytes()

    def mutate_after_result() -> None:
        bound_path.write_bytes(b"temporarily replaced\n")

    child = fixtures.FakeSubprocess(
        context, {"benchmark_1": {"after_write": mutate_after_result}}
    )
    with pytest.raises(runner.Contradiction, match="postcheck_binding_violation"):
        runner.run_pipeline(context, run_subprocess=child)

    stage = runner.stages(context)[0]
    result_before = _sealed_bytes(stage.canonical)
    execution_before = _sealed_bytes(stage.execution)
    execution = fixtures.read_sealed(stage.execution)
    assert execution["binding_check"]["postcheck_pass"] is False
    assert execution["result"]["state"] == "valid"
    assert execution["execution_pass"] is False

    bound_path.write_bytes(original)
    never = fixtures.FakeSubprocess(context)
    with pytest.raises(runner.Contradiction, match="terminal failed execution"):
        runner.run_pipeline(context, run_subprocess=never)
    assert never.commands == []
    assert _sealed_bytes(stage.canonical) == result_before
    assert _sealed_bytes(stage.execution) == execution_before


@pytest.mark.parametrize("has_bound_launch", [False, True], ids=["unbound", "bound"])
def test_exact_live_orphan_blocks_cleanup_and_relaunch(
    tmp_path: Path, has_bound_launch: bool
) -> None:
    context = fixtures.make_context(tmp_path)
    session, session_sha = fixtures.prepare_session(context)
    stage = runner.stages(context)[0]
    partial: Path | None = None
    if has_bound_launch:
        fixtures.prepare_launch(context, 1, session, session_sha)
        partial = stage.staging.with_name(stage.staging.name + ".tmp")
        partial.write_text("{partial", encoding="utf-8")
    process = fixtures.create_live_exact_child(context, stage, pid=6401)
    child = fixtures.FakeSubprocess(context)

    with pytest.raises(runner.Busy, match="exact benchmark child"):
        runner.run_pipeline(context, run_subprocess=child)
    assert child.commands == []
    if partial is not None:
        assert partial.read_bytes() == b"{partial"
    else:
        assert runner.pair_state(stage.launch) == "absent"
    _remove_fake_process(process)


def test_complete_contract_invalid_native_tmp_is_promoted_sealed_and_permanent(
    tmp_path: Path,
) -> None:
    context = fixtures.make_context(tmp_path)
    bad = fixtures.gpu_result(context)
    bad["checkpoint"] = str(tmp_path / "foreign-deployment.pt")
    child = fixtures.FakeSubprocess(
        context, {"benchmark_1": {"write": "tmp", "value": bad}}
    )
    with pytest.raises(runner.Contradiction, match="scientific_contract_invalid"):
        runner.run_pipeline(context, run_subprocess=child)

    stage = runner.stages(context)[0]
    temporary = stage.staging.with_name(stage.staging.name + ".tmp")
    assert not temporary.exists()
    assert fixtures.read_sealed(stage.canonical) == bad
    execution = fixtures.read_sealed(stage.execution)
    assert execution["result"]["state"] == "contract_invalid"
    assert execution["execution_pass"] is False
    durable_result = _sealed_bytes(stage.canonical)
    durable_execution = _sealed_bytes(stage.execution)

    never = fixtures.FakeSubprocess(context)
    with pytest.raises(runner.Contradiction, match="terminal failed execution"):
        runner.run_pipeline(context, run_subprocess=never)
    assert never.commands == []
    assert _sealed_bytes(stage.canonical) == durable_result
    assert _sealed_bytes(stage.execution) == durable_execution


def test_only_dead_child_malformed_native_tmp_is_retryable(tmp_path: Path) -> None:
    context = fixtures.make_context(tmp_path)
    session, session_sha = fixtures.prepare_session(context)
    stage, _, _ = fixtures.prepare_launch(context, 1, session, session_sha)
    temporary = stage.staging.with_name(stage.staging.name + ".tmp")
    temporary.write_bytes(b'{"incomplete":')
    child = fixtures.FakeSubprocess(context)

    receipt = runner.run_pipeline(context, run_subprocess=child)
    assert receipt["status"] == "pass"
    assert len(child.commands) == 4
    assert not temporary.exists()
    assert runner.pair_state(stage.canonical) == "pair"


def test_malformed_staging_is_preserved_terminal_not_treated_as_native_tmp(
    tmp_path: Path,
) -> None:
    context = fixtures.make_context(tmp_path)
    session, session_sha = fixtures.prepare_session(context)
    stage, _, _ = fixtures.prepare_launch(context, 1, session, session_sha)
    stage.staging.write_bytes(b'{"incomplete":')
    never = fixtures.FakeSubprocess(context)

    with pytest.raises(runner.Contradiction, match="scientific_contract_invalid"):
        runner.run_pipeline(context, run_subprocess=never)
    assert never.commands == []
    assert not stage.staging.exists()
    assert _sealed_bytes(stage.canonical)[0] == b'{"incomplete":'
    execution = fixtures.read_sealed(stage.execution)
    assert execution["result"]["state"] == "contract_invalid"
    assert execution["execution_pass"] is False


@pytest.mark.parametrize("directory_name", ["run", "output", "staging"])
def test_live_owner_runner_atomic_temp_is_never_deleted_or_ignored(
    tmp_path: Path, directory_name: str
) -> None:
    context = fixtures.make_context(tmp_path)
    context.output_dir.mkdir()
    staging = context.output_dir / ".staging"
    staging.mkdir()
    directories = {
        "run": context.run_dir,
        "output": context.output_dir,
        "staging": staging,
    }
    owner_pid = 8117
    (context.proc_root / str(owner_pid)).mkdir()
    atomic = directories[directory_name] / (
        f".finalbench-atomic-target.json-{owner_pid}-{'d' * 32}.tmp"
    )
    atomic.write_bytes(b"live owner's unpublished bytes")
    child = fixtures.FakeSubprocess(context)

    with pytest.raises(runner.Contradiction, match="owner is still live"):
        runner.run_pipeline(context, run_subprocess=child)
    assert atomic.read_bytes() == b"live owner's unpublished bytes"
    assert child.commands == []
    assert runner.pair_state(runner.session_path(context)) == "absent"


SCHEMA_MUTATIONS = (
    ("session", (), "extra"),
    ("session", (), "missing:kind"),
    ("launch", (), "extra"),
    ("launch", (), "missing:kind"),
    ("execution", (), "extra"),
    ("execution", (), "missing:kind"),
    ("execution", ("process",), "extra"),
    ("execution", ("binding_check",), "extra"),
    ("execution", ("result",), "extra"),
    ("receipt", (), "extra"),
    ("receipt", (), "missing:kind"),
    ("receipt", ("session",), "extra"),
    ("receipt", ("outcome",), "extra"),
    ("receipt", ("results", "benchmark_1"), "extra"),
)


@pytest.mark.parametrize(
    ("document", "path", "mode"),
    SCHEMA_MUTATIONS,
    ids=[f"{document}-{'top' if not path else '.'.join(path)}-{mode}" for document, path, mode in SCHEMA_MUTATIONS],
)
def test_persistent_document_schemas_are_exact_at_every_bound_layer(
    tmp_path: Path, document: str, path: tuple[str, ...], mode: str
) -> None:
    context, document_path, _, payload_field = _document_fixture(tmp_path, document)
    value = copy.deepcopy(fixtures.read_sealed(document_path))
    _mutate_path(value, path, mode)
    _rehash_document(value, payload_field)
    fixtures.rewrite_sealed(document_path, value)

    child = fixtures.FakeSubprocess(context)
    with pytest.raises(runner.Contradiction, match="fields changed"):
        runner.run_pipeline(context, run_subprocess=child)
    assert child.commands == []


NONCANONICAL_TIMESTAMP_VARIANTS = ("zulu", "space", "short_offset", "non_utc")


@pytest.mark.parametrize("document", ("session", "launch", "execution", "receipt"))
@pytest.mark.parametrize("variant", NONCANONICAL_TIMESTAMP_VARIANTS)
def test_all_persistent_document_timestamps_reject_noncanonical_forms(
    tmp_path: Path, document: str, variant: str
) -> None:
    context, document_path, timestamp_field, payload_field = _document_fixture(
        tmp_path, document
    )
    value = copy.deepcopy(fixtures.read_sealed(document_path))
    original = str(value[timestamp_field])
    if variant == "zulu":
        replacement = original.replace("+00:00", "Z")
    elif variant == "space":
        replacement = original.replace("T", " ", 1)
    elif variant == "short_offset":
        replacement = original.replace("+00:00", "+00")
    else:
        parsed = datetime.fromisoformat(original)
        replacement = parsed.astimezone(
            timezone(timedelta(hours=1))
        ).isoformat()
    assert replacement != original
    value[timestamp_field] = replacement
    _rehash_document(value, payload_field)
    fixtures.rewrite_sealed(document_path, value)

    child = fixtures.FakeSubprocess(context)
    with pytest.raises(runner.Contradiction):
        runner.run_pipeline(context, run_subprocess=child)
    assert child.commands == []


def test_scientific_acceptance_misses_still_complete_all_four_executions(
    tmp_path: Path,
) -> None:
    context = fixtures.make_context(tmp_path)
    child = fixtures.FakeSubprocess(
        context,
        {
            "benchmark_1": {
                "overrides": {
                    "timing.pass": False,
                    "constraint_pass": False,
                }
            },
            "benchmark_2": {
                "overrides": {
                    "timing.mean_headroom_pass": False,
                    "constraint_pass": False,
                }
            },
        },
    )
    receipt = runner.run_pipeline(context, run_subprocess=child)

    assert [Path(command[command.index("--json-output") + 1]).stem for command in child.commands] == list(
        runner.STAGE_NAMES
    )
    assert receipt["status"] == "fail"
    assert receipt["outcome"] == {
        "status": "fail",
        "failed_stages": ["benchmark_1", "benchmark_2"],
        "failure_reason": "acceptance_gate_failed",
    }
    for stage in runner.stages(context):
        execution = fixtures.read_sealed(stage.execution)
        assert execution["execution_pass"] is True
        assert execution["result"]["state"] == "valid"
