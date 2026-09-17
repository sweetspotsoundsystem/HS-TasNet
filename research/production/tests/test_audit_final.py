from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


AUDITOR_PATH = Path(__file__).resolve().parents[1] / "audit_final.py"
SPEC = importlib.util.spec_from_file_location("hs_tasnet_c91_audit_final", AUDITOR_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def complete_service_state() -> dict[str, str]:
    return {
        name: "" for name in audit.SERVICE_PROPERTIES
    } | {
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
        "Result": "success",
        "ExecMainCode": "exited",
        "ExecMainStatus": "0",
        "MainPID": "0",
        "RemainAfterExit": "no",
        "InvocationID": "a" * 32,
        "ExecMainStartTimestamp": "Sat 2026-07-18 00:00:00 PDT",
        "ExecMainExitTimestamp": "Sat 2026-07-18 01:00:00 PDT",
        "CollectMode": "inactive",
    }


def terminal_report() -> dict[str, object]:
    return {
        "status": "complete",
        "step": 300_000,
        "total_steps": 300_000,
        "peak_vram_bytes": 5_000_000_000,
        "maximum_peak_vram_bytes": 6_700_000_000,
        "peak_vram_pass": True,
        "completed_at_utc": "2026-07-18T08:00:00+00:00",
    }


def final_lr_config() -> dict[str, object]:
    return {
        "model": {"learning_rate": 0.0003},
        "schedule": {
            "endpoint_semantics": "exclusive_optimizer_steps_0_through_total_steps_minus_1",
            "total_steps": 300_000,
            "lr_decay_start": 250_000,
            "lr_decay_end": 300_000,
        },
    }


def final_lr_payload(group_lr: float) -> dict[str, object]:
    return {
        "optimizer_meta": {
            "autoresearch_step": 300_000,
            "base_lrs": (0.0003,),
        },
        "optimizer": {
            "param_groups": [{"lr": group_lr}],
        },
    }


def final_lr_provenance(config: dict[str, object]) -> dict[str, object]:
    return {
        "contract_identity_sha256": audit.EXPECTED_CONTRACT_IDENTITY,
        "run_uuid": audit.EXPECTED_RUN_UUID,
        "trainer_sha256": audit.EXPECTED_TRAINER_SHA256,
        "config_file_sha256": audit.EXPECTED_CONFIG_SHA256,
        "experiment_sha256": audit.EXPECTED_EXPERIMENT_SHA256,
        "config": config,
    }


def final_lr_checkpoint(config: dict[str, object], group_lr: float) -> dict[str, object]:
    return {
        "step": 300_000,
        "path": "/tmp/step-000000300000-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.pt",
        "pointer_path": "/tmp/checkpoints/latest.json",
        "sha256": "b" * 64,
        "model_state_sha256": "c" * 64,
        "raw_unfinalized": True,
        "learning_rate_endpoint": audit.verify_checkpoint_optimizer_learning_rate(
            final_lr_payload(group_lr),
            config,
        ),
    }


def final_progress(*, learning_rate: float, loss: float = 0.02, timestamp_second: int = 1) -> dict[str, object]:
    return {
        "event": "train_progress",
        "time": f"2026-07-19T08:00:{timestamp_second:02d}+00:00",
        "step": 300_000,
        "total_steps": 300_000,
        "loss": loss,
        "loss_mean_recent": 0.021,
        "learning_rate": learning_rate,
        "steps_per_second": 2.1,
        "eta_hours": 0.0,
        "peak_vram_bytes": 5_600_000_000,
        "free_disk_gb": 600.0,
    }


def final_lr_events(
    checkpoint: dict[str, object],
    progress_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, progress in enumerate(progress_records):
        records.append(
            {
                "event": "run_opened",
                "run_uuid": audit.EXPECTED_RUN_UUID,
                "contract_identity_sha256": audit.EXPECTED_CONTRACT_IDENTITY,
            }
        )
        if index == 0:
            records.append({"event": "initial_checkpoint_verified", "step": 0})
        else:
            records.append({"event": "resumed", "step": 299_000})
        records.append(progress)
    records.extend(
        [
            {
                "event": "checkpoint_verified",
                "step": 300_000,
                "path": checkpoint["path"],
                "pointer": checkpoint["pointer_path"],
                "sha256": checkpoint["sha256"],
                "model_state_sha256": checkpoint["model_state_sha256"],
            },
            {
                "event": "run_complete",
                "step": 300_000,
                "contract_identity_sha256": audit.EXPECTED_CONTRACT_IDENTITY,
            },
        ]
    )
    return records


def adam_coverage_fixture(*, active_parameter_count: int = 21):
    import torch

    class CoverageModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.active = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.full((2,), float(index + 1))) for index in range(active_parameter_count)]
            )
            self.conv_decode = torch.nn.Module()
            self.conv_decode.register_parameter("bias", torch.nn.Parameter(torch.zeros(2)))

    model = CoverageModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    for name, parameter in model.named_parameters():
        if name == audit.EXPECTED_STATELESS_PARAMETER_NAME:
            continue
        optimizer.state[parameter] = {
            "step": torch.tensor(300_000.0, dtype=torch.float32),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.ones_like(parameter),
        }
    return torch, model, optimizer


def initialize_source_inventory_repo(directory: Path) -> Path:
    repo = directory / "source"
    package = repo / "hs_tasnet"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from hs_tasnet.trainer import Trainer, MusDB18HQ\n",
        encoding="utf-8",
    )
    (package / "trainer.py").write_text(
        "class Trainer: pass\nclass MusDB18HQ: pass\n",
        encoding="utf-8",
    )
    (package / "hs_tasnet.py").write_text("class HSTasNet: pass\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "hs_tasnet"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Audit Test",
            "-c",
            "user.email=audit@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return repo


def write_sealed_json(path: Path, value: dict[str, object]) -> str:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = audit.sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="ascii",
    )
    return digest


def step250_arming_fixture(
    directory: Path | None = None,
    manifest_hashes: dict[str, str] | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "outcome": {"name": "native_fast_failure", "sha256": "a" * 64},
        "binding": {"sha256": "b" * 64},
        "automation": {"sha256": "c" * 64},
        "preflight": {"sha256": "d" * 64},
    }
    if directory is not None and manifest_hashes is not None:
        value["automation"] = {
            "path": str(directory / "automation.sha256"),
            "sha256": manifest_hashes["automation"],
        }
        value["preflight"] = {
            "path": str(directory / "armer-preflight.sha256"),
            "sha256": manifest_hashes["preflight"],
        }
    return value


def install_step250_validator_stubs(directory: Path) -> dict[str, str]:
    filenames = (
        "arm_step250_after_step225.py",
        "run_step250000_gate.py",
        "recover_missing_snapshot_step250000.py",
        "compare_step250000.py",
        "crashsafe_step250000.py",
    )
    for filename in filenames:
        (directory / filename).write_text(f"# test stub: {filename}\n", encoding="utf-8")
    automation_names = filenames[1:]
    automation = directory / "automation.sha256"
    automation.write_text(
        "".join(f"{audit.sha256_file(directory / name)}  {name}\n" for name in automation_names),
        encoding="ascii",
    )
    preflight = directory / "armer-preflight.sha256"
    preflight_names = (filenames[0], "automation.sha256")
    preflight.write_text(
        "".join(f"{audit.sha256_file(directory / name)}  {name}\n" for name in preflight_names),
        encoding="ascii",
    )
    return {
        "automation": audit.sha256_file(automation),
        "preflight": audit.sha256_file(preflight),
    }


class AuditFinalTests(unittest.TestCase):
    def test_final_lr_endpoint_exact_expected_value_passes(self) -> None:
        config = final_lr_config()
        schedule = audit.derive_final_learning_rate_contract(config)
        self.assertEqual(schedule["last_applied_preupdate_counter"], 299_999)
        self.assertEqual(schedule["last_applied_learning_rate"], 2.960881373414992e-13)
        self.assertEqual(schedule["theoretical_unapplied_preupdate_counter"], 300_000)
        self.assertEqual(schedule["theoretical_unapplied_learning_rate"], 0.0)

        checkpoint = final_lr_checkpoint(config, 2.960881373414992e-13)
        events = final_lr_events(
            checkpoint,
            [final_progress(learning_rate=2.960881373414992e-13)],
        )
        result = audit.verify_final_learning_rate_endpoint(
            events,
            len(events) - 1,
            final_lr_provenance(config),
            checkpoint,
        )
        self.assertEqual(result["step_300000_progress_record_count"], 1)
        self.assertTrue(result["theoretical_zero_endpoint_remained_unapplied"])

    def test_final_progress_rejects_wrong_zero_lr(self) -> None:
        config = final_lr_config()
        checkpoint = final_lr_checkpoint(config, 2.960881373414992e-13)
        events = final_lr_events(checkpoint, [final_progress(learning_rate=0.0)])
        with self.assertRaises(audit.AuditFailure):
            audit.verify_final_learning_rate_endpoint(
                events,
                len(events) - 1,
                final_lr_provenance(config),
                checkpoint,
            )

    def test_final_checkpoint_rejects_wrong_param_group_lr(self) -> None:
        with self.assertRaises(audit.AuditFailure):
            audit.verify_checkpoint_optimizer_learning_rate(
                final_lr_payload(0.0),
                final_lr_config(),
            )

    def test_duplicate_final_progress_disagreement_is_rejected(self) -> None:
        config = final_lr_config()
        expected = 2.960881373414992e-13
        checkpoint = final_lr_checkpoint(config, expected)
        events = final_lr_events(
            checkpoint,
            [
                final_progress(learning_rate=expected, loss=0.02, timestamp_second=1),
                final_progress(learning_rate=expected, loss=0.03, timestamp_second=2),
            ],
        )
        with self.assertRaisesRegex(audit.AuditFailure, "duplicate.*disagree"):
            audit.verify_final_learning_rate_endpoint(
                events,
                len(events) - 1,
                final_lr_provenance(config),
                checkpoint,
            )

    def test_identical_duplicate_final_progress_is_resume_safe(self) -> None:
        config = final_lr_config()
        expected = 2.960881373414992e-13
        checkpoint = final_lr_checkpoint(config, expected)
        first = final_progress(learning_rate=expected, timestamp_second=1)
        second = final_progress(learning_rate=expected, timestamp_second=2)
        second["steps_per_second"] = 1.9
        second["free_disk_gb"] = 599.0
        events = final_lr_events(checkpoint, [first, second])
        result = audit.verify_final_learning_rate_endpoint(
            events,
            len(events) - 1,
            final_lr_provenance(config),
            checkpoint,
        )
        self.assertEqual(result["step_300000_progress_record_count"], 2)
        self.assertTrue(result["duplicate_progress_records_agree_exact"])

    def test_final_checkpoint_event_requires_bound_pointer(self) -> None:
        config = final_lr_config()
        expected = audit.EXPECTED_FINAL_APPLIED_LR
        checkpoint = final_lr_checkpoint(config, expected)
        events = final_lr_events(checkpoint, [final_progress(learning_rate=expected)])
        events[-2]["pointer"] = "/tmp/checkpoints/not-latest.json"
        with self.assertRaisesRegex(audit.AuditFailure, "pointer"):
            audit.verify_final_learning_rate_endpoint(
                events,
                len(events) - 1,
                final_lr_provenance(config),
                checkpoint,
            )

    def test_final_checkpoint_must_be_last_verified_checkpoint_event(self) -> None:
        config = final_lr_config()
        expected = audit.EXPECTED_FINAL_APPLIED_LR
        checkpoint = final_lr_checkpoint(config, expected)
        events = final_lr_events(checkpoint, [final_progress(learning_rate=expected)])
        events.insert(
            -1,
            {
                "event": "checkpoint_verified",
                "step": 299_000,
                "path": "/tmp/regressive.pt",
                "pointer": checkpoint["pointer_path"],
                "sha256": "d" * 64,
                "model_state_sha256": "e" * 64,
            },
        )
        with self.assertRaisesRegex(audit.AuditFailure, "last verified checkpoint"):
            audit.verify_final_learning_rate_endpoint(
                events,
                len(events) - 1,
                final_lr_provenance(config),
                checkpoint,
            )

    def test_loaded_adam_coverage_exact_22_owned_21_state_passes(self) -> None:
        torch, model, optimizer = adam_coverage_fixture()
        serialized = audit.verify_serialized_adam_topology(optimizer.state_dict())
        result = audit.verify_loaded_adam_coverage(model, optimizer, torch)
        self.assertEqual(result["grouped_model_parameter_count"], 22)
        self.assertEqual(result["adam_parameter_state_count"], 21)
        self.assertEqual(result["stateless_grouped_parameter_name"], "conv_decode.bias")
        self.assertEqual(
            serialized["stateless_grouped_parameter_index"],
            result["stateless_grouped_parameter_index"],
        )

    def test_loaded_adam_rejects_missing_nonbias_state(self) -> None:
        torch, model, optimizer = adam_coverage_fixture()
        parameter = next(parameter for name, parameter in model.named_parameters() if name != "conv_decode.bias")
        optimizer.state.pop(parameter)
        with self.assertRaises(audit.AuditFailure):
            audit.verify_loaded_adam_coverage(model, optimizer, torch)

    def test_loaded_adam_rejects_nan_moment(self) -> None:
        torch, model, optimizer = adam_coverage_fixture()
        parameter = next(parameter for name, parameter in model.named_parameters() if name != "conv_decode.bias")
        optimizer.state[parameter]["exp_avg"][0] = float("nan")
        with self.assertRaisesRegex(audit.AuditFailure, "non-finite"):
            audit.verify_loaded_adam_coverage(model, optimizer, torch)

    def test_loaded_adam_rejects_wrong_state_keys_and_shape(self) -> None:
        for mutation in ("keys", "shape"):
            with self.subTest(mutation=mutation):
                torch, model, optimizer = adam_coverage_fixture()
                parameter = next(
                    parameter for name, parameter in model.named_parameters() if name != "conv_decode.bias"
                )
                if mutation == "keys":
                    optimizer.state[parameter]["unexpected"] = torch.zeros_like(parameter)
                else:
                    optimizer.state[parameter]["exp_avg"] = torch.zeros(3, dtype=parameter.dtype)
                with self.assertRaises(audit.AuditFailure):
                    audit.verify_loaded_adam_coverage(model, optimizer, torch)

    def test_loaded_adam_rejects_wrong_parameter_count(self) -> None:
        torch, model, optimizer = adam_coverage_fixture(active_parameter_count=20)
        with self.assertRaisesRegex(audit.AuditFailure, "count"):
            audit.verify_loaded_adam_coverage(model, optimizer, torch)

    def test_loaded_adam_rejects_unexpected_bias_state_or_other_stateless_parameter(self) -> None:
        for mutation in ("bias_state", "other_stateless"):
            with self.subTest(mutation=mutation):
                torch, model, optimizer = adam_coverage_fixture()
                parameters = dict(model.named_parameters())
                bias = parameters["conv_decode.bias"]
                optimizer.state[bias] = {
                    "step": torch.tensor(300_000.0),
                    "exp_avg": torch.zeros_like(bias),
                    "exp_avg_sq": torch.ones_like(bias),
                }
                if mutation == "other_stateless":
                    optimizer.state.pop(parameters["active.0"])
                with self.assertRaises(audit.AuditFailure):
                    audit.verify_loaded_adam_coverage(model, optimizer, torch)

    def test_source_inventory_binds_all_package_python_and_trainer_import(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            repo = initialize_source_inventory_repo(Path(raw_directory))
            observations: dict[str, dict[str, object]] = {}
            result = audit.verify_source_worktree_inventory(repo, observations)
            self.assertTrue(result["tracked_worktree_status_clean"])
            self.assertTrue(result["package_init_imports_trainer"])
            self.assertIn("hs_tasnet/trainer.py", result["package_python_file_sha256"])
            self.assertEqual(result["package_python_file_count"], 3)

    def test_source_inventory_rejects_dirty_tracked_package_python(self) -> None:
        for mutation in ("modified", "staged", "deleted"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw_directory:
                repo = initialize_source_inventory_repo(Path(raw_directory))
                trainer = repo / "hs_tasnet/trainer.py"
                if mutation == "deleted":
                    trainer.unlink()
                else:
                    trainer.write_text("class Trainer: dirty\n", encoding="utf-8")
                    if mutation == "staged":
                        subprocess.run(["git", "-C", str(repo), "add", str(trainer)], check=True)
                with self.assertRaisesRegex(audit.AuditFailure, "staged, modified, or deleted"):
                    audit.verify_source_worktree_inventory(repo, {})

    def test_source_inventory_rejects_untracked_package_python(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            repo = initialize_source_inventory_repo(Path(raw_directory))
            (repo / "hs_tasnet/injected.py").write_text("raise RuntimeError('injected')\n", encoding="utf-8")
            with self.assertRaisesRegex(audit.AuditFailure, "inventory differs"):
                audit.verify_source_worktree_inventory(repo, {})

    def test_bound_user_units_select_safe_stop_and_terminal_owner(self) -> None:
        unit_dir = Path.home() / ".config/systemd/user"
        trainer = (unit_dir / "hs-tasnet-c91-full-v1-autoresume.service").read_text(encoding="utf-8")
        final_audit = (unit_dir / "hs-tasnet-c91-full-v1-final-audit.service").read_text(encoding="utf-8")
        self.assertIn("\nKillMode=mixed\n", trainer)
        self.assertIn("\nKillSignal=SIGTERM\n", trainer)
        self.assertIn("--service hs-tasnet-c91-full-v1-autoresume.service", final_audit)

    def test_active_service_success_defaults_are_incomplete(self) -> None:
        state = complete_service_state() | {
            "ActiveState": "active",
            "SubState": "running",
            "ExecMainCode": "0",
            "MainPID": "21578",
            "ExecMainExitTimestamp": "",
        }
        with self.assertRaises(audit.IncompleteRun):
            audit.validate_service_not_active(state, audit.DEFAULT_SERVICE)

    def test_loaded_service_requires_real_exit_identity(self) -> None:
        audit.validate_service_complete(complete_service_state(), audit.DEFAULT_SERVICE)
        for mutation in (
            {"LoadState": "not-found"},
            {"InvocationID": ""},
            {"ExecMainCode": "0"},
        ):
            with self.subTest(mutation=mutation), self.assertRaises(audit.IncompleteRun):
                audit.validate_service_complete(complete_service_state() | mutation, audit.DEFAULT_SERVICE)

    def test_loaded_service_accepts_numeric_cld_exited_and_proves_invocation(self) -> None:
        state = complete_service_state() | {"ExecMainCode": "1"}
        audit.validate_service_complete(state, audit.DEFAULT_SERVICE)
        journal_evidence = {
            "proof_source": "systemd_journal_after_transient_collection",
            "invocation_id": state["InvocationID"],
            "terminal_report_equal_exact": True,
        }
        with mock.patch.object(audit, "prove_collected_service_from_journal", return_value=journal_evidence):
            result = audit.prove_service_complete(state, audit.DEFAULT_SERVICE, Path("/tmp/run"), {})
        self.assertEqual(result["proof_source"], "systemctl_loaded_unit_and_bound_journal")
        self.assertEqual(result["systemctl"]["ExecMainCode"], "1")

    def test_loaded_service_rejects_other_or_nonstring_exit_code_representations(self) -> None:
        for value in ("0", "2", "01", "CLD_EXITED", "Exited", " exited", "1 ", "", 1, True, None, ["1"]):
            with self.subTest(value=value), self.assertRaises(audit.IncompleteRun):
                audit.validate_service_complete(complete_service_state() | {"ExecMainCode": value}, audit.DEFAULT_SERVICE)

    def test_failed_service_rejects_both_normal_exit_representations_with_zero_status(self) -> None:
        for value in ("exited", "1"):
            state = complete_service_state() | {
                "ActiveState": "failed",
                "SubState": "failed",
                "Result": "signal",
                "ExecMainCode": value,
                "ExecMainStatus": "0",
            }
            with self.subTest(value=value), self.assertRaises(audit.IncompleteRun):
                audit.validate_loaded_failed_state(state, audit.DEFAULT_SERVICE)

    def test_companion_selection_is_symmetric_and_fail_closed(self) -> None:
        inactive = {name: "" for name in audit.SERVICE_PROPERTIES} | {
            "LoadState": "not-found",
            "ActiveState": "inactive",
            "SubState": "dead",
        }
        states = {
            audit.PRIMARY_SERVICE: inactive,
            audit.AUTORESTART_SERVICE: inactive,
        }
        with mock.patch.object(audit, "read_service_state", side_effect=lambda service: states[service]):
            self.assertEqual(
                audit.validate_companion_trainer_inactive(audit.PRIMARY_SERVICE),
                inactive,
            )
            self.assertEqual(
                audit.validate_companion_trainer_inactive(audit.AUTORESTART_SERVICE),
                inactive,
            )
        active_primary = inactive | {"ActiveState": "active", "SubState": "running"}
        with mock.patch.object(audit, "read_service_state", return_value=active_primary):
            with self.assertRaises(audit.IncompleteRun):
                audit.validate_companion_trainer_inactive(audit.AUTORESTART_SERVICE)

    def test_loaded_service_requires_matching_bound_journal_invocation(self) -> None:
        state = complete_service_state()
        evidence = {
            "proof_source": "systemd_journal_after_transient_collection",
            "invocation_id": state["InvocationID"],
            "terminal_report_equal_exact": True,
        }
        with mock.patch.object(audit, "prove_collected_service_from_journal", return_value=evidence):
            result = audit.prove_service_complete(state, audit.DEFAULT_SERVICE, Path("/tmp/run"), {})
        self.assertEqual(result["proof_source"], "systemctl_loaded_unit_and_bound_journal")
        self.assertEqual(result["invocation_id"], state["InvocationID"])
        self.assertEqual(result["systemctl"], state)

        mismatched = evidence | {"invocation_id": "b" * 32}
        with mock.patch.object(audit, "prove_collected_service_from_journal", return_value=mismatched):
            with self.assertRaises(audit.IncompleteRun):
                audit.prove_service_complete(state, audit.DEFAULT_SERVICE, Path("/tmp/run"), {})

    def test_loaded_post_reboot_service_uses_durable_bound_journal(self) -> None:
        state = complete_service_state() | {
            "ExecMainCode": "0",
            "ExecMainStatus": "0",
            "InvocationID": "",
            "ExecMainStartTimestamp": "",
            "ExecMainExitTimestamp": "",
        }
        evidence = {
            "proof_source": "systemd_journal_after_transient_collection",
            "invocation_id": "b" * 32,
            "terminal_report_equal_exact": True,
        }
        with mock.patch.object(audit, "prove_collected_service_from_journal", return_value=evidence):
            result = audit.prove_service_complete(state, audit.DEFAULT_SERVICE, Path("/tmp/run"), {})
        self.assertEqual(result["proof_source"], "systemctl_loaded_post_reboot_and_bound_journal")
        self.assertEqual(result["invocation_id"], "b" * 32)
        self.assertTrue(result["systemctl_volatile_runtime_identity_reset"])
        self.assertEqual(result["systemctl"], state)

        ambiguous = state | {"ExecMainStartTimestamp": "stale-start"}
        with mock.patch.object(audit, "prove_collected_service_from_journal", return_value=evidence):
            with self.assertRaises(audit.IncompleteRun):
                audit.prove_service_complete(ambiguous, audit.DEFAULT_SERVICE, Path("/tmp/run"), {})

    def test_recovered_exception_is_hash_bound_and_replayed(self) -> None:
        anchor_hash = "a" * 64
        continuation_hash = "b" * 64
        events: list[dict[str, object]] = [
            {
                "event": "checkpoint_verified",
                "step": 135_000,
                "model_state_sha256": anchor_hash,
            },
            {"event": "train_progress", "step": 135_025},
            {"event": "training_exception", "step": 135_045},
            {"event": "inventory_verified"},
            {"event": "run_opened"},
            {
                "event": "resumed",
                "step": 135_000,
                "model_state_sha256": anchor_hash,
            },
            {"event": "train_progress", "step": 135_025},
            {
                "event": "checkpoint_verified",
                "step": 136_000,
                "model_state_sha256": continuation_hash,
            },
            {"event": "run_complete", "step": 300_000},
        ]
        result = audit.validate_recovery_lineage(events, len(events) - 1)
        self.assertTrue(result["all_interruptions_recovered"])
        self.assertEqual(result["resume_count"], 1)
        self.assertEqual(result["prior_training_exception_count"], 1)
        recovery = result["training_exception_recoveries"][0]
        self.assertEqual(recovery["rollback_steps"], 45)
        self.assertEqual(recovery["continuation_step"], 136_000)

        wrong_hash = [dict(record) for record in events]
        wrong_hash[5]["model_state_sha256"] = "c" * 64
        with self.assertRaises(audit.AuditFailure):
            audit.validate_recovery_lineage(wrong_hash, len(wrong_hash) - 1)

        progressed_before_resume = [dict(record) for record in events]
        progressed_before_resume.insert(5, {"event": "train_progress", "step": 135_050})
        with self.assertRaises(audit.AuditFailure):
            audit.validate_recovery_lineage(progressed_before_resume, len(progressed_before_resume) - 1)

        unresolved = [dict(record) for record in events[:5]] + [{"event": "run_complete", "step": 300_000}]
        with self.assertRaises(audit.AuditFailure):
            audit.validate_recovery_lineage(unresolved, len(unresolved) - 1)

    def test_step_300000_resume_can_replay_finalization_without_an_optimizer_step(self) -> None:
        model_hash = "d" * 64
        events = [
            {
                "event": "checkpoint_verified",
                "step": 300_000,
                "model_state_sha256": model_hash,
            },
            {"event": "training_exception", "step": 300_000},
            {"event": "inventory_verified"},
            {"event": "run_opened"},
            {
                "event": "resumed",
                "step": 300_000,
                "model_state_sha256": model_hash,
            },
            {"event": "run_complete", "step": 300_000},
        ]
        result = audit.validate_recovery_lineage(events, len(events) - 1)
        self.assertEqual(result["resumes"][0]["continuation_step"], 300_000)
        self.assertEqual(result["training_exception_recoveries"][0]["rollback_steps"], 0)

    def test_preflight_requires_exact_terminal_agreement(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            run_dir = Path(raw_directory)
            report = terminal_report()
            contract = {"static": {"config": {"safety": {"maximum_peak_vram_gb": 6.7}}}}
            (run_dir / "run_contract.json").write_text(json.dumps(contract) + "\n", encoding="utf-8")
            report_bytes = json.dumps(report, indent=2, sort_keys=True).encode() + b"\n"
            (run_dir / "final_report.json").write_bytes(report_bytes)
            (run_dir / "status.json").write_bytes(report_bytes)
            event = {"event": "run_complete", "time": "2026-07-18T08:00:01+00:00", **report}
            (run_dir / "events.jsonl").write_text(json.dumps(event, sort_keys=True) + "\n", encoding="utf-8")

            result = audit.preflight_completion(run_dir)
            self.assertEqual(result["completion_event_count"], 1)

            # Semantic equality is insufficient; the terminal files must also
            # be byte-identical.
            (run_dir / "status.json").write_text(json.dumps(report) + "\n", encoding="utf-8")
            with self.assertRaises(audit.AuditFailure):
                audit.preflight_completion(run_dir)

            (run_dir / "status.json").write_bytes(report_bytes)
            with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            with self.assertRaises(audit.AuditFailure):
                audit.preflight_completion(run_dir)

    def test_strict_json_rejects_duplicates_and_nonfinite_values(self) -> None:
        with self.assertRaises(audit.AuditFailure):
            audit.parse_strict_json('{"step": 1, "step": 2}', "duplicate-test")
        with self.assertRaises(audit.AuditFailure):
            audit.parse_strict_json('{"loss": NaN}', "nan-test")

    def test_collected_service_requires_bound_terminal_journal(self) -> None:
        service = "test-c91.service"
        invocation = "b" * 32
        report = terminal_report()
        run_dir = Path("/tmp/test-c91-run")
        manifest_hash = "c" * 64
        command = [
            "/home/axel/miniforge3/bin/python",
            str(audit.BASE_DIR / "train_production.py"),
            "--config",
            str(audit.BASE_DIR / "full_config.json"),
            "--manifest",
            str(audit.BASE_DIR / "manifests/combined.manifest.json"),
            "--manifest-sha256",
            manifest_hash,
            "--run-dir",
            str(run_dir),
            "--resume",
            "auto",
            "--device",
            "cuda",
        ]
        records: list[dict[str, object]] = [
            {
                "MESSAGE": f"Started {service} - test trainer.",
                "USER_UNIT": service,
                "USER_INVOCATION_ID": invocation,
                "__CURSOR": "start-cursor",
            }
        ]
        for index, message in enumerate(json.dumps(report, indent=2, sort_keys=True).splitlines()):
            records.append(
                {
                    "MESSAGE": message,
                    "_TRANSPORT": "stdout",
                    "_SYSTEMD_USER_UNIT": service,
                    "_SYSTEMD_INVOCATION_ID": invocation,
                    "_CMDLINE": " ".join(command),
                    "__REALTIME_TIMESTAMP": str(1_800_000_000_000_000 + index),
                    "__CURSOR": f"stdout-{index}",
                }
            )
        records.append(
            {
                "MESSAGE": f"{service}: Consumed 1h CPU time, 1G memory peak.",
                "MESSAGE_ID": audit.RESOURCE_MESSAGE_ID,
                "USER_UNIT": service,
                "USER_INVOCATION_ID": invocation,
                "__CURSOR": "resource-cursor",
            }
        )
        preflight = {
            "report": report,
            "contract": {"static": {"manifest_file_sha256": manifest_hash}},
            "completion_event": {"time": "2026-07-18T08:00:01+00:00"},
        }
        state = {name: "" for name in audit.SERVICE_PROPERTIES} | {
            "LoadState": "not-found",
            "ActiveState": "inactive",
        }
        with mock.patch.object(audit, "read_service_journal", return_value=records):
            evidence = audit.prove_service_complete(state, service, run_dir, preflight)
        self.assertEqual(evidence["invocation_id"], invocation)
        self.assertTrue(evidence["terminal_report_equal_exact"])

        failing_records = records + [
            {
                "MESSAGE": f"{service}: Main process exited, code=exited, status=1/FAILURE",
                "MESSAGE_ID": "98e322203f7a4ed290d09fe03c09fe15",
                "USER_UNIT": service,
                "USER_INVOCATION_ID": invocation,
                "__CURSOR": "failure-cursor",
            }
        ]
        with mock.patch.object(audit, "read_service_journal", return_value=failing_records):
            with self.assertRaises(audit.AuditFailure):
                audit.prove_service_complete(state, service, run_dir, preflight)

        with mock.patch.object(audit, "read_service_journal", return_value=[]):
            with self.assertRaises(audit.IncompleteRun):
                audit.prove_service_complete(state, service, run_dir, preflight)

    def test_postpublication_service_failure_is_durably_bound(self) -> None:
        service = audit.DEFAULT_SERVICE
        invocation = "d" * 32
        run_dir = Path("/tmp/postpublication-c91-run")
        manifest_hash = "e" * 64
        report = terminal_report()
        progress = final_progress(
            learning_rate=audit.EXPECTED_FINAL_APPLIED_LR,
            timestamp_second=0,
        )
        completion = {"event": "run_complete", "time": "2026-07-19T08:00:01+00:00", **report}
        command = [
            "/home/axel/miniforge3/bin/python",
            str(audit.BASE_DIR / "train_production.py"),
            "--config",
            str(audit.BASE_DIR / "full_config.json"),
            "--manifest",
            str(audit.BASE_DIR / "manifests/combined.manifest.json"),
            "--manifest-sha256",
            manifest_hash,
            "--run-dir",
            str(run_dir),
            "--resume",
            "auto",
            "--device",
            "cuda",
        ]
        start_us = int(audit.dt.datetime.fromisoformat("2026-07-19T07:59:59+00:00").timestamp() * 1_000_000)
        progress_us = int(audit.dt.datetime.fromisoformat("2026-07-19T08:00:00+00:00").timestamp() * 1_000_000)
        failure_us = int(audit.dt.datetime.fromisoformat("2026-07-19T08:00:02+00:00").timestamp() * 1_000_000)
        records: list[dict[str, object]] = [
            {
                "MESSAGE": f"Started {service} - production trainer.",
                "USER_UNIT": service,
                "USER_INVOCATION_ID": invocation,
                "__REALTIME_TIMESTAMP": str(start_us),
                "__CURSOR": "post-start",
            },
            {
                "MESSAGE": json.dumps(progress, sort_keys=True),
                "_TRANSPORT": "stdout",
                "_SYSTEMD_USER_UNIT": service,
                "_SYSTEMD_INVOCATION_ID": invocation,
                "_CMDLINE": " ".join(command),
                "__REALTIME_TIMESTAMP": str(progress_us),
                "__CURSOR": "post-progress",
            },
            {
                "MESSAGE": "Main process exited after durable completion",
                "MESSAGE_ID": audit.PROCESS_EXIT_MESSAGE_ID,
                "USER_UNIT": service,
                "USER_INVOCATION_ID": invocation,
                "__REALTIME_TIMESTAMP": str(failure_us),
                "__CURSOR": "post-failure",
            },
            {
                "MESSAGE": "Consumed resources",
                "MESSAGE_ID": audit.RESOURCE_MESSAGE_ID,
                "USER_UNIT": service,
                "USER_INVOCATION_ID": invocation,
                "__REALTIME_TIMESTAMP": str(failure_us + 1),
                "__CURSOR": "post-resource",
            },
        ]
        preflight = {
            "report": report,
            "contract": {"static": {"manifest_file_sha256": manifest_hash}},
            "events": [progress, completion],
            "completion_event": completion,
            "completion_event_index": 1,
        }
        failed_state = complete_service_state() | {
            "ActiveState": "failed",
            "SubState": "failed",
            "Result": "signal",
            "ExecMainCode": "killed",
            "ExecMainStatus": "9",
            "InvocationID": invocation,
        }
        with mock.patch.object(audit, "read_service_journal", return_value=records):
            evidence = audit.prove_service_complete(failed_state, service, run_dir, preflight)
        self.assertEqual(evidence["proof_source"], "systemctl_loaded_failed_after_durable_completion")
        self.assertTrue(evidence["postpublication_failure_accepted"])
        self.assertEqual(evidence["invocation_id"], invocation)

        later_start = records + [
            {
                "MESSAGE": f"Started {service} - later replay.",
                "USER_UNIT": service,
                "USER_INVOCATION_ID": "f" * 32,
                "__REALTIME_TIMESTAMP": str(failure_us + 2),
            }
        ]
        with mock.patch.object(audit, "read_service_journal", return_value=later_start):
            with self.assertRaises(audit.AuditFailure):
                audit.prove_service_complete(failed_state, service, run_dir, preflight)

        early_failure = [dict(record) for record in records]
        early_failure[2]["__REALTIME_TIMESTAMP"] = str(progress_us)
        with mock.patch.object(audit, "read_service_journal", return_value=early_failure):
            with self.assertRaises(audit.IncompleteRun):
                audit.prove_service_complete(failed_state, service, run_dir, preflight)

    def test_step250_terminal_absence_retries_but_partial_or_contradictory_pairs_fail(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            with self.assertRaises(audit.IncompleteRun):
                audit.verify_step250_diagnostic_terminal(directory)

            comparison = directory / "comparison.json"
            comparison.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(audit.AuditFailure):
                audit.verify_step250_diagnostic_terminal(directory)

        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            orphan = directory / "frozen-fast-gate-failure.json.sha256"
            orphan.write_text(f"{'0' * 64}  frozen-fast-gate-failure.json\n", encoding="ascii")
            with self.assertRaises(audit.AuditFailure):
                audit.verify_step250_diagnostic_terminal(directory)

        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            write_sealed_json(directory / "comparison.json", {"status": "ok"})
            write_sealed_json(directory / "frozen-fast-gate-failure.json", {"status": "failed_closed"})
            with self.assertRaises(audit.AuditFailure):
                audit.verify_step250_diagnostic_terminal(directory)

    def test_step250_terminal_without_arming_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            write_sealed_json(directory / "frozen-fast-gate-failure.json", {"status": "failed_closed"})
            with self.assertRaises(audit.AuditFailure):
                audit.verify_step250_diagnostic_terminal(directory)

    def test_step250_each_canonical_terminal_uses_its_strong_validator(self) -> None:
        cases = (
            ("native_frozen_batched_comparison", "comparison.json", "validate_comparison_marker"),
            ("native_fast_failure", "frozen-fast-gate-failure.json", "validate_failure_receipt"),
            ("missing_native_snapshot_recovery", "recovery-terminal.json", "verify_committed_recovery"),
        )
        for outcome_name, filename, expected_callback in cases:
            with self.subTest(outcome=outcome_name), tempfile.TemporaryDirectory() as raw_directory:
                directory = Path(raw_directory)
                manifest_hashes = install_step250_validator_stubs(directory)
                arming = step250_arming_fixture(directory, manifest_hashes)
                arming_sha256 = write_sealed_json(directory / "arming-terminal.json", arming)
                outcome = {"kind": outcome_name, "step": 250_000}
                outcome_sha256 = write_sealed_json(directory / filename, outcome)
                if outcome_name == "native_frozen_batched_comparison":
                    for support in (
                        "step250000-full14.json",
                        "audit-receipt.json",
                        "lr-boundary-audit-receipt.json",
                        "step225-reference-binding.json",
                    ):
                        write_sealed_json(directory / support, {"support": support})

                armer_validator = mock.Mock()
                comparison_validator = mock.Mock()
                failure_validator = mock.Mock()
                recovery_validator = mock.Mock(return_value=outcome)
                modules = {
                    "arm_step250_after_step225.py": types.SimpleNamespace(validate_terminal=armer_validator),
                    "run_step250000_gate.py": types.SimpleNamespace(
                        validate_comparison_marker=comparison_validator,
                        validate_failure_receipt=failure_validator,
                    ),
                    "recover_missing_snapshot_step250000.py": types.SimpleNamespace(
                        verify_committed_recovery=recovery_validator,
                    ),
                }

                def fake_loader(path: Path, _label: str):
                    return modules[path.name], audit.sha256_file(path)

                with mock.patch.object(audit, "load_step250_validator_module", side_effect=fake_loader):
                    evidence = audit.verify_step250_diagnostic_terminal(directory, directory)

                self.assertEqual(evidence["status"], "validated")
                self.assertEqual(evidence["arming_terminal"]["sha256"], arming_sha256)
                self.assertEqual(evidence["canonical_outcome"]["name"], outcome_name)
                self.assertEqual(evidence["canonical_outcome"]["sha256"], outcome_sha256)
                self.assertTrue(evidence["canonical_outcome"]["exactly_one_sealed"])
                self.assertEqual(
                    evidence["validator_manifests"]["automation"]["sha256"],
                    manifest_hashes["automation"],
                )
                self.assertEqual(
                    evidence["validator_manifests"]["armer_preflight"]["sha256"],
                    manifest_hashes["preflight"],
                )
                armer_validator.assert_called_once_with(arming)
                callbacks = {
                    "validate_comparison_marker": comparison_validator,
                    "validate_failure_receipt": failure_validator,
                    "verify_committed_recovery": recovery_validator,
                }
                callbacks[expected_callback].assert_called_once()

    def test_step250_strong_validator_rejection_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            manifest_hashes = install_step250_validator_stubs(directory)
            write_sealed_json(
                directory / "arming-terminal.json",
                step250_arming_fixture(directory, manifest_hashes),
            )
            write_sealed_json(
                directory / "frozen-fast-gate-failure.json",
                {"kind": "native_fast_failure", "step": 250_000},
            )
            armer = types.SimpleNamespace(validate_terminal=mock.Mock())
            gate = types.SimpleNamespace(
                validate_failure_receipt=mock.Mock(side_effect=RuntimeError("tampered failure")),
            )

            def fake_loader(path: Path, _label: str):
                module = armer if path.name == "arm_step250_after_step225.py" else gate
                return module, audit.sha256_file(path)

            with mock.patch.object(audit, "load_step250_validator_module", side_effect=fake_loader):
                with self.assertRaisesRegex(audit.AuditFailure, "tampered failure"):
                    audit.verify_step250_diagnostic_terminal(directory, directory)

    def test_step250_validator_sys_path_mutation_is_exactly_restored_on_success(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            manifest_hashes = install_step250_validator_stubs(directory)
            write_sealed_json(
                directory / "arming-terminal.json",
                step250_arming_fixture(directory, manifest_hashes),
            )
            outcome = {"kind": "native_fast_failure", "step": 250_000}
            write_sealed_json(directory / "frozen-fast-gate-failure.json", outcome)
            armer_validator = mock.Mock()
            promoted_path = str(directory / "promoted-by-validator")
            injected_path = str(directory / "added-by-import")

            def mutate_import_path(_value: dict[str, object]) -> None:
                sys.path.insert(0, promoted_path)
                sys.path.append(injected_path)

            gate_validator = mock.Mock(side_effect=mutate_import_path)
            modules = {
                "arm_step250_after_step225.py": types.SimpleNamespace(
                    validate_terminal=armer_validator,
                ),
                "run_step250000_gate.py": types.SimpleNamespace(
                    validate_failure_receipt=gate_validator,
                ),
            }

            def fake_loader(path: Path, _label: str):
                return modules[path.name], audit.sha256_file(path)

            original_path_object = sys.path
            original_entries = list(sys.path)
            try:
                with mock.patch.object(audit, "load_step250_validator_module", side_effect=fake_loader):
                    evidence = audit.verify_step250_diagnostic_terminal(directory, directory)
                self.assertEqual(evidence["status"], "validated")
                self.assertIs(sys.path, original_path_object)
                self.assertEqual(sys.path, original_entries)
            finally:
                sys.path = original_path_object
                original_path_object[:] = original_entries
            armer_validator.assert_called_once()
            gate_validator.assert_called_once_with(outcome)

    def test_step250_validator_sys_path_mutation_is_exactly_restored_on_exception(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            manifest_hashes = install_step250_validator_stubs(directory)
            write_sealed_json(
                directory / "arming-terminal.json",
                step250_arming_fixture(directory, manifest_hashes),
            )
            write_sealed_json(
                directory / "frozen-fast-gate-failure.json",
                {"kind": "native_fast_failure", "step": 250_000},
            )
            promoted_path = str(directory / "promoted-before-error")
            injected_path = str(directory / "added-before-error")

            def mutate_then_reject(_value: dict[str, object]) -> None:
                sys.path.insert(0, promoted_path)
                sys.path.append(injected_path)
                raise RuntimeError("validator rejected after path mutation")

            modules = {
                "arm_step250_after_step225.py": types.SimpleNamespace(
                    validate_terminal=mock.Mock(),
                ),
                "run_step250000_gate.py": types.SimpleNamespace(
                    validate_failure_receipt=mutate_then_reject,
                ),
            }

            def fake_loader(path: Path, _label: str):
                return modules[path.name], audit.sha256_file(path)

            original_path_object = sys.path
            original_entries = list(sys.path)
            try:
                with mock.patch.object(audit, "load_step250_validator_module", side_effect=fake_loader):
                    with self.assertRaisesRegex(
                        audit.AuditFailure,
                        "validator rejected after path mutation",
                    ):
                        audit.verify_step250_diagnostic_terminal(directory, directory)
                self.assertIs(sys.path, original_path_object)
                self.assertEqual(sys.path, original_entries)
            finally:
                sys.path = original_path_object
                original_path_object[:] = original_entries

    def test_step250_validator_sys_path_object_replacement_fails_closed_and_restores(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            manifest_hashes = install_step250_validator_stubs(directory)
            write_sealed_json(
                directory / "arming-terminal.json",
                step250_arming_fixture(directory, manifest_hashes),
            )
            write_sealed_json(
                directory / "frozen-fast-gate-failure.json",
                {"kind": "native_fast_failure", "step": 250_000},
            )

            def replace_import_path(_value: dict[str, object]) -> None:
                sys.path = [str(directory / "replacement-object")]

            modules = {
                "arm_step250_after_step225.py": types.SimpleNamespace(
                    validate_terminal=mock.Mock(),
                ),
                "run_step250000_gate.py": types.SimpleNamespace(
                    validate_failure_receipt=replace_import_path,
                ),
            }

            def fake_loader(path: Path, _label: str):
                return modules[path.name], audit.sha256_file(path)

            original_path_object = sys.path
            original_entries = list(sys.path)
            try:
                with mock.patch.object(audit, "load_step250_validator_module", side_effect=fake_loader):
                    with self.assertRaisesRegex(
                        audit.AuditFailure,
                        "replaced the sys.path object",
                    ):
                        audit.verify_step250_diagnostic_terminal(directory, directory)
                self.assertIs(sys.path, original_path_object)
                self.assertEqual(sys.path, original_entries)
            finally:
                sys.path = original_path_object
                original_path_object[:] = original_entries

    def test_step250_validator_manifest_is_verified_before_dynamic_import(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            manifest_hashes = install_step250_validator_stubs(directory)
            write_sealed_json(
                directory / "arming-terminal.json",
                step250_arming_fixture(directory, manifest_hashes),
            )
            write_sealed_json(
                directory / "frozen-fast-gate-failure.json",
                {"kind": "native_fast_failure", "step": 250_000},
            )
            (directory / "run_step250000_gate.py").write_text("# tampered after sealing\n", encoding="utf-8")
            with mock.patch.object(audit, "load_step250_validator_module") as loader:
                with self.assertRaisesRegex(audit.AuditFailure, "checksum mismatch"):
                    audit.verify_step250_diagnostic_terminal(directory, directory)
            loader.assert_not_called()

    def test_step250_evidence_disappearance_or_change_after_validation_fails_closed(self) -> None:
        initial = {"status": "validated", "canonical_outcome": {"sha256": "a" * 64}}
        with mock.patch.object(audit, "verify_step250_diagnostic_terminal", return_value=initial):
            audit.revalidate_step250_diagnostic_terminal(initial)
        with mock.patch.object(
            audit,
            "verify_step250_diagnostic_terminal",
            side_effect=audit.IncompleteRun("gone"),
        ):
            with self.assertRaises(audit.AuditFailure):
                audit.revalidate_step250_diagnostic_terminal(initial)
        changed = {"status": "validated", "canonical_outcome": {"sha256": "b" * 64}}
        with mock.patch.object(audit, "verify_step250_diagnostic_terminal", return_value=changed):
            with self.assertRaises(audit.AuditFailure):
                audit.revalidate_step250_diagnostic_terminal(initial)

    def test_final_receipt_records_step250_prerequisite(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            paths = {
                name: directory / filename
                for name, filename in {
                    "report": "final_report.json",
                    "status": "status.json",
                    "events": "events.jsonl",
                }.items()
            }
            observations = {
                str(path.resolve()): {"sha256": character * 64}
                for path, character in zip(paths.values(), "abc", strict=True)
            }
            preflight = {
                "report": terminal_report(),
                "paths": paths,
                "completion_event_index": 10,
                "completion_event_count": 1,
                "prior_training_exception_count": 0,
                "recovery_lineage": {"validated": True},
            }
            step250 = {
                "status": "validated",
                "step": 250_000,
                "arming_terminal": {"sha256": "d" * 64},
                "canonical_outcome": {
                    "name": "native_fast_failure",
                    "sha256": "e" * 64,
                },
            }
            torch = types.SimpleNamespace(
                __version__="test",
                cuda=types.SimpleNamespace(is_available=lambda: False, is_initialized=lambda: False),
            )
            receipt = audit.build_receipt(
                directory,
                audit.DEFAULT_SERVICE,
                {},
                None,
                preflight,
                {},
                {},
                {},
                {},
                {},
                step250,
                observations,
                torch,
            )
            self.assertEqual(receipt["step250000_diagnostic_prerequisite"], step250)

    def test_receipt_publication_is_new_only_and_temp_scan_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            receipt = directory / audit.RECEIPT_NAME
            first = {"audit_status": "pass", "value": 1}
            audit.atomic_write_new_json(receipt, first)
            original = receipt.read_bytes()
            with self.assertRaises(audit.AuditFailure):
                audit.atomic_write_new_json(receipt, {"audit_status": "pass", "value": 2})
            self.assertEqual(receipt.read_bytes(), original)
            self.assertEqual(audit.find_temporary_files(directory), [])
            (directory / ".artifact.pt.123.tmp").write_bytes(b"partial")
            self.assertEqual(audit.find_temporary_files(directory), [".artifact.pt.123.tmp"])


if __name__ == "__main__":
    unittest.main()
