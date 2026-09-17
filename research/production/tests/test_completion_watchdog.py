from __future__ import annotations

import json
import fcntl
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

WATCHDOG_PATH = Path(__file__).resolve().parents[1] / "completion_watchdog.py"
SPEC = importlib.util.spec_from_file_location("hs_tasnet_c91_completion_watchdog", WATCHDOG_PATH)
assert SPEC is not None and SPEC.loader is not None
watchdog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watchdog)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_artifact(path: Path, content: bytes = b"artifact") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    digest = watchdog.sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return digest


def make_complete_run(run_dir: Path) -> dict:
    raw_hash = write_artifact(run_dir / "final-training-raw.pt", b"raw")
    deploy_hash = write_artifact(run_dir / "final-deployment.pt", b"deploy")
    checkpoint_hash = write_artifact(run_dir / "checkpoints/final.pt", b"checkpoint")
    raw_state = "a" * 64
    deploy_state = "b" * 64
    report = {
        "status": "complete",
        "step": 300_000,
        "total_steps": 300_000,
        "peak_vram_bytes": 10,
        "maximum_peak_vram_bytes": 20,
        "peak_vram_pass": True,
        "contract_identity_sha256": watchdog.EXPECTED_CONTRACT,
        "raw_artifact": {
            "path": str(run_dir / "final-training-raw.pt"),
            "sha256": raw_hash,
            "model_state_sha256": raw_state,
        },
        "deployment": {
            "path": str(run_dir / "final-deployment.pt"),
            "sha256": deploy_hash,
            "model_state_sha256": deploy_state,
        },
    }
    write_json(run_dir / "final-deployment.pt.json", report["deployment"])
    write_json(run_dir / "checkpoints/latest.json", {
        "kind": "hs_tasnet_c91_checkpoint_pointer",
        "step": 300_000,
        "generation": "final.pt",
        "path": "checkpoints/final.pt",
        "sha256": checkpoint_hash,
        "model_state_sha256": raw_state,
    })
    write_json(run_dir / "final_report.json", report)
    (run_dir / "status.json").write_bytes((run_dir / "final_report.json").read_bytes())
    event = {"event": "run_complete", "time": "now", **report}
    (run_dir / "events.jsonl").write_text(json.dumps(event, sort_keys=True) + "\n", encoding="utf-8")
    return report


def make_passed_audit_receipt(run_dir: Path) -> dict:
    report = json.loads((run_dir / "final_report.json").read_text(encoding="utf-8"))
    pointer = json.loads((run_dir / "checkpoints/latest.json").read_text(encoding="utf-8"))
    receipt = {
        "schema_version": 1,
        "kind": "hs_tasnet_c91_final_audit_receipt",
        "audit_status": "pass",
        "audited_at_utc": "2026-01-01T00:00:00+00:00",
        "auditor": {
            "cuda_visible_devices": "",
            "torch_cuda_available": False,
            "torch_cuda_initialized": False,
        },
        "service": {"name": watchdog.AUTORESTART_SERVICE},
        "run": {"contract_identity_sha256": watchdog.EXPECTED_CONTRACT},
        "completion": {
            "status": "complete",
            "step": 300_000,
            "total_steps": 300_000,
            "final_report_sha256": watchdog.sha256_file(run_dir / "final_report.json"),
            "status_sha256": watchdog.sha256_file(run_dir / "status.json"),
            "events_sha256": watchdog.sha256_file(run_dir / "events.jsonl"),
            "peak_vram_pass": True,
            "peak_vram_bytes": report["peak_vram_bytes"],
            "maximum_peak_vram_bytes": report["maximum_peak_vram_bytes"],
        },
        "artifacts": {
            "final_checkpoint": {
                "sha256": pointer["sha256"],
                "model_state_sha256": pointer["model_state_sha256"],
            },
            "final_training_raw": {
                "sha256": report["raw_artifact"]["sha256"],
                "model_state_sha256": report["raw_artifact"]["model_state_sha256"],
            },
            "final_deployment": {
                "sha256": report["deployment"]["sha256"],
                "model_state_sha256": report["deployment"]["model_state_sha256"],
            },
        },
        "step250000_diagnostic_prerequisite": {
            "status": "validated",
            "step": 250_000,
            "canonical_outcome": {"exactly_one_sealed": True, "sha256": "d" * 64},
        },
        "functional_verification": {
            "transformation": {
                "post_optimizer_only": True,
                "decoder_weight_equals_raw_times_hann_exact": True,
            },
            "streaming": {
                "device": "cpu",
                "finite": True,
                "reset_determinism_exact": True,
            },
        },
        "filesystem": {"temporary_files": []},
    }
    receipt["audit_payload_sha256"] = watchdog.canonical_sha256(receipt)
    write_json(run_dir / "final_audit_receipt.json", receipt)
    return receipt


class CompletionWatchdogTests(unittest.TestCase):
    def test_empty_run_is_recoverable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = watchdog.publication_state(Path(directory))
            self.assertEqual(state["status"], "incomplete_recoverable")

    def test_report_without_completion_event_is_recoverable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            write_json(run_dir / "final_report.json", {"status": "complete"})
            state = watchdog.publication_state(run_dir)
            self.assertEqual(state["status"], "incomplete_recoverable")
            self.assertTrue(state["final_report_present"])

    def test_complete_publication_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            make_complete_run(run_dir)
            state = watchdog.publication_state(run_dir)
            self.assertEqual(state["status"], "publication_complete")

    def test_completion_event_without_status_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            make_complete_run(run_dir)
            (run_dir / "status.json").unlink()
            with self.assertRaises(watchdog.PublicationContradiction):
                watchdog.publication_state(run_dir)

    def test_duplicate_completion_event_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            make_complete_run(run_dir)
            line = (run_dir / "events.jsonl").read_text(encoding="utf-8")
            (run_dir / "events.jsonl").write_text(line + line, encoding="utf-8")
            with self.assertRaisesRegex(watchdog.PublicationContradiction, "duplicate"):
                watchdog.publication_state(run_dir)

    def test_event_after_completion_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            make_complete_run(run_dir)
            with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write('{"event":"later"}\n')
            with self.assertRaisesRegex(watchdog.PublicationContradiction, "not the final"):
                watchdog.publication_state(run_dir)

    def test_artifact_corruption_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            make_complete_run(run_dir)
            (run_dir / "final-deployment.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(watchdog.PublicationContradiction, "hash mismatch"):
                watchdog.publication_state(run_dir)

    def test_passed_audit_receipt_is_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            make_complete_run(run_dir)
            make_passed_audit_receipt(run_dir)
            self.assertEqual(watchdog.publication_state(run_dir)["status"], "audited_complete")

    def test_audit_receipt_hash_and_underlying_publication_are_revalidated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            make_complete_run(run_dir)
            receipt = make_passed_audit_receipt(run_dir)
            receipt["audit_status"] = "changed-after-hash"
            write_json(run_dir / "final_audit_receipt.json", receipt)
            with self.assertRaisesRegex(watchdog.PublicationContradiction, "did not pass"):
                watchdog.publication_state(run_dir)

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            make_complete_run(run_dir)
            make_passed_audit_receipt(run_dir)
            (run_dir / "status.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(watchdog.PublicationContradiction):
                watchdog.publication_state(run_dir)

    def test_stale_temporary_cleanup_requires_a_dead_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            stale = run_dir / ".checkpoint.99999999.tmp"
            stale.write_bytes(b"partial")
            unknown = run_dir / ".unknown.tmp"
            unknown.write_bytes(b"preserve")
            self.assertEqual(watchdog.remove_stale_temporary_files(run_dir), [stale.name])
            self.assertFalse(stale.exists())
            self.assertTrue(unknown.exists())

            live = run_dir / f".checkpoint.{os.getpid()}.tmp"
            live.write_bytes(b"active")
            with self.assertRaisesRegex(watchdog.PublicationContradiction, "still alive"):
                watchdog.remove_stale_temporary_files(run_dir)
            self.assertTrue(live.exists())

    def test_invalid_and_valid_unterminated_event_tails_reconcile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            events = run_dir / "events.jsonl"
            first = json.dumps({"event": "train_progress", "step": 10}, sort_keys=True).encode("utf-8") + b"\n"
            events.write_bytes(first + b'{"event":')
            result = watchdog.reconcile_incomplete_event_tail(run_dir)
            self.assertEqual(result["action"], "truncated_invalid_unterminated_tail")
            self.assertEqual(events.read_bytes(), first)

            tail = json.dumps({"event": "checkpoint_verified", "step": 10}, sort_keys=True).encode("utf-8")
            events.write_bytes(first + tail)
            result = watchdog.reconcile_incomplete_event_tail(run_dir)
            self.assertEqual(result["action"], "completed_valid_unterminated_tail")
            self.assertEqual(events.read_bytes(), first + tail + b"\n")

    def test_event_tail_after_run_complete_is_never_reconciled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            complete = json.dumps({"event": "run_complete"}, sort_keys=True).encode("utf-8") + b"\n"
            (run_dir / "events.jsonl").write_bytes(complete + b'{"event":')
            with self.assertRaisesRegex(watchdog.PublicationContradiction, "after a complete run"):
                watchdog.reconcile_incomplete_event_tail(run_dir)

    def test_invalid_tail_reconstructs_a_newer_verified_checkpoint_pointer_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            generation = "step-000000001000-" + "a" * 32 + ".pt"
            checkpoint_hash = write_artifact(run_dir / "checkpoints" / generation, b"verified-checkpoint")
            model_hash = "b" * 64
            write_json(
                run_dir / "checkpoints/latest.json",
                {
                    "schema_version": 1,
                    "kind": "hs_tasnet_c91_checkpoint_pointer",
                    "updated_at_utc": "2026-01-01T00:00:00+00:00",
                    "generation": generation,
                    "path": f"checkpoints/{generation}",
                    "sha256": checkpoint_hash,
                    "step": 1_000,
                    "model_state_sha256": model_hash,
                },
            )
            progress = json.dumps({"event": "train_progress", "step": 1_000}, sort_keys=True).encode("utf-8") + b"\n"
            (run_dir / "events.jsonl").write_bytes(progress + b'{"event": "checkpoint_verified"')
            result = watchdog.reconcile_incomplete_event_tail(run_dir)
            self.assertEqual(result["reconstructed_checkpoint_step"], 1_000)
            records = watchdog.read_events(run_dir / "events.jsonl")
            self.assertEqual(records[-1]["event"], "checkpoint_verified")
            self.assertEqual(records[-1]["sha256"], checkpoint_hash)
            self.assertTrue(records[-1]["reconstructed_by_completion_watchdog"])

            (run_dir / "events.jsonl").write_bytes(progress)
            result = watchdog.reconcile_incomplete_event_tail(run_dir)
            self.assertEqual(result["action"], "reconstructed_missing_pointer_checkpoint_event")
            self.assertEqual(result["reconstructed_checkpoint_step"], 1_000)

            (run_dir / "events.jsonl").unlink()
            result = watchdog.reconcile_incomplete_event_tail(run_dir)
            self.assertEqual(result["action"], "reconstructed_missing_pointer_checkpoint_event")
            self.assertEqual(watchdog.read_events(run_dir / "events.jsonl")[-1]["step"], 1_000)

    def test_missing_emergency_checkpoint_event_preserves_exception_lineage(self) -> None:
        for invalid_tail in (False, True):
            with self.subTest(invalid_tail=invalid_tail), tempfile.TemporaryDirectory() as directory:
                run_dir = Path(directory)
                generation = "step-000000001000-" + "c" * 32 + ".pt"
                checkpoint_hash = write_artifact(run_dir / "checkpoints" / generation, b"emergency-checkpoint")
                model_hash = "d" * 64
                write_json(
                    run_dir / "checkpoints/latest.json",
                    {
                        "schema_version": 1,
                        "kind": "hs_tasnet_c91_checkpoint_pointer",
                        "updated_at_utc": "2026-01-01T00:00:00+00:00",
                        "generation": generation,
                        "path": f"checkpoints/{generation}",
                        "sha256": checkpoint_hash,
                        "step": 1_000,
                        "model_state_sha256": model_hash,
                    },
                )
                exception = json.dumps(
                    {"event": "training_exception", "step": 1_000},
                    sort_keys=True,
                ).encode("utf-8") + b"\n"
                suffix = b'{"event": "emergency_checkpoint_verified"' if invalid_tail else b""
                (run_dir / "events.jsonl").write_bytes(exception + suffix)

                result = watchdog.reconcile_incomplete_event_tail(run_dir)

                self.assertIsNotNone(result)
                records = watchdog.read_events(run_dir / "events.jsonl")
                self.assertEqual(records[-2]["event"], "training_exception")
                self.assertEqual(records[-1]["event"], "emergency_checkpoint_verified")
                self.assertEqual(records[-1]["step"], 1_000)
                self.assertEqual(records[-1]["sha256"], checkpoint_hash)
                self.assertTrue(records[-1]["reconstructed_by_completion_watchdog"])

    def test_missing_checkpoint_event_after_mismatched_exception_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            generation = "step-000000001000-" + "e" * 32 + ".pt"
            checkpoint_hash = write_artifact(run_dir / "checkpoints" / generation, b"older-checkpoint")
            write_json(
                run_dir / "checkpoints/latest.json",
                {
                    "schema_version": 1,
                    "kind": "hs_tasnet_c91_checkpoint_pointer",
                    "updated_at_utc": "2026-01-01T00:00:00+00:00",
                    "generation": generation,
                    "path": f"checkpoints/{generation}",
                    "sha256": checkpoint_hash,
                    "step": 1_000,
                    "model_state_sha256": "f" * 64,
                },
            )
            (run_dir / "events.jsonl").write_text(
                json.dumps({"event": "training_exception", "step": 1_001}, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(watchdog.PublicationContradiction, "does not match"):
                watchdog.reconcile_incomplete_event_tail(run_dir)

    def test_unpublished_final_generation_is_rolled_back_to_verified_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            checkpoint_dir = run_dir / "checkpoints"
            pointer_generation = "step-000000299000-" + "1" * 32 + ".pt"
            unpublished_generation = "step-000000300000-" + "2" * 32 + ".pt"
            pointer_path = checkpoint_dir / pointer_generation
            unpublished_path = checkpoint_dir / unpublished_generation
            pointer_hash = write_artifact(pointer_path, b"step299")
            write_artifact(unpublished_path, b"step300")
            write_json(
                checkpoint_dir / "latest.json",
                {
                    "schema_version": 1,
                    "kind": "hs_tasnet_c91_checkpoint_pointer",
                    "updated_at_utc": "2026-01-01T00:00:00+00:00",
                    "generation": pointer_generation,
                    "path": f"checkpoints/{pointer_generation}",
                    "sha256": pointer_hash,
                    "step": 299_000,
                    "model_state_sha256": "3" * 64,
                },
            )
            records = [
                {
                    "event": "checkpoint_verified",
                    "step": 299_000,
                    "path": str(pointer_path),
                    "sha256": pointer_hash,
                    "model_state_sha256": "3" * 64,
                    "pointer": str(checkpoint_dir / "latest.json"),
                },
                {"event": "train_progress", "step": 300_000},
            ]

            result = watchdog.reconcile_unpublished_checkpoint_generations(run_dir, records)

            self.assertEqual(result["action"], "rolled_back_unpublished_checkpoint_generations")
            self.assertEqual(result["pointer_step"], 299_000)
            self.assertEqual(result["maximum_removed_step"], 300_000)
            self.assertFalse(unpublished_path.exists())
            self.assertFalse(unpublished_path.with_suffix(".pt.sha256").exists())
            self.assertTrue(pointer_path.exists())
            self.assertEqual(watchdog.sha256_file(pointer_path), pointer_hash)
            self.assertIsNone(watchdog.reconcile_unpublished_checkpoint_generations(run_dir, records))

    def test_stale_verified_pause_status_allows_later_unpublished_generation_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            checkpoint_dir = run_dir / "checkpoints"
            pointer_generation = "step-000000299000-" + "c" * 32 + ".pt"
            unpublished_generation = "step-000000300000-" + "d" * 32 + ".pt"
            pointer_path = checkpoint_dir / pointer_generation
            unpublished_path = checkpoint_dir / unpublished_generation
            pointer_hash = write_artifact(pointer_path, b"paused-anchor")
            write_artifact(unpublished_path, b"post-pause-step300")
            write_json(
                checkpoint_dir / "latest.json",
                {
                    "schema_version": 1,
                    "kind": "hs_tasnet_c91_checkpoint_pointer",
                    "updated_at_utc": "2026-01-01T00:00:00+00:00",
                    "generation": pointer_generation,
                    "path": f"checkpoints/{pointer_generation}",
                    "sha256": pointer_hash,
                    "step": 299_000,
                    "model_state_sha256": "e" * 64,
                },
            )
            status = {
                "status": "paused",
                "step": 200_000,
                "total_steps": 300_000,
                "peak_vram_bytes": 123,
                "contract_identity_sha256": watchdog.EXPECTED_CONTRACT,
            }
            write_json(run_dir / "status.json", status)
            records = [{"event": "run_paused", "time": "now", **status}]

            result = watchdog.reconcile_unpublished_checkpoint_generations(run_dir, records)

            self.assertEqual(result["maximum_removed_step"], 300_000)
            self.assertFalse(unpublished_path.exists())
            self.assertTrue(pointer_path.exists())

    def test_unpublished_generation_reconciliation_fails_closed_on_published_or_ambiguous_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            checkpoint_dir = run_dir / "checkpoints"
            pointer_generation = "step-000000249000-" + "4" * 32 + ".pt"
            unpublished_generation = "step-000000250000-" + "5" * 32 + ".pt"
            pointer_hash = write_artifact(checkpoint_dir / pointer_generation, b"step249")
            write_artifact(checkpoint_dir / unpublished_generation, b"step250")
            write_json(
                checkpoint_dir / "latest.json",
                {
                    "schema_version": 1,
                    "kind": "hs_tasnet_c91_checkpoint_pointer",
                    "updated_at_utc": "2026-01-01T00:00:00+00:00",
                    "generation": pointer_generation,
                    "path": f"checkpoints/{pointer_generation}",
                    "sha256": pointer_hash,
                    "step": 249_000,
                    "model_state_sha256": "6" * 64,
                },
            )
            with self.assertRaisesRegex(watchdog.PublicationContradiction, "advance beyond"):
                watchdog.reconcile_unpublished_checkpoint_generations(
                    run_dir,
                    [{"event": "checkpoint_verified", "step": 250_000}],
                )
            self.assertTrue((checkpoint_dir / unpublished_generation).exists())

            disguised = {
                "event": "checkpoint_verified",
                "step": 249_000,
                "path": str(checkpoint_dir / unpublished_generation),
                "pointer": str(checkpoint_dir / "latest.json"),
                "sha256": watchdog.sha256_file(checkpoint_dir / unpublished_generation),
                "model_state_sha256": "7" * 64,
            }
            with self.assertRaisesRegex(watchdog.PublicationContradiction, "filename/step"):
                watchdog.reconcile_unpublished_checkpoint_generations(run_dir, [disguised])
            self.assertTrue((checkpoint_dir / unpublished_generation).exists())

            with self.assertRaisesRegex(watchdog.PublicationContradiction, "terminal completion"):
                watchdog.reconcile_unpublished_checkpoint_generations(
                    run_dir,
                    [{"event": "run_complete", "step": 300_000}],
                )
            self.assertTrue((checkpoint_dir / unpublished_generation).exists())

            duplicate_generation = "step-000000250000-" + "a" * 32 + ".pt"
            write_artifact(checkpoint_dir / duplicate_generation, b"alternate-step250")
            with self.assertRaisesRegex(watchdog.PublicationContradiction, "same step"):
                watchdog.reconcile_unpublished_checkpoint_generations(run_dir, [])
            self.assertTrue((checkpoint_dir / unpublished_generation).exists())
            self.assertTrue((checkpoint_dir / duplicate_generation).exists())

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            generation = "step-000000000000-" + "7" * 32 + ".pt"
            write_artifact(run_dir / "checkpoints" / generation, b"initial")
            with self.assertRaisesRegex(watchdog.PublicationContradiction, "without a checkpoint pointer"):
                watchdog.reconcile_unpublished_checkpoint_generations(run_dir, [])

    def test_step250_emergency_checkpoint_rolls_back_for_native_snapshot_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            checkpoint_dir = run_dir / "checkpoints"
            anchor_generation = "step-000000249000-" + "8" * 32 + ".pt"
            emergency_generation = "step-000000250000-" + "9" * 32 + ".pt"
            anchor_path = checkpoint_dir / anchor_generation
            emergency_path = checkpoint_dir / emergency_generation
            anchor_hash = write_artifact(anchor_path, b"anchor249")
            emergency_hash = write_artifact(emergency_path, b"emergency250")
            pointer_path = checkpoint_dir / "latest.json"
            write_json(
                pointer_path,
                {
                    "schema_version": 1,
                    "kind": "hs_tasnet_c91_checkpoint_pointer",
                    "updated_at_utc": "2026-01-01T00:00:00+00:00",
                    "generation": emergency_generation,
                    "path": f"checkpoints/{emergency_generation}",
                    "sha256": emergency_hash,
                    "step": 250_000,
                    "model_state_sha256": "a" * 64,
                },
            )
            records = [
                {
                    "event": "checkpoint_verified",
                    "step": 249_000,
                    "path": str(anchor_path),
                    "pointer": str(pointer_path),
                    "sha256": anchor_hash,
                    "model_state_sha256": "b" * 64,
                },
                {"event": "train_progress", "step": 250_000},
                {"event": "training_exception", "step": 250_000},
                {
                    "event": "emergency_checkpoint_verified",
                    "step": 250_000,
                    "path": str(emergency_path),
                    "pointer": str(pointer_path),
                    "sha256": emergency_hash,
                    "model_state_sha256": "a" * 64,
                },
            ]

            result = watchdog.reconcile_step250_emergency_checkpoint(run_dir, records)

            self.assertEqual(result["action"], "rolled_back_step250000_emergency_checkpoint")
            self.assertTrue(result["restored_pointer"])
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            self.assertEqual(pointer["step"], 249_000)
            self.assertEqual(pointer["generation"], anchor_generation)
            self.assertEqual(pointer["sha256"], anchor_hash)
            self.assertFalse(emergency_path.exists())
            self.assertFalse(emergency_path.with_suffix(".pt.sha256").exists())
            self.assertTrue(anchor_path.exists())

            replay = watchdog.reconcile_step250_emergency_checkpoint(run_dir, records)
            self.assertFalse(replay["restored_pointer"])
            self.assertEqual(replay["removed"], [])
            self.assertIsNone(watchdog.reconcile_unpublished_checkpoint_generations(run_dir, records))

            records.extend(
                [
                    {"event": "inventory_verified"},
                    {"event": "run_opened"},
                    {"event": "resumed", "step": 249_000, "model_state_sha256": "b" * 64},
                ]
            )
            for _ in range(2):
                resumed_retry = watchdog.reconcile_step250_emergency_checkpoint(run_dir, records)
                self.assertFalse(resumed_retry["restored_pointer"])
                self.assertEqual(resumed_retry["removed"], [])

            records[-1]["model_state_sha256"] = "c" * 64
            with self.assertRaisesRegex(watchdog.PublicationContradiction, "verified anchor"):
                watchdog.reconcile_step250_emergency_checkpoint(run_dir, records)
            records[-1]["model_state_sha256"] = "b" * 64

            records.append(dict(records[-1]))
            with self.assertRaisesRegex(watchdog.PublicationContradiction, "multiple resumes"):
                watchdog.reconcile_step250_emergency_checkpoint(run_dir, records)
            records.pop()

            records.append({"event": "checkpoint_verified", "step": 250_000})
            self.assertIsNone(watchdog.reconcile_step250_emergency_checkpoint(run_dir, records))

    def test_pointer_event_reconciliation_accepts_exact_step250_emergency_replay_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            checkpoint_dir = run_dir / "checkpoints"
            generation = "step-000000250000-" + "f" * 32 + ".pt"
            artifact = checkpoint_dir / generation
            digest = write_artifact(artifact, b"replayed-step250")
            pointer_path = checkpoint_dir / "latest.json"
            write_json(
                pointer_path,
                {
                    "schema_version": 1,
                    "kind": "hs_tasnet_c91_checkpoint_pointer",
                    "updated_at_utc": "2026-01-01T00:00:00+00:00",
                    "generation": generation,
                    "path": f"checkpoints/{generation}",
                    "sha256": digest,
                    "step": 250_000,
                    "model_state_sha256": "1" * 64,
                },
            )
            records = [
                {"event": "training_exception", "step": 250_000},
                {
                    "event": "emergency_checkpoint_verified",
                    "step": 250_000,
                    "path": str(checkpoint_dir / ("step-000000250000-" + "e" * 32 + ".pt")),
                    "pointer": str(pointer_path),
                    "sha256": "2" * 64,
                    "model_state_sha256": "3" * 64,
                },
                {"event": "inventory_verified"},
                {"event": "run_opened"},
                {"event": "resumed", "step": 249_000, "model_state_sha256": "4" * 64},
                {"event": "train_progress", "step": 250_000},
                {
                    "event": "checkpoint_verified",
                    "step": 250_000,
                    "path": str(artifact),
                    "pointer": str(pointer_path),
                    "sha256": digest,
                    "model_state_sha256": "1" * 64,
                },
            ]

            self.assertIsNone(watchdog.recover_missing_pointer_checkpoint_event(run_dir, records))

            emergency_record = records[1]
            valid_emergency_path = emergency_record["path"]
            invalid_emergency_paths = {
                "path escape": "/etc/passwd",
                "wrong-step filename": str(
                    checkpoint_dir / ("step-000000249000-" + "e" * 32 + ".pt")
                ),
            }
            for label, invalid_path in invalid_emergency_paths.items():
                with self.subTest(label=label):
                    emergency_record["path"] = invalid_path
                    with self.assertRaisesRegex(
                        watchdog.PublicationContradiction,
                        "emergency replay evidence is malformed",
                    ):
                        watchdog.recover_missing_pointer_checkpoint_event(run_dir, records)
            emergency_record["path"] = valid_emergency_path
            self.assertIsNone(watchdog.recover_missing_pointer_checkpoint_event(run_dir, records))

            records.append(dict(records[-1]))
            with self.assertRaisesRegex(watchdog.PublicationContradiction, "multiple verified"):
                watchdog.recover_missing_pointer_checkpoint_event(run_dir, records)

    def test_reconciliation_lock_detects_trainer_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            lock = run_dir / ".run.lock"
            lock.touch()
            with lock.open("rb") as owner:
                fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with watchdog.hold_reconciliation_lock(run_dir) as acquired:
                    self.assertFalse(acquired)
                fcntl.flock(owner.fileno(), fcntl.LOCK_UN)
            with watchdog.hold_reconciliation_lock(run_dir) as acquired:
                self.assertTrue(acquired)

    @mock.patch.object(watchdog, "service_properties")
    @mock.patch.object(watchdog, "publication_state")
    @mock.patch.object(watchdog, "start_autoresume")
    def test_active_trainer_short_circuits(self, start: mock.Mock, publication: mock.Mock, properties: mock.Mock) -> None:
        properties.side_effect = [
            {"ActiveState": "inactive"},
            {"ActiveState": "active", "MainPID": "12"},
        ]
        with mock.patch.object(watchdog, "hold_watchdog_lock") as serialized:
            serialized.return_value.__enter__.return_value = None
            serialized.return_value.__exit__.return_value = False
            self.assertEqual(watchdog.main(), 0)
            publication.assert_not_called()
            start.assert_not_called()

    @mock.patch.object(watchdog, "service_properties", return_value={"ActiveState": "inactive"})
    @mock.patch.object(watchdog, "publication_state", return_value={"status": "incomplete_recoverable"})
    @mock.patch.object(watchdog, "start_autoresume", return_value={"ActiveState": "active"})
    @mock.patch.object(
        watchdog,
        "reconcile_crash_leftovers",
        return_value={
            "removed_temporary_files": [],
            "event_tail": None,
            "step250_emergency": None,
            "checkpoint_generations": None,
        },
    )
    def test_partial_publication_starts_autoresume(
        self,
        reconcile: mock.Mock,
        start: mock.Mock,
        publication: mock.Mock,
        properties: mock.Mock,
    ) -> None:
        with mock.patch.object(watchdog, "hold_reconciliation_lock") as locked:
            locked.return_value.__enter__.return_value = True
            locked.return_value.__exit__.return_value = False
            with mock.patch.object(watchdog, "hold_watchdog_lock") as serialized:
                serialized.return_value.__enter__.return_value = None
                serialized.return_value.__exit__.return_value = False
                self.assertEqual(watchdog.main(), 0)
                start.assert_called_once_with()
                reconcile.assert_called_once_with(watchdog.RUN_DIR)


if __name__ == "__main__":
    unittest.main()
