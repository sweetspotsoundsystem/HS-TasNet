"""Losslessly archive two completed history watchdog logs to reserve pilot quality space.

The original monitor receipts remain unchanged. Adjacent gzip files preserve
every original byte and hash; no model, RNG, training journal or failed run is
removed. This is a reversible representation change for these two logs only.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import os
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_vocal_focus_checkpoint import require_space


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Archive plan or cwd differs")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-closed-history-monitor-archive-plan-v1"
            and plan["temporary_archive_reserve_bytes"] == 2_000_000
            and plan["minimum_completion_and_quality_reserve_bytes"] == 380_000_000,
            "Different archive scope or storage reserve")
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "receipt.json").exists(), "Preserve archive receipt")
    for key in ("history_review", "history_review_execution", "history_retirement", "history_retirement_execution"):
        item = plan[key]
        require(plan["source_bindings"].get(item["path"]) == item["sha256"] == sha(item["path"]),
                "Unbound closed-history evidence")
    review, execution = (read(plan[key]["path"]) for key in ("history_review", "history_review_execution"))
    retired, retirement_execution = (read(plan[key]["path"]) for key in ("history_retirement", "history_retirement_execution"))
    require(review["schema"] == "latency58-history-500-review-v1" and review["status"] == "pass"
            and review["source_bindings_unchanged"] and review["arms"]["long"]["training_closed"]
            and review["arms"]["long"]["completed_step"] == review["arms"]["long"]["original_maximum_step"] == 500
            and review["arms"]["long"]["further_optimizer_updates"] == 0
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"] and execution["plan_sha256"] == review["plan_sha256"]
            and retired["schema"] == "latency58-closed-history-optimizer-retirement-v1"
            and retired["status"] == "complete" and retired["source_bindings_unchanged"]
            and retirement_execution["actual_exit_code"] == 0 and not retirement_execution["timed_out"]
            and retirement_execution["source_bindings_unchanged"]
            and retirement_execution["plan_sha256"] == retired["plan_sha256"], "History schedules are not closed")
    for item in plan["active_plans"]:
        require(plan["source_bindings"].get(item["path"]) == item["sha256"] == sha(item["path"]), "Active plan changed")
        verify_inputs(read(item["path"]))
    base = ROOT / "research/direct/runs/latency11/smoke/gpu-crash-followup"
    expected = [base / (f"latency58-sdr-history-long-to-{step:06d}-001") / "watchdog.jsonl" for step in (250, 500)]
    require([Path(row["original"]["path"]) for row in plan["logs"]] == expected, "Only these two closed logs may be archived")
    before = require_space(plan, 2_000_000)
    prepared = []
    for row in plan["logs"]:
        original = Path(row["original"]["path"])
        target = Path(row["archive"]["path"])
        require(target == original.with_suffix(".jsonl.gz") and not target.exists()
                and original.is_file() and not original.is_symlink()
                and str(original) not in plan["source_bindings"], "Different or active archive input")
        for key in ("monitor", "training_execution"):
            item = row[key]
            require(plan["source_bindings"].get(item["path"]) == item["sha256"] == sha(item["path"]), "Unbound monitor")
        monitor, training_execution = (read(row[key]["path"]) for key in ("monitor", "training_execution"))
        require(monitor["schema"] == "gpu-process-watchdog-v1" and monitor["status"] == monitor["supervisor_health"] == "pass"
                and monitor["child_exit_code"] == 0 and monitor["post_exit_quiet_completed"]
                and monitor["artifacts"]["watchdog_log"] == row["original"]
                and training_execution["actual_exit_code"] == 0 and training_execution["source_bindings_unchanged"]
                and training_execution["monitor_result"] == row["monitor"]["path"], "Monitor did not finish cleanly")
        raw = original.read_bytes()
        require(len(raw) == row["original"]["bytes"] and hashlib.sha256(raw).hexdigest() == row["original"]["sha256"],
                "Original log differs from its completed monitor receipt")
        packed = gzip.compress(raw, compresslevel=9, mtime=0)
        require(gzip.decompress(packed) == raw and len(packed) == row["archive"]["bytes"]
                and hashlib.sha256(packed).hexdigest() == row["archive"]["sha256"], "Archive encoding or round-trip differs")
        prepared.append((row, original, target, packed))
    require(sum(len(data) for _, _, _, data in prepared) < 2_000_000, "Temporary archive allowance exceeded")
    savings = sum(row["original"]["bytes"] - len(data) for row, _, _, data in prepared)
    require(before - savings + 380_000_000 < plan["stop_counted_bytes"], "Archive would not reserve training and quality")
    write(out / "intent.json", {"plan_sha256": args.plan_sha256, "logs": plan["logs"], "original_bytes_recoverable": True})
    for row, original, target, packed in prepared:
        with target.open("xb") as stream:
            stream.write(packed)
            stream.flush()
            os.fsync(stream.fileno())
        require(sha(target) == row["archive"]["sha256"]
                and hashlib.sha256(gzip.decompress(target.read_bytes())).hexdigest() == row["original"]["sha256"]
                and sha(original) == row["original"]["sha256"], "Archive failed immediate readback")
    verify_inputs(plan)
    for row, original, target, _ in prepared:
        require(sha(original) == row["original"]["sha256"], "Log changed before representation replacement")
        original.unlink()
        write(original.with_suffix(".jsonl.archive.json"), {"original": row["original"], "archive": row["archive"],
              "archive_plan": {"path": str(args.plan), "sha256": args.plan_sha256}, "lossless": True})
        require(not original.exists() and len(gzip.decompress(target.read_bytes())) == row["original"]["bytes"]
                and hashlib.sha256(gzip.decompress(target.read_bytes())).hexdigest() == row["original"]["sha256"],
                "Archived original bytes cannot be recovered")
    verify_inputs(plan)
    for item in plan["active_plans"]:
        verify_inputs(read(item["path"]))
    after = require_space(plan, 380_000_000)
    write(out / "receipt.json", {"schema": "latency58-closed-history-monitor-archive-v1", "status": "complete",
          "plan_sha256": args.plan_sha256, "logs": plan["logs"], "original_bytes_recoverable": True,
          "all_original_sha256_reproduced": True, "source_bindings_unchanged": True, "active_plan_inputs_unchanged": True,
          "counted_bytes_before": before, "counted_bytes_after": after, "payload_savings_bytes": savings,
          "reserved_completion_and_quality_bytes": 380_000_000, "source_audio_or_models_changed": False,
          "training_journals_or_rng_changed": False, "failed_run_artifacts_changed": False})
    print({"status": "complete", "logs": 2, "payload_savings_bytes": savings, "counted_bytes_after": after}, flush=True)


if __name__ == "__main__":
    main()
