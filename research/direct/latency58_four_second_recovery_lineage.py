"""Authenticate receipt segments across lossless packed training resumes.

Each new run starts a fresh publication chain under its own plan hash. The
resume binding connects it to the retained prior save; historical receipts
must never be copied or relabelled as publications by a later plan.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from research.direct.run_latency58_quality import read, require, sha


def segments(plan_path, checkpoint):
    result, seen = [], set()
    plan_path = Path(plan_path)
    while True:
        require(str(plan_path) not in seen, "Cycle in recovery plans")
        seen.add(str(plan_path))
        plan = read(plan_path)
        resume = plan.get("resume_checkpoint")
        start = resume["snapshot"]["step"] if resume else 0
        require(type(start) is int and type(checkpoint["step"]) is int
                and 0 <= start < checkpoint["step"] <= plan["config"]["steps"],
                "Invalid recovery segment endpoints")
        result.append({"plan_path": plan_path, "plan": plan, "plan_sha256": sha(plan_path),
                       "checkpoint": checkpoint, "start_step": start, "end_step": checkpoint["step"]})
        if resume is None:
            break
        parent = Path(resume["training_plan"]["path"])
        require(sha(parent) == resume["training_plan"]["sha256"], "Recovery parent plan changed")
        parent_plan = read(parent)
        for key in ("config", "parent_checkpoint", "parent_model_state_sha256", "parent_training_updates",
                    "ema", "fixed_buffers_sha256", "objective_version", "grouped_vocal_loss",
                    "accumulation_policy", "qualified_data_prefix", "inference_architecture", "recovery_checkpoint"):
            require(plan[key] == parent_plan[key], "Recovery changed the scientific trajectory")
        plan_path, checkpoint = parent, resume["snapshot"]
    return list(reversed(result))


def verify_receipts(receipts, *, start, end, plan_sha, journal, checkpoint_sha, planned_stop):
    """Pure validation used by the real saved-state auditor and corruption tests."""
    require(0 <= start < end <= planned_stop and start % 50 == end % 50 == 0,
            "Invalid receipt segment endpoints")
    expected = list(range(start + 50, end + 1, 50))
    require([row["step"] for row in receipts] == expected, "Packed recovery receipt schedule differs")
    lines = journal.splitlines(keepends=True)
    require(len(lines) >= end, "Journal is shorter than the saved segment")
    prefixes, digest = {}, hashlib.sha256()
    for step, line in enumerate(lines[:end], 1):
        digest.update(line)
        if step in expected:
            prefixes[step] = digest.hexdigest()
    previous = None
    for row in receipts:
        require(row["plan_sha256"] == plan_sha and row["previous"] == previous
                and row["planned_stop_step"] == planned_stop and 0 < row["bytes"] <= 500_000_000
                and row["journal_sha256"] == prefixes[row["step"]],
                "Packed receipt plan, previous generation or journal differs")
        previous = {key: row[key] for key in ("step", "sha256", "bytes")}
    require(previous["sha256"] == checkpoint_sha, "Segment does not end at its bound recovery save")
    return expected


def bindings_for_segments(plan_path, checkpoint):
    bindings = {}
    for segment in segments(plan_path, checkpoint):
        saved = segment["checkpoint"]
        paths = [segment["plan_path"], Path(saved["path"]), Path(saved["receipt"])]
        paths.extend((Path(saved["path"]).parent / "packed-recovery-receipts").glob("step-*.json"))
        for path in paths:
            require(path.is_file() and not path.is_symlink(), "Invalid retained recovery artifact")
            bindings[str(path)] = sha(path)
    return bindings


def audit_lineage(plan_path, checkpoint, journal):
    from research.direct.train_latency58 import verify_inputs
    from research.direct.latency58_four_second_recovery_files import read_snapshot
    reports, all_steps = [], []
    for segment in segments(plan_path, checkpoint):
        source, saved = segment["plan"], segment["checkpoint"]
        verify_inputs(source)
        snapshot, audited = read_snapshot(saved, source, segment["plan_sha256"])
        end = segment["end_step"]
        require(snapshot["journal"] == b"".join(journal.splitlines(keepends=True)[:end]),
                "Resumed journal differs from retained parent save")
        paths = sorted((Path(saved["path"]).parent / "packed-recovery-receipts").glob("step-*.json"))
        require([p.name for p in paths] == [f"step-{step:06d}.json"
                for step in range(segment["start_step"] + 50, end + 1, 50)], "Receipt filename inventory differs")
        require(all(not p.is_symlink() for p in paths), "Receipt aliases are forbidden")
        steps = verify_receipts([read(p) for p in paths], start=segment["start_step"], end=end,
            plan_sha=segment["plan_sha256"], journal=journal, checkpoint_sha=saved["sha256"],
            planned_stop=source["config"]["steps"])
        all_steps.extend(steps)
        reports.append({"plan": {"path": str(segment["plan_path"]), "sha256": segment["plan_sha256"]},
            "start_step": segment["start_step"], "end_step": end, "receipt_steps": steps,
            "checkpoint": saved, "complete_raw_adam_ema_rng_audit_passed": True,
            "journal_prefix_exact": True})
        del snapshot, audited
    require(all_steps == list(range(50, checkpoint["step"] + 1, 50)), "Recovery lineage has gaps or duplicates")
    return {"status": "pass", "segments": reports, "receipt_chain_generations": len(all_steps),
            "resume_bindings_connect_retained_saves": True, "receipts_relabelled": False}
