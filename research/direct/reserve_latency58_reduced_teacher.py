"""Retire two completed, unselected Adam files to retain SDR training space."""
from __future__ import annotations

import json
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import disk_bytes, verify_inputs
from research.direct.report_latency58_sdr import load_completed


def main():
    require(Path.cwd() == ROOT, "Unexpected cwd")
    out = PHASE / "reduced-teacher-storage-001"
    require(not out.exists(), "Preserve existing retirement evidence")
    protected = {str(Path(__file__).resolve()): sha(Path(__file__))}
    active_paths = [PHASE / "hare-loss-ablation-001/training-plan.json",
                    PHASE / "latency58-reduced-teacher-001/resource-plan.json"]
    active = [read(p) for p in active_paths]
    for path, plan in zip(active_paths, active, strict=True):
        verify_inputs(plan)
        protected.update(plan["source_bindings"])
        protected[str(path)] = sha(path)
    parent_plan, parent_report = load_completed(PHASE / "leader-cleanup-250-full14-001", protected)
    protected.update(parent_plan["source_bindings"])
    parent_state = parent_report["results"][0]["model"]["model_state_sha256"]
    require(parent_state == "c204b0fcb9627ca7fecd287db42fb869a1ae6783a1bc24cf2d8864c3b4a565fb",
            "Accepted rollback model differs")
    remove = []
    for prefix, run_name in (
        ("cleanup-followup", "cleanup-followup-b16-micro4-lr1e5-250"),
        ("quarter-controlled", "quarter-controlled-b16-micro4-lr1e5-250"),
    ):
        run = PHASE / run_name
        config, status, latest = [read(run / leaf) for leaf in ("config.json", "status.json", "audit-latest.json")]
        require(status["status"] == "complete" and status["step"] == config["config"]["steps"] == 250
                and not config["automatic_continuation"], "Old training schedule is not closed")
        for proc in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                command = proc.read_bytes()
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            require(run_name.encode() not in command and
                    ("research.direct.train_latency58_" + prefix.replace("-", "_")).encode()
                    not in command, "Old checkpoint remains in a running command")
        audit, execution = [read(latest[k]["path"]) for k in ("audit", "execution")]
        require(all(sha(latest[k]["path"]) == latest[k]["sha256"] for k in ("audit", "execution"))
                and audit["status"] == "pass" and audit["source_bindings_unchanged"]
                and execution["actual_exit_code"] == 0 and not execution["timed_out"]
                and execution["source_bindings_unchanged"], "Saved-state audit is incomplete or changed")
        quality, report = load_completed(PHASE / (prefix + "-250-full14-001"), protected)
        protected.update(quality["source_bindings"])
        generation = run / "checkpoints/step-000250"
        receipt = read(generation / "receipt.json")
        fingerprint = report["results"][0]["model"]["model_state_sha256"]
        require(fingerprint == receipt["model_state_sha256"] == audit["model_state_sha256"]
                and fingerprint != parent_state and quality["generation"] == str(generation)
                and report["results"][0]["aggregate"]["full_sdr_db"]
                    < parent_report["results"][0]["aggregate"]["full_sdr_db"]
                and sha(generation / "receipt.json") == audit["generation_receipt_sha256"]
                    == latest["generation_receipt_sha256"], "Different audited or scored endpoint")
        for leaf, expected in receipt["files"].items():
            path = generation / leaf
            require(path.is_file() and not path.is_symlink() and path.stat().st_size == expected["bytes"]
                    and sha(path) == expected["sha256"], "Saved generation changed")
            if leaf == "optimizer.pt":
                remove.append({"path": str(path), **expected, "model_state_sha256": fingerprint})
            else:
                protected[str(path)] = expected["sha256"]
        for path in [generation / "receipt.json", *[run / n for n in ("config.json", "status.json", "audit-latest.json")],
                     *[Path(latest[k]["path"]) for k in ("audit", "execution")]]:
            protected[str(path)] = sha(path)
    require(len(remove) == 2 and set(protected).isdisjoint(row["path"] for row in remove),
            "An optimizer is still a protected input")
    verify_inputs({"source_bindings": protected})
    before = sum(disk_bytes(Path(p)) for p in active[0]["counted_roots"])
    freed = sum(row["bytes"] for row in remove)
    # Current training completion plus one further monitored checkpoint/score.
    require(before - freed + 800_000_000 + 800_000_000 < 80_000_000_000,
            "Retirement cannot fund the current and proposed bounded trials")
    out.mkdir()
    write(out / "intent.json", {"status": "audited", "remove": remove, "protected_bindings": protected,
          "counted_bytes_before": before, "outside_roots_reservation_bytes": 800_000_000,
          "training_and_quality_reserve_bytes": 800_000_000,
          "reason": "Free obsolete Adam state from two completed, lower-SDR trials. Keep all inference models, RNG, journals, receipts and accepted baselines."})
    for row in remove:
        path = Path(row["path"])
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == row["bytes"]
                and sha(path) == row["sha256"], "Optimizer changed immediately before retirement")
        path.unlink()
    verify_inputs({"source_bindings": protected})
    after = sum(disk_bytes(Path(p)) for p in active[0]["counted_roots"])
    write(out / "receipt.json", {"status": "complete", "intent_sha256": sha(out / "intent.json"),
          "retired": remove, "freed_bytes": freed, "protected_bindings_unchanged": True,
          "active_training_inputs_unchanged": True, "counted_bytes_after": after,
          "combined_forecast_bytes": after + 800_000_000 + 800_000_000})
    print(json.dumps({"status": "complete", "freed_bytes": freed, "counted_bytes_after": after}), flush=True)


if __name__ == "__main__":
    main()
