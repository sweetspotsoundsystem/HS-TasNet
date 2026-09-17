"""Reserve 500 MB after review using two specifically identified rejected/intermediate resumes."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess

from research.direct.run_latency58_quality import ROOT, PHASE, read, write, sha, require
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_sdr_checkpoint import require_space
from research.direct.reclaim_latency58_attention_build_objects import require_no_build
from research.direct.reserve_latency58_attention_sdr_blend_storage import no_historical_user

RESERVE = 500_000_000
OLA_STEP = 1275
OLA_RESUME_SHA = "2a591c8a40d1cd580bc4c55b7c9a915ab21ce3b7612ca8df3dc80c363f24fe1c"
OLA_RESUME_BYTES = 318_519_698


def main():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Use the CPU workspace")
    parent, rejected = PHASE / "attention-sdr-blend-001", PHASE / "attention-grouped-001"
    source = read(parent / "plan.json")
    best, review = read(parent / "selection-review.json"), read(rejected / "selection-review.json")
    require(best["status"] == "selected_for_research" and best["actual_root_exit_code"] == 0
            and best["best_full_sdr_db"] == 4.3535847721055445 < 5.0
            and best["optimizer_may_be_retired_after_this_review"] is False
            and review["status"] == "not_selected" and review["actual_root_exit_code"] == 0
            and review["optimizer_may_be_retired_after_this_review"] is True,
            "Require completed saved selection and explicit eligibility of the rejected optimizer")
    verify_inputs(source)
    verify_inputs(best)
    verify_inputs(review)
    protected = {}
    previous_path = PHASE / "attention-sdr-blend-storage-001/intent.json"
    previous, receipt = read(previous_path), read(previous_path.parent / "receipt.json")
    require(receipt["status"] == "complete" and receipt["preserved_bindings_unchanged"]
            and receipt["intent_sha256"] == sha(previous_path), "Previous storage recovery is incomplete")
    for bindings in (previous["preserved_bindings"], source["source_bindings"],
                     best["source_bindings"], review["source_bindings"]):
        for path, digest in bindings.items():
            require(path not in protected or protected[path] == digest, "Conflicting protected identities")
            protected[path] = digest
    paths = [Path(__file__).resolve(), previous_path, previous_path.parent / "receipt.json", parent / "plan.json"]
    for root in (parent, rejected):
        for name in ("root-execution.json", "production-stage/execution.json", "full14/execution.json"):
            execution = read(root / name)
            require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                    and not execution.get("timed_out", False), "A prior execution remains incomplete")
            paths.append(root / name)
        paths.extend(root / name for name in ("selection-review.json", "result.json", "checkpoint-audit.json",
                     "full14/result.json", "production-run/checkpoint/model.pt", "production-run/checkpoint/receipt.json"))
    paths.append(parent / "production-run/checkpoint/optimizer.pt")
    monitor_path = Path(read(parent / "production-stage/execution.json")["monitor_result"])
    monitor = read(monitor_path)
    require(monitor["status"] == monitor["supervisor_health"] == "pass" and monitor["child_exit_code"] == 0
            and monitor["post_exit_quiet_completed"], "The latest GPU monitor has not closed successfully")
    paths.append(monitor_path)
    worktree = PHASE / "best-model-stemgen-rt-001"
    require_no_build(worktree)
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=worktree, text=True).strip()
            == "c84805083c1127f6749f140ce8a49ae8b5acbf54"
            and not subprocess.check_output(["git", "status", "--porcelain"], cwd=worktree),
            "Preserve the published plugin worktree")
    old = ROOT / "research/direct/runs/latency11/ola512-right-baked-gpu-b4-bf16-projection01-lr3e-5"
    locks = []
    for run in (old, rejected / "production-run"):
        lock = (run / "trainer.lock").open("r")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
        no_historical_user(run)
    latest, status = read(old / "latest.json"), read(old / "status.json")
    require(status["status"] == "complete" and status["step"] == latest["step"] == 2000
            and status["latest_checkpoint"] == latest, "Historical training is incomplete")
    terminal = Path(latest["generation"])
    require(terminal == old / "checkpoints/step-002000", "Wrong historical terminal")
    for name, digest in latest["files"].items():
        require(sha(terminal / name) == digest, "Historical terminal changed")
        paths.append(terminal / name)
    require(sha(terminal / "receipt.json") == latest["receipt_sha256"], "Historical terminal receipt changed")
    paths.append(terminal / "receipt.json")
    paths.extend(old / name for name in ("status.json", "latest.json", "config.json", "metrics.jsonl"))
    generation = old / "checkpoints" / f"step-{OLA_STEP:06d}"
    historical_receipt = read(generation / "receipt.json")
    require(historical_receipt["status"] == "pass" and historical_receipt["step"] == OLA_STEP
            and historical_receipt["plan_sha256"] == latest["plan_sha256"], "Historical intermediate identity changed")
    for name in ("model.pt", "metrics.jsonl"):
        require(sha(generation / name) == historical_receipt["files"][name], "Historical model or journal changed")
        paths.append(generation / name)
    paths.append(generation / "receipt.json")
    group_generation = rejected / "production-run/checkpoint"
    group_receipt, group_audit = read(group_generation / "receipt.json"), read(rejected / "checkpoint-audit.json")
    group_resume = group_receipt["files"]["optimizer.pt"]
    require(group_resume["sha256"] == group_audit["optimizer_sha256"] == review["audited_optimizer_sha256"],
            "Rejected grouped optimizer differs from its saved audit")
    chosen = [
        {"kind": "rejected_trial_optimizer", "path": str(group_generation / "optimizer.pt"), **group_resume},
        {"kind": "historical_intermediate_optimizer", "path": str(generation / "resume.pt"),
         "bytes": OLA_RESUME_BYTES, "sha256": OLA_RESUME_SHA, "step": OLA_STEP},
    ]
    require(historical_receipt["files"]["resume.pt"] == OLA_RESUME_SHA, "Wrong historical resume receipt")
    for path in paths:
        digest = sha(path)
        require(str(path) not in protected or protected[str(path)] == digest, "Protected file changed")
        protected[str(path)] = digest
    for item in chosen:
        path = Path(item["path"])
        require(path.is_file() and not path.is_symlink() and path.resolve() == path
                and path.stat().st_nlink == 1 and path.stat().st_size == item["bytes"]
                and sha(path) == item["sha256"] and str(path) not in protected, "Invalid retirement target")
    verify_inputs({"source_bindings": protected})
    before = require_space(source, 0)
    reclaimed = sum(item["bytes"] for item in chosen)
    require(before - reclaimed + RESERVE + 4_000_000 < source["stop_counted_bytes"]
            and all(before - item["bytes"] + RESERVE >= source["stop_counted_bytes"] for item in chosen),
            "Retire both only if required and sufficient for this reservation")
    out = PHASE / "branch-memory-storage-001"
    require(not out.exists(), "Preserve previous storage records")
    out.mkdir()
    write(out / "intent.json", {"targets": chosen, "preserved_bindings": protected,
          "counted_bytes_before": before, "reserved_bytes": RESERVE,
          "all_inference_models_and_source_audio_preserved": True, "all_selected_optimizers_preserved": True,
          "limitation": "The rejected grouped trial and historical OLA step-1275 lose exact optimizer/RNG restart. Their inference models, receipts and journals, all selected optimizer checkpoints and the OLA step-2000 terminal resume remain."})
    verify_inputs({"source_bindings": protected})
    require_no_build(worktree)
    for run in (old, rejected / "production-run"):
        no_historical_user(run)
    for item in chosen:
        path = Path(item["path"])
        require(path.is_file() and not path.is_symlink() and path.resolve() == path and path.stat().st_nlink == 1
                and path.stat().st_size == item["bytes"] and sha(path) == item["sha256"]
                and str(path) not in protected, "Retirement target changed")
        path.unlink()
    verify_inputs({"source_bindings": protected})
    require(all(not Path(item["path"]).exists() for item in chosen), "Retirement is incomplete")
    result = {"status": "complete", "intent_sha256": sha(out / "intent.json"), "removed": chosen,
              "reclaimed_bytes": reclaimed, "reserved_bytes": RESERVE, "preserved_bindings_unchanged": True,
              "all_inference_models_and_source_audio_preserved": True, "all_selected_optimizers_preserved": True,
              "historical_terminal_resume_preserved": True, "plugin_and_native_binaries_preserved": True,
              "counted_bytes_after": require_space(source, RESERVE)}
    write(out / "receipt.json", result)
    for lock in locks:
        lock.close()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
