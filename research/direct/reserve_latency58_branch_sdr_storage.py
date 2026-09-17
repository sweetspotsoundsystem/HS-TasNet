"""Reserve one continuation by retiring only historical OLA step-1300 resume data."""
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

RESERVE = 460_000_000
OLA_STEP = 1300
OLA_RESUME_SHA = "1168e064e9c355f301eb1729089b3e270def477ccc8fd2503579eb5b736431d7"
OLA_RESUME_BYTES = 318_519_698


def main():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Use the CPU workspace")
    parent = PHASE / "branch-memory-001"
    endpoint = parent / "validation-recovery-001"
    source, best = read(parent / "plan.json"), read(parent / "selection-review.json")
    require(best["status"] == "selected_for_research" and best["actual_root_exit_code"] == 0
            and best["best_full_sdr_db"] == 4.391035784116766 < 5.0
            and best["optimizer_may_be_retired_after_this_review"] is False
            and best["completion_mode"] == "validation_recovery"
            and best["original_root_actual_exit_code"] is None
            and best["best_research_reference_result"] == str(endpoint / "full14/result.json"),
            "Require the completed recovered branch-memory selection")
    verify_inputs(source)
    verify_inputs(best)
    previous_path = PHASE / "branch-memory-storage-001/intent.json"
    previous, receipt = read(previous_path), read(previous_path.parent / "receipt.json")
    require(receipt["status"] == "complete" and receipt["preserved_bindings_unchanged"]
            and receipt["intent_sha256"] == sha(previous_path), "Previous retirement is incomplete")
    protected = {}
    for bindings in (previous["preserved_bindings"], source["source_bindings"], best["source_bindings"]):
        for path, digest in bindings.items():
            require(path not in protected or protected[path] == digest, "Conflicting protected identities")
            protected[path] = digest
    paths = [Path(__file__).resolve(), previous_path, previous_path.parent / "receipt.json", parent / "plan.json"]
    for path in (parent / "root-execution.json", parent / "production-stage/execution.json",
                 endpoint / "full14/execution.json", PHASE / "branch-memory-review-stage-001/execution.json"):
        execution = read(path)
        require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                and not execution.get("timed_out", False), "A preceding execution remains incomplete")
        paths.append(path)
    paths.extend(parent / name for name in ("selection-review.json", "checkpoint-audit.json",
                 "production-run/checkpoint/model.pt", "production-run/checkpoint/optimizer.pt",
                 "production-run/checkpoint/receipt.json"))
    paths.extend(endpoint / name for name in ("result.json", "full14/result.json"))
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
    lock = (old / "trainer.lock").open("r")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    no_historical_user(old)
    no_historical_user(parent / "production-run")
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
            and historical_receipt["plan_sha256"] == latest["plan_sha256"]
            and historical_receipt["files"]["resume.pt"] == OLA_RESUME_SHA,
            "Historical intermediate identity changed")
    for name in ("model.pt", "metrics.jsonl"):
        require(sha(generation / name) == historical_receipt["files"][name], "Historical model or journal changed")
        paths.append(generation / name)
    paths.append(generation / "receipt.json")
    for path in paths:
        digest = sha(path)
        require(str(path) not in protected or protected[str(path)] == digest, "Protected file changed")
        protected[str(path)] = digest
    chosen = {"kind": "historical_intermediate_optimizer", "path": str(generation / "resume.pt"),
              "bytes": OLA_RESUME_BYTES, "sha256": OLA_RESUME_SHA, "step": OLA_STEP}

    def verify_target():
        path = Path(chosen["path"])
        require(path.is_file() and not path.is_symlink() and path.resolve() == path
                and path.stat().st_nlink == 1 and path.stat().st_size == chosen["bytes"]
                and sha(path) == chosen["sha256"] and str(path) not in protected, "Invalid retirement target")

    verify_target()
    verify_inputs({"source_bindings": protected})
    before = require_space(source, 0)
    require(before + RESERVE >= source["stop_counted_bytes"]
            and before - OLA_RESUME_BYTES + RESERVE + 4_000_000 < source["stop_counted_bytes"],
            "Retire this one resume only if necessary and sufficient")
    out = PHASE / "branch-sdr-blend-storage-001"
    require(not out.exists(), "Preserve previous storage records")
    out.mkdir()
    write(out / "intent.json", {"targets": [chosen], "preserved_bindings": protected,
          "counted_bytes_before": before, "reserved_bytes": RESERVE,
          "reservation_details": "440 MB checkpoint guard plus 20 MB for 1000-update journal, monitor and metadata growth",
          "all_inference_models_and_source_audio_preserved": True, "all_selected_optimizers_preserved": True,
          "limitation": "Historical OLA step-1300 loses exact optimizer/RNG restart. Its model, receipts and journals, all selected optimizer checkpoints and the OLA step-2000 terminal resume remain."})
    verify_inputs({"source_bindings": protected})
    require_no_build(worktree)
    no_historical_user(old)
    verify_target()
    Path(chosen["path"]).unlink()
    verify_inputs({"source_bindings": protected})
    require(not Path(chosen["path"]).exists(), "Retirement is incomplete")
    result = {"status": "complete", "intent_sha256": sha(out / "intent.json"), "removed": [chosen],
              "reclaimed_bytes": OLA_RESUME_BYTES, "reserved_bytes": RESERVE, "preserved_bindings_unchanged": True,
              "all_inference_models_and_source_audio_preserved": True, "all_selected_optimizers_preserved": True,
              "historical_terminal_resume_preserved": True, "plugin_and_native_binaries_preserved": True,
              "counted_bytes_after": require_space(source, RESERVE)}
    write(out / "receipt.json", result)
    lock.close()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
