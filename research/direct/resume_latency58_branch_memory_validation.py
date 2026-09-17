"""Recover full14 validation after its original launcher was interrupted.

Preserve the original attempt, wait for its surviving processes to exit, and
repeat the identical saved-checkpoint evaluation with a recorded child exit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, execute, read, require, sha, write
from research.direct.train_latency58 import verify_inputs


TRIAL = PHASE / "branch-memory-001"
OUT = TRIAL / "validation-recovery-001"
ORIGINAL_PROCESSES = {54151: 31762135, 91246: 33091395, 91248: 33091957, 91251: 33091962}


def live_original_processes():
    live = []
    for pid, expected_start in ORIGINAL_PROCESSES.items():
        try:
            fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(") ", 1)[1].split()
        except FileNotFoundError:
            continue
        if int(fields[19]) == expected_start and fields[0] not in ("Z", "X"):
            live.append(pid)
    return live


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root-session", type=int, required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and args.original_root_session == 78470, "Wrong interrupted trial")
    require(not OUT.exists(), "Preserve existing recovery evidence")
    require(54151 not in live_original_processes(), "The original launcher still owns validation")
    plan_path = TRIAL / "plan.json"
    plan = read(plan_path)
    verify_inputs(plan)
    training = read(TRIAL / "production-run/result.json")
    audit = read(TRIAL / "checkpoint-audit.json")
    command = read(TRIAL / "root-command.json")
    require(command["actual_root_session"] == args.original_root_session, "Original launcher identity changed")
    require(training["status"] == audit["status"] == "pass" and training["updates"] == audit["step"] == 4000
            and training["checkpoint_written"] and training["source_bindings_unchanged"]
            and audit["source_bindings_unchanged"] and audit["saved_optimizer_tensor_count"] == 40
            and audit["algorithmic_latency_samples"] == 256
            and training["checkpoint"] == audit["checkpoint"], "Require the completed, audited training endpoint")
    for stage in ("resource-stage", "production-stage"):
        execution = read(TRIAL / stage / "execution.json")
        monitor = read(execution["monitor_result"])
        require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                and monitor["status"] == monitor["supervisor_health"] == "pass"
                and monitor["child_exit_code"] == 0 and monitor["post_exit_quiet_completed"],
                "Training monitor did not close successfully")
    checkpoint_dir = Path(training["checkpoint"]["path"]).parent
    receipt = read(checkpoint_dir / "receipt.json")
    require(receipt["step"] == 4000 and receipt["plan_sha256"] == sha(plan_path), "Wrong saved generation")
    for name, entry in receipt["files"].items():
        path = checkpoint_dir / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == entry["bytes"]
                and sha(path) == entry["sha256"], "Saved generation changed")
    from research.direct.latency58_sdr_checkpoint import require_space
    require_space(plan, 12_000_000)
    OUT.mkdir()
    original_quality_path = TRIAL / "full14/plan.json"
    original_quality = read(original_quality_path)
    bindings = {**original_quality["source_bindings"], str(original_quality_path): sha(original_quality_path),
                str(Path(__file__).resolve()): sha(__file__),
                **{str(checkpoint_dir / name): entry["sha256"] for name, entry in receipt["files"].items()},
                str(checkpoint_dir / "receipt.json"): sha(checkpoint_dir / "receipt.json")}
    intent = {"schema": "latency58-branch-memory-validation-recovery-v1", "original_root_session": 78470,
              "original_root_actual_exit_code": None, "original_root_exit_status_available": False,
              "reason": "The Codex launcher disappeared during the pause; its evaluator survived without an exit receipt.",
              "original_process_start_ticks": {str(k): v for k, v in ORIGINAL_PROCESSES.items()},
              "original_live_processes_at_recovery_start": live_original_processes(),
              "training_repeated": False, "checkpoint": training["checkpoint"],
              "source_bindings": bindings}
    write(OUT / "intent.json", intent)
    began = time.monotonic()
    while live := live_original_processes():
        require(time.monotonic() - began < 1800, "Original evaluation is still active; no duplicate launched")
        print(json.dumps({"event": "waiting_for_original_evaluation", "live_pids": live}), flush=True)
        time.sleep(10)
    original_result_path = TRIAL / "full14/result.json"
    original_result = read(original_result_path) if original_result_path.exists() else None
    observation = {"original_processes_terminal_or_absent": True, "original_evaluator_exit_code": None,
                   "exit_status_not_inferred_from_output": True,
                   "original_result": ({"path": str(original_result_path), "sha256": sha(original_result_path)}
                                       if original_result is not None else None)}
    write(OUT / "original-attempt.json", observation)
    bindings[str(OUT / "intent.json")] = sha(OUT / "intent.json")
    bindings[str(OUT / "original-attempt.json")] = sha(OUT / "original-attempt.json")
    if original_result is not None:
        bindings[str(original_result_path)] = sha(original_result_path)
    verify_inputs({"source_bindings": bindings})
    quality = OUT / "full14"
    quality.mkdir()
    quality_plan = {**original_quality, "output_directory": str(quality), "source_bindings": bindings}
    require({k: v for k, v in quality_plan.items() if k not in ("output_directory", "source_bindings")}
            == {k: v for k, v in original_quality.items() if k not in ("output_directory", "source_bindings")},
            "Recovery changed the checkpoint or validation protocol")
    write(quality / "plan.json", quality_plan)
    execution_bindings = {**bindings, str(quality / "plan.json"): sha(quality / "plan.json")}
    print(json.dumps({"event": "repeating_saved_checkpoint_full14", "checkpoint": training["checkpoint"]}), flush=True)
    execute([PYTHON, "-u", "-m", "research.direct.evaluate_latency58_branch_memory", "--plan", str(quality / "plan.json"),
             "--plan-sha256", sha(quality / "plan.json")], quality, "evaluation", 1800,
            execution_bindings, {"plan_sha256": sha(quality / "plan.json")})
    report = read(quality / "result.json")
    require(report["status"] == "pass" and report["track_count"] == 14 and report["excerpt_count"] == 28
            and report["source_bindings_unchanged"] and report["results"][0]["checkpoint"] == training["checkpoint"],
            "Recovered validation is incomplete")
    agreement = None
    if original_result is not None:
        # Streaming timings may vary. Scores, every track/stem cell, model
        # identities and reconstruction errors must repeat exactly.
        before, after = original_result["results"][0], report["results"][0]
        agreement = {key: before[key] == after[key] for key in
                     ("checkpoint", "tracks", "aggregate", "reconstruction_max_abs")}
        require(all(agreement.values()), "Repeated quality differs; retain both attempts and investigate")
    verify_inputs({"source_bindings": execution_bindings})
    write(OUT / "result.json", {"status": "training_audit_and_full14_complete", "recovered_validation": True,
          "original_root_actual_exit_code": None, "checkpoint": training["checkpoint"],
          "full_sdr_db": report["results"][0]["aggregate"]["full_sdr_db"], "target_full_sdr_db": 5.0,
          "target_reached": report["target_reached"], "quality_result": {
              "path": str(quality / "result.json"), "sha256": sha(quality / "result.json")},
          "original_attempt_exact_quality_agreement": agreement, "training_repeated": False,
          "plugin_replaced": False, "source_bindings": execution_bindings})
    print(json.dumps({"event": "validation_recovery_complete", "full_sdr_db": report["results"][0]["aggregate"]["full_sdr_db"]}), flush=True)


if __name__ == "__main__":
    main()
