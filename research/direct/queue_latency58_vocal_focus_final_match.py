"""Wait on the exact mixer supervisor, then authenticate all three pilots."""
from __future__ import annotations

import argparse
import ctypes
from pathlib import Path
import select
import time

from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, read, require, sha, write, execute
from research.direct.train_latency58 import verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Queue plan or cwd changed")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-vocal-focus-final-match-queue-plan-v1"
            and plan["maximum_wait_seconds"] == 1800 and plan["suffix"] == "001"
            and not plan["training_updates_authorized"] and not plan["quality_selection_authorized"],
            "Different matching queue scope")
    out, stage = Path(plan["output_directory"]), Path(plan["stage_directory"])
    expected_result = PHASE / "vocal-focus-training-match-250-001/result.json"
    require(out.parent == stage.parent == PHASE and out.is_dir() and not (out / "result.json").exists()
            and not expected_result.parent.exists(), "Preserve matching evidence")
    identity = plan["supervisor"]
    waited = 0.0
    if not ((stage / "execution.json").is_file() and (stage / "audit-execution.json").is_file()):
        libc = ctypes.CDLL(None, use_errno=True)
        libc.pidfd_open.argtypes = [ctypes.c_int, ctypes.c_uint]
        libc.pidfd_open.restype = ctypes.c_int
        descriptor = libc.pidfd_open(identity["pid"], 0)
        require(descriptor >= 0, "Cannot observe the exact live mixer supervisor")
        try:
            proc = Path("/proc") / str(identity["pid"])
            command = [x.decode() for x in (proc / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")]
            stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            require(int(stat[19]) == identity["start_ticks"] and command == identity["argv"]
                    and command[command.index("-m") + 1] == "research.direct.run_latency58_vocal_focus_stage"
                    and command[command.index("--plan") + 1] == plan["training_plan"]["path"]
                    and command[command.index("--output-directory") + 1] == str(stage), "Different live supervisor")
            write(out / "wait-start.json", {"pidfd_opened": True, "supervisor": identity,
                                            "plan_sha256": args.plan_sha256, "wall_time": time.time()})
            print({"event": "waiting_for_exact_mixer_supervisor", "pid": identity["pid"]}, flush=True)
            began = time.monotonic()
            poller = select.poll()
            poller.register(descriptor, select.POLLIN)
            while not poller.poll(30000):
                require(time.monotonic() - began < plan["maximum_wait_seconds"], "Bounded matching wait expired")
            waited = time.monotonic() - began
        finally:
            import os
            os.close(descriptor)
    execution, audited, audit = (read(stage / name) for name in ("execution.json", "audit-execution.json", "audit.json"))
    monitor = read(execution["monitor_result"])
    require(execution["actual_exit_code"] == audited["actual_exit_code"] == 0
            and execution["source_bindings_unchanged"] and audited["source_bindings_unchanged"]
            and not audited["timed_out"] and audit["status"] == "pass" and audit["source_bindings_unchanged"]
            and audit["arm"] == "focused_mixer" and audit["step"] == 250
            and execution["plan_sha256"] == audited["plan_sha256"] == audit["plan_sha256"] == plan["training_plan"]["sha256"]
            and monitor["status"] == monitor["supervisor_health"] == "pass"
            and monitor["child_exit_code"] == 0 and monitor["post_exit_quiet_completed"],
            "Mixer training or saved-state audit failed; final matching is not launched")
    verify_inputs(plan)
    write(out / "wait-complete.json", {"waited_seconds": waited, "actual_training_exit_code": 0,
                                       "training_execution_sha256": sha(stage / "execution.json"),
                                       "audit_execution_sha256": sha(stage / "audit-execution.json")})
    execute([PYTHON, "-u", "-m", "research.direct.run_latency58_vocal_focus_comparison",
             "training-match", "--suffix", "001"], out, "training-match", 480,
            plan["source_bindings"], {"queue_plan_sha256": args.plan_sha256})
    result = read(expected_result)
    inner_execution = read(expected_result.parent / "match-execution.json")
    require(result["status"] == "pass" and result["source_bindings_unchanged"]
            and result["updates_per_arm"] == 250 and result["final_rng_states_exact"]
            and inner_execution["actual_exit_code"] == 0 and not inner_execution["timed_out"]
            and inner_execution["source_bindings_unchanged"], "Final matching did not complete")
    verify_inputs(plan)
    write(out / "result.json", {"schema": "latency58-vocal-focus-final-match-queue-v1", "status": "pass",
          "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
          "match_result": {"path": str(expected_result), "sha256": sha(expected_result)},
          "match_execution": {"path": str(expected_result.parent / "match-execution.json"),
                              "sha256": sha(expected_result.parent / "match-execution.json")},
          "training_updates_executed": 0, "quality_selected": False})
    print({"event": "final_training_match_complete", "result": str(expected_result)}, flush=True)


if __name__ == "__main__":
    main()
