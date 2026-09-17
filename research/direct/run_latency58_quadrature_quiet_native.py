"""Run the declared saved-graph comparison after training and scoring have closed."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs


def require_no_training_or_scoring():
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        for argument in argv:
            name = argument.rsplit(b"/", 1)[-1]
            require(not name.startswith((b"train_latency58", b"evaluate_latency58",
                                          b"research.direct.train_latency58", b"research.direct.evaluate_latency58")),
                    "A latency58 training or scoring process is still active")


def main():
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Use the CPU workspace")
    intent_path = PHASE / "m4-quadrature-candidates-quiet-native-intent-001/plan.json"
    require(sha(intent_path) == "1d1fab42fdb341d6667940df9b15b5a9e2c17b4b350b45561d9479aa73e78ec9",
            "The declared native comparison changed")
    intent = read(intent_path)
    verify_inputs(intent)
    previous = PHASE / "quadrature-continuation-001"
    plan_path = previous / "plan.json"
    plan = read(plan_path)
    verify_inputs(plan)
    result, review = read(previous / "result.json"), read(previous / "selection-review.json")
    require(result["status"] == "training_audit_and_full14_complete" and review["actual_root_exit_code"] == 0
            and bool(review["actual_root_completion_evidence"]), "Require the completed root and reviewed full14 result")
    verify_inputs(review)
    for name in ("production-stage/execution.json", "full14/execution.json"):
        execution = read(previous / name)
        require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"],
                "Previous training or scoring did not close successfully")
    lock = (previous / "production-run/trainer.lock").open("r")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require_no_training_or_scoring()
    counted = require_space(plan, intent["extra_reservation_bytes"])
    paths = [intent_path, Path(__file__).resolve(), plan_path, *map(Path, intent["required_completed_before_launch"])]
    bindings = {**intent["source_bindings"], **{str(path): sha(path) for path in paths}}
    verify_inputs({"source_bindings": bindings})
    out, stage = Path(intent["output_directory"]), Path(intent["stage_directory"])
    require(not out.exists() and not stage.exists(), "Preserve existing timing evidence")
    stage.mkdir()
    write(stage / "command.json", {"argv": intent["argv"], "source_bindings": bindings,
          "counted_bytes_before": counted, "our_training_and_scoring_active_at_start": False,
          "other_user_programs_idle_claimed": False})
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    began, timed_out = time.monotonic(), False
    with (stage / "console.log").open("x") as log:
        child = subprocess.Popen(intent["argv"], cwd=ROOT, env=environment, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=900)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                code = child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait(timeout=15)
    unchanged = all(sha(path) == digest for path, digest in bindings.items())
    require_no_training_or_scoring()
    write(stage / "execution.json", {"actual_exit_code": code, "timed_out": timed_out,
          "elapsed_seconds": time.monotonic() - began, "source_bindings_unchanged": unchanged,
          "intent_sha256": sha(intent_path), "our_training_and_scoring_active_at_end": False,
          "m4_measured": False, "plugin_host_qualified": False})
    require(code == 0 and not timed_out and unchanged, "Native comparison failed; inspect its retained logs")
    require_space(plan, 0)
    print(json.dumps(read(stage / "execution.json")), flush=True)


if __name__ == "__main__":
    main()
