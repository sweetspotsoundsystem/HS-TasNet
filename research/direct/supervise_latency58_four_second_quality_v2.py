"""Wait for actual recovery completion, then own the CPU quality controller."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from research.direct.run_latency58_quality import ROOT, PYTHON, read, require, sha, write
from research.direct.latency58_four_second_monitor import require_monitor_closed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    root = ROOT / "research/four_second_20260916/branch-four-second-015"
    require(Path.cwd() == ROOT and args.plan.resolve() == root / "plan-recovery002.json"
            and sha(args.plan) == args.plan_sha256 and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require the prepared recovery and CUDA-hidden CPU1 environment")
    source = ROOT / "research/direct/run_latency58_four_second_recovery_quality_v2.py"
    lineage = ROOT / "research/direct/latency58_four_second_recovery_lineage.py"
    bindings = {str(p): sha(p) for p in (args.plan, source, lineage, Path(__file__).resolve())}
    argv = [PYTHON, "-u", "-m", "research.direct.run_latency58_four_second_recovery_quality_v2",
            "--training-plan", str(args.plan)]
    command_path = root / "quality-command-002.json"
    require(not command_path.exists() and not (root / "quality-plan.json").exists(), "Preserve quality execution")
    write(command_path, {"argv": argv, "source_bindings": bindings, "maximum_wait_seconds": 13500,
                        "supervisor_pid": os.getpid(), "plan_sha256": args.plan_sha256})
    began, code, status, error = time.monotonic(), None, "waiting_for_training", None
    try:
        receipt = root / "recovery-root-execution-002.json"
        while not receipt.exists():
            require(time.monotonic() - began < 13500, "Recovery completion receipt did not arrive")
            time.sleep(10)
        completed = read(receipt)
        require(completed["actual_exit_code"] == 0 and completed["source_bindings_unchanged"]
                and completed["plan_sha256"] == args.plan_sha256,
                "Training controller did not complete successfully")
        execution = read(root / "recovery-stage-002/execution.json")
        require_monitor_closed(execution, read(execution["monitor_result"]), final_step=2000)
        require(all(sha(p) == digest for p, digest in bindings.items()), "Queued quality sources changed")
        status = "evaluating"
        with (root / "quality-controller-console-002.log").open("x") as log:
            child = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            write(root / "quality-launch-002.json", {"pid": child.pid, "argv": argv,
                "start_ticks": Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()[19],
                "command_sha256": sha(command_path)})
            code = child.wait()
            log.flush(); os.fsync(log.fileno())
        require(code == 0, "Saved quality controller failed")
        require(all(sha(p) == digest for p, digest in bindings.items()), "Quality sources changed during evaluation")
        status = "pass"
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        result = {"status": status, "error": error, "actual_exit_code": code,
            "elapsed_seconds": time.monotonic() - began, "plan_sha256": args.plan_sha256,
            "command_sha256": sha(command_path), "source_bindings_unchanged":
                all(sha(p) == digest for p, digest in bindings.items())}
        if status == "pass":
            result["result_sha256"] = sha(root / "result.json")
        write(root / "quality-root-execution-002.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
