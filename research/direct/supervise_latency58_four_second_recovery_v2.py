"""Own the recovery controller and record its actual exit after monitored work."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from research.direct.run_latency58_quality import ROOT, PYTHON, read, require, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Recovery plan changed")
    plan = read(args.plan)
    root = ROOT / "research/four_second_20260916/branch-four-second-015"
    require(Path(plan["output_directory"]) == root and plan.get("resume_checkpoint") is not None,
            "Require the prepared trajectory recovery")
    argv = [PYTHON, "-u", "-m", "research.direct.resume_latency58_four_second_shared_v2", "--launch-plan", str(args.plan)]
    sources = {**plan["source_bindings"], str(args.plan): args.plan_sha256, str(Path(__file__).resolve()): sha(__file__)}
    began = time.monotonic()
    with (root / "recovery-controller-console-002.log").open("x") as output:
        child = subprocess.Popen(argv, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        write(root / "recovery-root-launch-002.json", {"supervisor_pid": os.getpid(), "controller_pid": child.pid,
            "controller_start_ticks": Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()[19],
            "argv": argv, "plan_sha256": args.plan_sha256, "supervisor_source_sha256": sha(__file__)})
        code = child.wait()
        output.flush(); os.fsync(output.fileno())
    unchanged = all(sha(p) == digest for p, digest in sources.items())
    result = {"actual_exit_code": code, "timed_out": False, "elapsed_seconds": time.monotonic() - began,
        "source_bindings_unchanged": unchanged, "plan_sha256": args.plan_sha256,
        "stage_execution": str(root / "recovery-stage-002/execution.json")}
    if code == 0:
        result.update(stage_execution_sha256=sha(result["stage_execution"]),
                      result_sha256=sha(root / "production-run/result.json"))
    write(root / "recovery-root-execution-002.json", result)
    require(code == 0 and unchanged, "Recovery controller failed")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
