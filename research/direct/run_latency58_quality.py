"""Score one audited endpoint with the already executed three-part protocol.

This synchronous CPU launcher reuses the previous evaluation's exact inputs,
replacing only the audited checkpoint. It does not continue training.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / "research/direct/runs/latency58"
PYTHON = "/home/axel/miniforge3/bin/python"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def checkpoint_bindings(checkpoint):
    return {checkpoint["path"]: checkpoint["sha256"],
            checkpoint["audit"]["path"]: checkpoint["audit"]["sha256"],
            checkpoint["audit_execution"]["path"]: checkpoint["audit_execution"]["sha256"]}


def execute(argv, out, stem, timeout, bindings, extra):
    require(all(sha(path) == digest for path, digest in bindings.items()), "Execution inputs changed")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    started = time.monotonic()
    timed_out = False
    with (out / (stem + "-console.log")).open("x") as log:
        child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                code = child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait(timeout=15)
    unchanged = all(sha(path) == digest for path, digest in bindings.items())
    receipt = {"actual_exit_code": code, "timed_out": timed_out,
               "elapsed_seconds": time.monotonic() - started, "argv": argv,
               "source_bindings_unchanged": unchanged, **extra}
    write(out / ("execution.json" if stem == "evaluation" else stem + "-execution.json"), receipt)
    print(json.dumps({"output_directory": str(out), "operation": stem, **receipt}), flush=True)
    require(code == 0 and not timed_out and unchanged, "Evaluation failed; inspect the retained logs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--stage-directory", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--template-prefix", default="pilot1000")
    args = parser.parse_args()
    require(Path.cwd() == ROOT and args.step > 0 and all(
        name and all(c.isalnum() or c in "-_" for c in name)
        for name in (args.output_prefix, args.template_prefix)), "Invalid endpoint or directory prefix")
    stage = args.stage_directory.resolve(strict=True)
    require(stage.is_relative_to(PHASE), "Use an existing stage within this phase")
    audit_path, audit_execution_path = stage / "checkpoint-audit.json", stage / "audit-execution.json"
    audit, audit_execution, execution = read(audit_path), read(audit_execution_path), read(stage / "root-execution.json")
    command = read(stage / "root-command.json")["argv"]
    monitor = read(Path(command[command.index("--output-dir") + 1]) / "result.json")
    latest = read(PHASE / "raw4-b4-lr3e-5-pilot/latest.json")
    resume = PHASE / "raw4-b4-lr3e-5-pilot/resume.pt"
    require(audit["step"] == latest["step"] == args.step and audit["status"] == "pass"
            and execution["actual_exit_code"] == audit_execution["actual_exit_code"] == monitor["child_exit_code"] == 0
            and not audit_execution["timed_out"] and execution["source_bindings_unchanged"]
            and monitor["status"] == monitor["supervisor_health"] == "pass" and monitor["post_exit_quiet_completed"]
            and sha(resume) == latest["resume_sha256"] == audit["resume_sha256"],
            "Require a successfully completed and separately audited current endpoint")
    checkpoint = {"kind": "audited_resume", "path": str(resume), "sha256": sha(resume),
                  "audit": {"path": str(audit_path), "sha256": sha(audit_path)},
                  "audit_execution": {"path": str(audit_execution_path), "sha256": sha(audit_execution_path)}}
    require(sum(path.stat().st_size for path in PHASE.rglob("*") if path.is_file()) + 25_000_000 < 900_000_000,
            "Insufficient phase allowance for the four Actions WAVs and reports")
    prepared = []
    # Validate all three protocols before creating or running any new evaluation.
    for mode in ("actions60", "full14", "probes"):
        template_dir = PHASE / (args.template_prefix + "-" + mode + "-001")
        template_path = template_dir / "plan.json"
        template_execution = read(template_dir / "execution.json")
        require(template_execution["actual_exit_code"] == 0 and not template_execution["timed_out"]
                and template_execution["source_bindings_unchanged"]
                and template_execution["plan_sha256"] == sha(template_path), "Template lacks a successful actual execution")
        template = read(template_path)
        old_checkpoint = checkpoint_bindings(template["checkpoint"])
        require(all(template["source_bindings"].get(path) == digest for path, digest in old_checkpoint.items()),
                "Template checkpoint binding inventory differs")
        unchanged_bindings = {path: digest for path, digest in template["source_bindings"].items()
                              if path not in old_checkpoint}
        require(all(sha(path) == digest for path, digest in unchanged_bindings.items()),
                "Protocol source/input differs; review changes before creating a new protocol")
        out = PHASE / (args.output_prefix + "-" + mode + "-001")
        require(not out.exists(), "Preserve existing evaluation: " + str(out))
        plan = copy.deepcopy(template)
        plan.update(checkpoint=checkpoint, output_directory=str(out),
                    source_bindings={**unchanged_bindings, **checkpoint_bindings(checkpoint),
                                     str(Path(__file__).resolve()): sha(__file__)},
                    protocol_template={"path": str(template_path), "sha256": sha(template_path)})
        if mode != "probes":
            plan["label"] = "hop128_raw4_pilot" + str(args.step)
        prepared.append((mode, out, plan))
    for mode, out, plan in prepared:
        out.mkdir()
        plan_path = out / "plan.json"
        write(plan_path, plan)
        module = "research.direct.evaluate_latency58_probes" if mode == "probes" else "research.direct.evaluate_latency58"
        argv = [PYTHON, "-u", "-m", module, "--plan", str(plan_path), "--plan-sha256", sha(plan_path)]
        print(json.dumps({"event": "evaluate", "mode": mode, "step": args.step,
                          "plan_sha256": sha(plan_path)}), flush=True)
        execute(argv, out, "evaluation", plan["timeout_seconds"], plan["source_bindings"],
                {"plan_sha256": sha(plan_path)})
        if mode != "probes":
            comparator = ROOT / "research/direct/compare_latency58.py"
            bindings = {str(comparator): sha(comparator), str(out / "result.json"): sha(out / "result.json")}
            argv = [PYTHON, "-u", "-m", "research.direct.compare_latency58", "--candidate", str(out), "--mode", mode]
            execute(argv, out, "comparison", 120, bindings, {"source_bindings": bindings})
    print(json.dumps({"event": "quality_bundle_completed", "step": args.step,
                      "quality_selected": False}), flush=True)


if __name__ == "__main__":
    main()
