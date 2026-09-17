"""Check real saved metadata after primary evaluation, without exporting a graph."""
from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import select
import time

from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, read, require, sha, write, execute
from research.direct.report_latency58_sdr import load_completed
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_sdr_checkpoint import require_space


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Metadata workflow changed")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-leader-export-metadata-workflow-plan-v1"
            and plan["maximum_wait_seconds"] == 12000, "Different bounded metadata workflow")
    out, destination = Path(plan["output_directory"]), Path(plan["metadata_directory"])
    require(out.parent == destination.parent == PHASE and out.is_dir()
            and not (out / "result.json").exists() and not destination.exists(), "Preserve metadata output")
    queue = plan["quality_queue_plan"]
    require(plan["source_bindings"].get(queue["path"]) == queue["sha256"] == sha(queue["path"]),
            "Unbound quality queue")
    source = read(queue["path"])
    require(source["schema"] == "latency58-leader-cleanup-queued-quality-plan-v1"
            and source["quality_prefix"] == "leader-cleanup-250", "Different queued candidate")
    primary = PHASE / "leader-cleanup-250-full14-001"
    terminal, owner = primary / "execution.json", plan["execution_owner"]
    began = time.monotonic()
    if not terminal.exists():
        libc = ctypes.CDLL(None, use_errno=True)
        libc.pidfd_open.argtypes, libc.pidfd_open.restype = [ctypes.c_int, ctypes.c_uint], ctypes.c_int
        fd = libc.pidfd_open(owner["pid"], 0)
        if fd < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        try:
            proc = Path(f"/proc/{owner['pid']}")
            stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            argv = [x.decode() for x in (proc / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")]
            require(int(stat[19]) == owner["start_ticks"] and argv == owner["argv"], "Different live quality owner")
            write(out / "wait-start.json", {"execution_owner": owner, "pidfd_opened": True,
                                           "plan_sha256": args.plan_sha256, "wall_time": time.time()})
            poller = select.poll(); poller.register(fd, select.POLLIN)
            while not terminal.exists():
                require(time.monotonic() - began < plan["maximum_wait_seconds"], "Bounded metadata wait expired")
                require(not poller.poll(5000) or terminal.exists(), "Quality owner ended without completed primary execution")
        finally:
            os.close(fd)
    verify_inputs(plan)
    evidence = {**plan["source_bindings"], str(args.plan): args.plan_sha256}
    models = {}
    for kind, prefix in (("counterfactual_candidate", "counterfactual-teacher-ordinary-only-250"),
                         ("controlled_deployed_candidate", "controlled-deployed-half-250"),
                         ("leader_cleanup_candidate", "leader-cleanup-250")):
        directory = PHASE / (prefix + "-full14-001")
        quality, report = load_completed(directory, evidence)
        require(quality["step"] == 250 and report["inputs_unchanged"]
                and len(report["results"][0]["tracks"]) == 14, "Incomplete saved primary endpoint")
        if kind == "leader_cleanup_candidate":
            require(quality["training_plan"] == source["training_plan"], "Metadata candidate differs from queued training")
        path = directory / "plan.json"
        models[kind] = {"path": str(path), "sha256": sha(path)}
    prepared = {"schema": "latency58-leader-export-metadata-plan-v1", "models": models,
                "output_directory": str(destination), "counted_roots": plan["counted_roots"],
                "stop_counted_bytes": plan["stop_counted_bytes"], "source_bindings": evidence}
    require_space(prepared, 2_000_000)
    destination.mkdir(); write(destination / "plan.json", prepared)
    execute([PYTHON, "-u", "-m", "research.direct.check_latency58_leader_export_metadata",
             "--plan", str(destination / "plan.json"), "--plan-sha256", sha(destination / "plan.json")],
            destination, "metadata", 900, evidence, {"plan_sha256": sha(destination / "plan.json")})
    result = read(destination / "result.json")
    require(result["status"] == "pass" and result["invalid_transfer_metadata_rejected"] == 3
            and result["retained_candidate_metadata_exact"] and not result["onnx_export_executed"],
            "Real leader metadata did not pass")
    verify_inputs(plan)
    write(out / "result.json", {"schema": "latency58-leader-export-metadata-workflow-v1", "status": "pass",
          "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
          "metadata_result": {"path": str(destination / "result.json"), "sha256": sha(destination / "result.json")},
          "onnx_export_executed": False, "inference_executed": False, "quality_selected": False})
    print({"status": "pass", "real_saved_metadata_checked": True, "onnx_export_executed": False}, flush=True)


if __name__ == "__main__":
    main()
