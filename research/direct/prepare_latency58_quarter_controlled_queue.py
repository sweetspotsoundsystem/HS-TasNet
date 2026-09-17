"""Bind the live lower-rate supervisor and freeze one subsequent GPU trial."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.train_latency58 import verify_inputs, disk_bytes


def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervisor-pid", type=int, required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU queue preparation required")
    resource_path = PHASE / "quarter-controlled-prep-001/resource-plan.json"
    previous_resource = PHASE / "cleanup-lr3e6-prep-001/resource-plan.json"
    previous_queue_path = PHASE / "cleanup-lr-sweep-queue-001/plan.json"
    resource, previous_queue = read(resource_path), read(previous_queue_path)
    verify_inputs(resource)
    verify_inputs(previous_queue)
    require(previous_queue["resource_plan"] == binding(previous_resource), "Lower-rate queue changed its training plan")
    from research.direct.latency58_quarter_controlled_checkpoint import validate_recipe
    validate_recipe(resource)
    proc = Path("/proc", str(args.supervisor_pid))
    argv = [part.decode() for part in (proc / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")]
    ticks = int((proc / "stat").read_text().rsplit(")", 1)[1].split()[19])
    require(argv[argv.index("-m") + 1] == "research.direct.queue_latency58_cleanup_lr_sweep"
            and argv[argv.index("--plan") + 1] == str(previous_queue_path)
            and argv[argv.index("--plan-sha256") + 1] == sha(previous_queue_path), "Different live GPU queue")
    counted = sum(disk_bytes(Path(path)) for path in resource["counted_roots"])
    outside = 146342157 + 111344465 + disk_bytes(ROOT / ".git/lfs") + disk_bytes(ROOT / ".git/objects")
    require(outside < 500_000_000 and counted + 1_250_000_000 < 79_500_000_000, "Prospective three-trial storage reserve exceeded")
    out = PHASE / "quarter-controlled-queue-001"
    require(not out.exists(), "Preserve existing queue")
    out.mkdir()
    bindings = {**resource["source_bindings"], **previous_queue["source_bindings"]}
    for path in (resource_path, previous_resource, previous_queue_path, Path(__file__).resolve(),
                 ROOT / "research/direct/queue_latency58_quarter_controlled.py"):
        bindings[str(path)] = sha(path)
    plan = {"schema": "latency58-quarter-controlled-queue-v1", "maximum_wait_seconds": 9000,
            "maximum_new_production_updates": 250, "maximum_resource_updates": 2,
            "training_waits_for_plugin_qualification": False, "training_waits_for_cpu_quality": False,
            "output_directory": str(out), "previous_stage_directory": str(PHASE / "cleanup-lr3e6-to-000250-001"),
            "previous_resource_plan": binding(previous_resource), "previous_queue_plan": binding(previous_queue_path),
            "resource_plan": binding(resource_path), "source_bindings": bindings,
            "supervisor": {"pid": args.supervisor_pid, "start_ticks": ticks, "argv": argv},
            "storage": {"counted_bytes": counted, "outside_counted_roots_bytes": outside,
                        "reserve_for_pending_lr_and_new_trial_bytes": 1_250_000_000,
                        "combined_cap_bytes": 80_000_000_000}}
    verify_inputs(plan)
    write(out / "plan.json", plan)
    print(json.dumps({"status": "prepared", "plan": binding(out / "plan.json"), "supervisor": plan["supervisor"]}), flush=True)


if __name__ == "__main__":
    main()
