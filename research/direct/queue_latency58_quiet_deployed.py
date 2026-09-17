"""Wait for authenticated primary/control results, then score candidate quiet fidelity."""
from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import select
import time

from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, read, require, sha, write, execute
from research.direct.train_latency58 import verify_inputs
from research.direct.report_latency58_sdr import load_completed as load_primary
from research.direct.compare_latency58_quiet_wanted import load_completed as load_quiet
from research.direct.latency58_sdr_checkpoint import require_space


def binding(path):
    return {"path": str(path), "sha256": sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Queue plan changed")
    plan = read(args.plan)
    verify_inputs(plan)
    for key in ("training_plan", "geometry_plan"):
        item = plan[key]
        require(plan["source_bindings"].get(item["path"]) == item["sha256"] == sha(item["path"]),
                "Unbound candidate or physical-geometry plan")
    require(plan["schema"] == "latency58-quiet-deployed-queue-plan-v1"
            and plan["maximum_wait_seconds"] == 9000 and plan["evaluation_timeout_seconds"] == 2400
            and set(plan["owners"]) == {"primary", "ordinary"}, "Different bounded quiet work")
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "result.json").exists(), "Preserve queue")
    primary = PHASE / "controlled-deployed-half-250-full14-001"
    ordinary = PHASE / "quiet-wanted-ordinary-only-001"
    destination = PHASE / "quiet-wanted-controlled-deployed-half-001"
    require(plan["primary_directory"] == str(primary) and plan["ordinary_directory"] == str(ordinary)
            and plan["evaluation_directory"] == str(destination) and not destination.exists(), "Different or existing evaluation")
    terminals = {"primary": primary / "execution.json", "ordinary": ordinary / "execution.json"}
    libc = ctypes.CDLL(None, use_errno=True)
    libc.pidfd_open.argtypes, libc.pidfd_open.restype = [ctypes.c_int, ctypes.c_uint], ctypes.c_int
    descriptors = {}
    began = time.monotonic()
    try:
        for label, owner in plan["owners"].items():
            if terminals[label].is_file():
                continue
            fd = libc.pidfd_open(owner["pid"], 0)
            if fd < 0:
                if terminals[label].is_file():
                    continue
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error))
            descriptors[label] = fd
            proc = Path(f"/proc/{owner['pid']}")
            try:
                stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
                argv = [p.decode() for p in (proc / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")]
            except FileNotFoundError:
                require(terminals[label].is_file(), "Prerequisite ended during identity capture without a receipt")
                continue
            require(int(stat[19]) == owner["start_ticks"] and argv == owner["argv"], "Different prerequisite supervisor")
        write(out / "wait-start.json", {"owners": plan["owners"], "pidfds_opened": list(descriptors),
                                       "plan_sha256": args.plan_sha256, "wall_time": time.time()})
        while not all(path.is_file() for path in terminals.values()):
            require(time.monotonic() - began < plan["maximum_wait_seconds"], "Bounded prerequisite wait expired")
            for label, fd in descriptors.items():
                poller = select.poll()
                poller.register(fd, select.POLLIN)
                require(not poller.poll(0) or terminals[label].is_file(), "Prerequisite supervisor ended without its terminal receipt")
            time.sleep(5)
    finally:
        for fd in descriptors.values():
            os.close(fd)
    verify_inputs(plan)
    evidence = {**plan["source_bindings"], str(args.plan): args.plan_sha256}
    quality, report = load_primary(primary, evidence)
    control = load_quiet(ordinary, evidence)
    training = read(quality["training_plan"]["path"])
    require(quality["schema"] == "latency58-controlled-deployed-parallel-music-plan-v1" and quality["step"] == 250
            and training["teacher_mode"] == "ordinary_only" and training["additional_loss_weight"] == .5
            and quality["training_plan"] == plan["training_plan"]
            and control["model"]["kind"] == "ordinary_only", "Different primary endpoint or control")
    from research.direct.audit_latency58_controlled_deployed_training_match import load_completed_match
    matched = load_completed_match(quality["training_match"], evidence)
    fingerprint = report["results"][0]["model"]["model_state_sha256"]
    require(matched["model_states"] == {"reference": control["model"]["model_state_sha256"], "candidate": fingerprint},
            "Quiet evaluation does not use the authenticated matched pair")
    evidence.update(quality["source_bindings"])
    geometry = read(plan["geometry_plan"]["path"])
    inventory = Path(plan["inventory_directory"])
    require(control["reference_inventory"] == binding(inventory / "result.json"), "Different source support")
    for path in (inventory / "plan.json", inventory / "result.json", inventory / "inventory-execution.json"):
        evidence[str(path)] = sha(path)
    quiet_plan = {"schema": "latency58-quiet-wanted-evaluation-plan-v1", "workers": 2, "track_indices": list(range(14)),
                  "model": {"kind": "controlled_deployed", "prefix": "controlled-deployed-half-250",
                            "model_state_sha256": fingerprint}, "quality_plan": binding(primary / "plan.json"),
                  "quality_result": binding(primary / "result.json"), "manifest": geometry["manifest"], "config": geometry["config"],
                  "reference_inventory": binding(inventory / "result.json"),
                  "inventory_execution": binding(inventory / "inventory-execution.json"),
                  "output_directory": str(destination), "source_bindings": evidence,
                  "counted_roots": training["counted_roots"], "stop_counted_bytes": training["stop_counted_bytes"]}
    verify_inputs(quiet_plan)
    require_space(quiet_plan, 5_000_000)
    destination.mkdir()
    write(destination / "plan.json", quiet_plan)
    write(out / "wait-complete.json", {"waited_seconds": time.monotonic() - began,
                                       "primary_execution": binding(terminals["primary"]),
                                       "ordinary_execution": binding(terminals["ordinary"]), "training_match": quality["training_match"]})
    execute([PYTHON, "-u", "-m", "research.direct.evaluate_latency58_quiet_deployed",
             "--plan", str(destination / "plan.json"), "--plan-sha256", sha(destination / "plan.json")],
            destination, "evaluation", plan["evaluation_timeout_seconds"], evidence,
            {"plan_sha256": sha(destination / "plan.json")})
    result = load_quiet(destination, {})
    verify_inputs(plan)
    write(out / "result.json", {"schema": "latency58-quiet-deployed-queue-v1", "status": "pass",
          "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
          "evaluation": binding(destination / "result.json"), "model": result["model"], "training_updates_executed": 0,
          "quality_selected": False, "human_listening_completed": False})
    print({"status": "pass", "candidate_quiet_fidelity_complete": True}, flush=True)


if __name__ == "__main__":
    main()
