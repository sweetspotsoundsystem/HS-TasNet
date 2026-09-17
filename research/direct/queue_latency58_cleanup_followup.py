"""Start one prospectively bounded training block after the current GPU audit.

Plugin qualification and the parallel CPU quality queue are not prerequisites.
This continues an experimental trajectory without selecting deployment weights.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
import json
import os
from pathlib import Path
import select
import subprocess
import time

from research.direct.run_latency58_quality import PHASE, ROOT, PYTHON, read, require, sha, write
from research.direct.train_latency58 import disk_bytes, state_sha256, verify_inputs


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Require frozen CPU preparation")
    prospective = read(args.plan)
    verify_inputs(prospective)
    require(prospective["schema"] == "latency58-cleanup-followup-queue-v1"
            and prospective["maximum_wait_seconds"] == 9000
            and prospective["maximum_new_production_updates"] == 250
            and prospective["maximum_resource_updates"] == 2
            and prospective["training_waits_for_plugin_qualification"] is False
            and prospective["parent_selected_by"] == "prospective_trajectory_endpoint",
            "Different followup scope")
    out = Path(prospective["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "result.json").exists(), "Preserve queue")
    parent_stage = Path(prospective["parent_stage_directory"])
    identity = prospective["supervisor"]
    waited = 0.0
    if not (parent_stage / "audit-execution.json").exists():
        libc = ctypes.CDLL(None, use_errno=True)
        libc.pidfd_open.argtypes = [ctypes.c_int, ctypes.c_uint]
        libc.pidfd_open.restype = ctypes.c_int
        fd = libc.pidfd_open(identity["pid"], 0)
        if fd < 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        try:
            proc = Path("/proc", str(identity["pid"]))
            stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            argv = [x.decode() for x in (proc / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")]
            require(int(stat[19]) == identity["start_ticks"] and argv == identity["argv"]
                    and argv[argv.index("-m") + 1] == "research.direct.run_latency58_cleanup_successor_v2",
                    "Different parent supervisor")
            write(out / "wait-start.json", {"supervisor": identity, "pidfd_opened": True,
                                            "prospective_plan_sha256": args.plan_sha256})
            began = time.monotonic()
            poller = select.poll()
            poller.register(fd, select.POLLIN)
            while not poller.poll(30000):
                require(time.monotonic() - began < prospective["maximum_wait_seconds"], "Wait expired")
            waited = time.monotonic() - began
        finally:
            os.close(fd)
    verify_inputs(prospective)
    parent_binding = prospective["parent_training_plan"]
    require(sha(parent_binding["path"]) == parent_binding["sha256"], "Parent plan changed")
    reference = read(parent_binding["path"])
    verify_inputs(reference)
    execution, audit_execution, audit = (read(parent_stage / name) for name in
                                         ("execution.json", "audit-execution.json", "audit.json"))
    monitor_path = Path(execution["monitor_result"])
    monitor = read(monitor_path)
    require(execution["actual_exit_code"] == audit_execution["actual_exit_code"] == monitor["child_exit_code"] == 0
            and execution["source_bindings_unchanged"] and audit_execution["source_bindings_unchanged"]
            and not audit_execution["timed_out"] and audit["status"] == "pass"
            and audit["source_bindings_unchanged"] and audit["step"] == 250
            and audit["plan_sha256"] == execution["plan_sha256"] == audit_execution["plan_sha256"]
            == parent_binding["sha256"]
            and monitor["status"] == monitor["supervisor_health"] == "pass"
            and monitor["post_exit_quiet_completed"], "Parent did not finish and audit healthily")
    generation = Path(audit["generation"])
    require(generation == Path(reference["run_dir"]) / "checkpoints/step-000250", "Wrong parent endpoint")
    from research.direct.latency58_cleanup_successor_checkpoint_v2 import read_generation
    receipt = read_generation(generation, expected_plan_sha=parent_binding["sha256"])
    require(audit["generation_receipt_sha256"] == sha(generation / "receipt.json")
            and audit["model_state_sha256"] == receipt["model_state_sha256"], "Parent audit differs")
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    payload = torch.load(generation / "model.pt", map_location="cpu", weights_only=True)
    require(state_sha256(payload["model"]) == receipt["model_state_sha256"]
            and not torch.cuda.is_initialized(), "Parent tensor identity or CPU scope differs")
    parent = {"kind": "cleanup_successor", "generation": str(generation), "training_plan": parent_binding,
              "checkpoint": binding(generation / "model.pt"), "audit": binding(parent_stage / "audit.json"),
              "audit_execution": binding(parent_stage / "audit-execution.json"),
              "execution": binding(parent_stage / "execution.json"), "monitor": binding(monitor_path),
              "model_state_sha256": receipt["model_state_sha256"], "provenance": payload["provenance"]}
    require(payload["architecture"] == reference["architecture"], "Parent architecture changed")
    del payload
    config = {**reference["config"], "seed": 20260923, "data_start": 980000}
    require(config == prospective["config"] and receipt["next_sample_index"] == config["data_start"],
            "Followup sample address or schedule differs")
    extra = (146342157 + 111344465 + disk_bytes(ROOT / ".git/lfs") + disk_bytes(ROOT / ".git/objects"))
    counted = sum(disk_bytes(Path(path)) for path in reference["counted_roots"])
    require(extra < 500_000_000 and counted + 400_000_000 < 79_500_000_000,
            "Followup exceeds the combined 80 GB artifact cap")
    prep = PHASE / "cleanup-followup-prep-001"
    require(not prep.exists(), "Preserve followup preparation")
    prep.mkdir()
    decision = {"schema": "latency58-cleanup-followup-decision-v1",
                "status": "train_followup_independent_of_deployment", "config": config, "training_parent": parent,
                "maximum_production_updates": 250, "qualification_blocks_training": False,
                "parent_selected_by": "prospective_trajectory_endpoint", "quality_promotion_claimed": False,
                "prospective_queue_plan": binding(args.plan), "matched_training_effect_claimed": False,
                "same_adam_rng_continuation_claimed": False, "optimizer_initialization": "fresh_adam"}
    write(prep / "decision.json", decision)
    plan = copy.deepcopy(reference)
    for key in ("resource_plan", "full_resource", "full_resource_execution"):
        plan.pop(key, None)
    plan.update(schema="latency58-cleanup-followup-training-v1", config=config, resource_only=True,
                run_dir=str(PHASE / "cleanup-followup-resource-run-001"), parent=parent,
                initialized_model_state_sha256=parent["model_state_sha256"],
                reference_training_plan=parent_binding, preparation_decision=binding(prep / "decision.json"),
                comparison_variable="fresh_adam_followup_with_new_batches", stop_counted_bytes=79_500_000_000,
                source_bindings={**reference["source_bindings"], **prospective["source_bindings"]})
    paths = [args.plan.resolve(), prep / "decision.json", Path(parent_binding["path"]), monitor_path,
             *(parent_stage / name for name in ("execution.json", "audit.json", "audit-execution.json", "stage.json")),
             *(generation / name for name in ("model.pt", "receipt.json", "rng.pt", "metrics.jsonl"))]
    plan["source_bindings"].update({str(path): sha(path) for path in paths})
    from research.direct.latency58_cleanup_followup_checkpoint import validate_recipe, load_parent
    verify_inputs(plan)
    validate_recipe(plan)
    checked = load_parent(plan)
    require(state_sha256(checked.state_dict()) == parent["model_state_sha256"], "Followup loader differs")
    del checked
    write(prep / "resource-plan.json", plan)
    write(out / "wait-complete.json", {"waited_seconds": waited, "parent_model_state_sha256": parent["model_state_sha256"],
                                       "parent_audit": parent["audit"], "followup_plan": binding(prep / "resource-plan.json")})
    command = [PYTHON, "-u", "-m", "research.direct.run_latency58_cleanup_followup",
               "--resource-plan", str(prep / "resource-plan.json"), "--plan-sha256", sha(prep / "resource-plan.json"),
               "--previous-execution", str(parent_stage / "execution.json")]
    write(out / "command.json", {"argv": command, "prospective_plan_sha256": args.plan_sha256})
    print(json.dumps({"event": "start_bounded_followup", "parent_state": parent["model_state_sha256"],
                      "production_updates": 250, "resource_updates": 2}), flush=True)
    began = time.monotonic()
    with (out / "followup-console.log").open("x") as log:
        child = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=9000)
    verify_inputs(prospective)
    write(out / "result.json", {"actual_exit_code": child.returncode, "elapsed_seconds": time.monotonic() - began,
                               "source_bindings_unchanged": True, "quality_promotion_claimed": False,
                               "training_waited_for_plugin_qualification": False,
                               "resource_plan": binding(prep / "resource-plan.json")})
    require(child.returncode == 0, "Followup failed; preserve its monitored evidence")


if __name__ == "__main__":
    main()
