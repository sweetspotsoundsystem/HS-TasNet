"""Retire the closed B4 drum trial's Adam file after the next rehearsal passes."""
from __future__ import annotations

import argparse
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_sdr_drum_v3_checkpoint import read_generation, require_space


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Retirement plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-terminal-drum-optimizer-retirement-plan-v1", "Unsupported retirement")
    verify_inputs(plan)
    closure = read(plan["closure_decision"]["path"])
    require(sha(plan["closure_decision"]["path"]) == plan["closure_decision"]["sha256"]
            and closure["schema"] == "latency58-sdr-drum-terminal-decision-v3"
            and closure["status"] == "closed_choose_training_parent" and closure["scheduled_horizon_complete"]
            and closure["final_step"] == 1000 and closure["authorized_further_updates_in_this_trial"] == 0,
            "Require explicit closure at the original horizon")
    target = closure["storage_at_decision"]["terminal_drum_optimizer_retirement_permitted_after_this_closure"]
    optimizer = Path(target["path"])
    training_binding = closure["training_plan"]
    training = read(training_binding["path"])
    require(sha(training_binding["path"]) == training_binding["sha256"]
            and optimizer == Path(training["run_dir"]) / "checkpoints/step-001000/optimizer.pt"
            and optimizer.is_relative_to(PHASE) and not optimizer.is_symlink(), "Unexpected disposable file")
    generation = optimizer.parent
    receipt = read_generation(generation, expected_plan_sha=training_binding["sha256"])
    audit, audit_execution = read(plan["audit"]), read(plan["audit_execution"])
    require(audit["status"] == "pass" and audit["source_bindings_unchanged"] and audit["step"] == 1000
            and audit["generation_receipt_sha256"] == sha(generation / "receipt.json")
            and audit_execution["actual_exit_code"] == 0 and not audit_execution["timed_out"]
            and audit_execution["source_bindings_unchanged"]
            and audit["plan_sha256"] == audit_execution["plan_sha256"] == training_binding["sha256"],
            "Original terminal generation lacks its independent audit")
    resource_plan = read(plan["resource_plan"]["path"])
    require(sha(plan["resource_plan"]["path"]) == plan["resource_plan"]["sha256"]
            and resource_plan["schema"] == "latency58-sdr-drum-accum-training-v1"
            and resource_plan["resource_only"], "Different follow-up rehearsal")
    verify_inputs(resource_plan)
    resource, execution = read(plan["resource_result"]), read(plan["resource_execution"])
    monitor = read(execution["monitor_result"])
    require(resource["status"] == "pass" and resource["source_bindings_unchanged"]
            and resource["training_updates_executed"] == 2 and not resource["checkpoint_written"]
            and resource["initial_model_state_sha256"] == closure["selected_training_parent_state_sha256"]
            and execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
            and execution["plan_sha256"] == resource["plan_sha256"] == plan["resource_plan"]["sha256"]
            and monitor["status"] == monitor["supervisor_health"] == "pass"
            and monitor["child_exit_code"] == 0 and monitor["post_exit_quiet_completed"],
            "Require a healthy completed rehearsal from the selected parent")
    require(sha(optimizer) == target["sha256"] == receipt["files"]["optimizer.pt"]["sha256"]
            and optimizer.stat().st_size == target["bytes"] == receipt["files"]["optimizer.pt"]["bytes"]
            and str(optimizer) not in plan["source_bindings"]
            and str(optimizer) not in resource_plan["source_bindings"], "Adam bytes changed or remain active")
    protected = {str(generation / name): sha(generation / name)
                 for name in ("model.pt", "rng.pt", "metrics.jsonl", "receipt.json")}
    require(all(plan["source_bindings"].get(p) == s for p, s in protected.items()), "Unbound retained generation")
    out = Path(plan["output_directory"])
    require(out.is_dir() and out.is_relative_to(PHASE) and not (out / "receipt.json").exists(), "Preserve retirement")
    write(out / "intent.json", {"plan_sha256": args.plan_sha256, "target": target, "protected_files": protected})
    optimizer.unlink()
    require(read_generation(generation, expected_plan_sha=training_binding["sha256"], require_optimizer=False) == receipt,
            "Retained inference generation changed")
    verify_inputs(plan)
    verify_inputs(resource_plan)
    counted = require_space(resource_plan, 350_000_000)
    write(out / "receipt.json", {
        "schema": "latency58-terminal-drum-optimizer-retirement-v1", "status": "complete",
        "plan_sha256": args.plan_sha256, "intent_sha256": sha(out / "intent.json"),
        "freed_bytes": target["bytes"], "retired_path": str(optimizer), "protected_files": protected,
        "source_bindings_unchanged": True, "followup_inputs_unchanged": True,
        "counted_bytes_after": counted, "headroom_before_stop_bytes": resource_plan["stop_counted_bytes"] - counted,
    })
    print({"status": "complete", "freed_bytes": target["bytes"], "counted_bytes_after": counted}, flush=True)


if __name__ == "__main__":
    main()
