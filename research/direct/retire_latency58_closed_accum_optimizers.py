"""Retire only the two audited, closed accumulation trials' terminal Adam files."""
from __future__ import annotations

import argparse
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_sdr_accum_checkpoint import read_generation as read_accum
from research.direct.latency58_sdr_drum_accum_checkpoint import read_generation as read_drum
from research.direct.latency58_vocal_focus_checkpoint import require_space

TRIALS = {
    "sdr-accum": ("sdr-accum-terminal-decision-001", read_accum),
    "sdr-drum-accum": ("sdr-drum-accum-1000-terminal-001", read_drum),
}
RETAINED = ("model.pt", "rng.pt", "metrics.jsonl", "receipt.json")


def authenticate_trial(arm, plan, active):
    closure_name, reader = TRIALS[arm]
    closure_dir = PHASE / closure_name
    closure = read(closure_dir / "decision.json")
    binding = closure["training_plan"]
    require(binding["path"] == str(PHASE / (arm + "-prep-001/training-plan.json"))
            and sha(binding["path"]) == binding["sha256"], "Different original training plan")
    training = read(binding["path"])
    run = PHASE / (arm + "-b16-micro4-lr3e5-1000")
    generation = run / "checkpoints/step-001000"
    receipt = reader(generation, expected_plan_sha=binding["sha256"])
    require(training["run_dir"] == str(run) and training["config"]["steps"] == receipt["step"] == 1000,
            "Only the original terminal generation may be retired")
    if arm == "sdr-accum":
        require(closure["schema"] == "latency58-sdr-accum-terminal-decision-v1"
                and closure["status"] == "closed_choose_training_parent"
                and closure["final_step"] == 1000 and closure["scheduled_horizon_complete"]
                and closure["complete_quality_review"]
                and closure["selected_training_parent_state_sha256"] == receipt["model_state_sha256"],
                "Accumulation schedule has not closed after complete review")
    else:
        execution = read(closure_dir / "decision-execution.json")
        require(closure["schema"] == "latency58-sdr-drum-accum-terminal-v1"
                and closure["status"] == "closed_at_planned_limit" and closure["step"] == 1000
                and closure["further_optimizer_updates"] == 0 and not closure["training_plan_extended"]
                and closure["all_track_stem_low_band_sir_absence_probe_dc_actions_reviewed"]
                and closure["model_state_sha256"] == receipt["model_state_sha256"]
                and execution["actual_exit_code"] == 0 and not execution["timed_out"]
                and execution["source_bindings_unchanged"] and closure["source_bindings_unchanged"]
                and execution["plan_sha256"] == closure["plan_sha256"] == sha(closure_dir / "plan.json"),
                "Accumulated drum schedule has not closed after complete review")
    require(not closure["quality_selected"] and not closure["human_listening_completed"],
            "Different historical closure decision")
    stage = PHASE / (arm + "-to-001000-001")
    audit, audit_execution, execution = (read(stage / name)
        for name in ("audit.json", "audit-execution.json", "execution.json"))
    monitor = read(execution["monitor_result"])
    status = read(run / "status.json")
    require(audit["status"] == "pass" and audit["source_bindings_unchanged"]
            and audit["step"] == 1000 and audit["generation"] == str(generation)
            and audit["generation_receipt_sha256"] == sha(generation / "receipt.json")
            and audit["model_state_sha256"] == receipt["model_state_sha256"]
            and audit["parameter_tensors"] == audit["adam_state_pairs"] == 21
            and audit["buffer_tensors"] == 6 and audit["fixed_buffers_unchanged"]
            and audit["batch_identity_rows"] == 1000 and audit["microbatch_identity_rows"] == 4000
            and audit["augmented_examples"] == 16000 and audit["exact_reset_replay"]
            and audit_execution["actual_exit_code"] == execution["actual_exit_code"] == 0
            and not audit_execution["timed_out"] and audit_execution["source_bindings_unchanged"]
            and execution["source_bindings_unchanged"]
            and audit["plan_sha256"] == audit_execution["plan_sha256"] == execution["plan_sha256"] == binding["sha256"]
            and monitor["status"] == monitor["supervisor_health"] == "pass"
            and monitor["child_exit_code"] == 0 and monitor["post_exit_quiet_completed"]
            and status["status"] == "complete" and status["step"] == status["stop_step"] == 1000,
            "Original saved-state audit or monitored terminal completion is incomplete")
    if arm == "sdr-drum-accum":
        require(audit["normalized_drum_objective_journal_verified"], "Original drum objective audit failed")
    optimizer = generation / "optimizer.pt"
    target = {"path": str(optimizer), **receipt["files"]["optimizer.pt"]}
    require(target == plan["targets"][arm] and target["bytes"] == 222_602_662,
            "Different disposable optimizer bytes")
    protected = {str(generation / name): sha(generation / name) for name in RETAINED}
    paths = [closure_dir / "decision.json", Path(binding["path"]), run / "status.json",
             Path(execution["monitor_result"]), *(stage / n for n in ("audit.json", "audit-execution.json", "execution.json"))]
    if arm == "sdr-drum-accum":
        paths += [closure_dir / "plan.json", closure_dir / "decision-execution.json"]
    require(all(plan["source_bindings"].get(str(p)) == sha(p) for p in paths)
            and all(plan["source_bindings"].get(p) == s for p, s in protected.items()),
            "Retirement omitted original audit, closure or retained evidence")
    for document in (plan, training, closure, *active):
        require(str(optimizer) not in document["source_bindings"], "Adam remains an active input")
        verify_inputs(document)
    require(all(plan["source_bindings"].get(p) == s for document in (training, closure)
                for p, s in document["source_bindings"].items()), "Unbound historical closure inputs")
    return target, protected, generation, binding["sha256"], reader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Retirement plan or cwd changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-closed-accum-optimizer-retirement-plan-v1"
            and set(plan["targets"]) == set(TRIALS) and plan["reserve_after_bytes"] == 400_000_000,
            "Different bounded retirement scope")
    verify_inputs(plan)
    active = []
    for binding in plan["active_plans"]:
        require(plan["source_bindings"].get(binding["path"]) == binding["sha256"] == sha(binding["path"]),
                "Unbound active plan")
        document = read(binding["path"])
        verify_inputs(document)
        require(all(plan["source_bindings"].get(p) == s for p, s in document["source_bindings"].items()),
                "Retirement omitted active inputs")
        active.append(document)
    require(len(active) == 3 and active[0]["schema"] == "latency58-vocal-focus-training-v1"
            and active[0]["arm"] == "focused_mixer" and not active[0]["resource_only"]
            and active[1]["schema"] == "latency58-vocal-focus-queued-quality-plan-v1"
            and active[2]["schema"] == "latency58-vocal-focus-views-evaluation-plan-v1",
            "Different current training or quality work")
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "intent.json").exists()
            and not (out / "receipt.json").exists(), "Preserve retirement evidence")
    authenticated = [authenticate_trial(arm, plan, active) for arm in TRIALS]
    before = require_space(plan, 0)
    freed = sum(row[0]["bytes"] for row in authenticated)
    require(before - freed + plan["reserve_after_bytes"] < plan["stop_counted_bytes"],
            "Retirement would not preserve the next work reservation")
    protected = {p: s for row in authenticated for p, s in row[1].items()}
    write(out / "intent.json", {"plan_sha256": args.plan_sha256, "targets": plan["targets"],
                               "protected_files": protected, "counted_bytes_before": before})
    for target, _, _, _, _ in authenticated:
        path = Path(target["path"])
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == target["bytes"]
                and sha(path) == target["sha256"], "Adam changed immediately before retirement")
        path.unlink()
    for _, _, generation, training_sha, reader in authenticated:
        reader(generation, expected_plan_sha=training_sha, require_optimizer=False)
    verify_inputs(plan)
    for document in active:
        verify_inputs(document)
    require(all(sha(p) == s for p, s in protected.items())
            and all(not Path(row[0]["path"]).exists() for row in authenticated), "Retained state changed")
    after = require_space(plan, plan["reserve_after_bytes"])
    write(out / "receipt.json", {
        "schema": "latency58-closed-accum-optimizer-retirement-v1", "status": "complete",
        "plan_sha256": args.plan_sha256, "intent_sha256": sha(out / "intent.json"),
        "retired_files": plan["targets"], "freed_bytes": freed, "protected_files": protected,
        "source_bindings_unchanged": True, "active_inputs_unchanged": True,
        "original_terminal_generations_audited": True, "original_training_schedules_closed": True,
        "counted_bytes_before": before, "counted_bytes_after": after,
        "headroom_before_stop_bytes": plan["stop_counted_bytes"] - after,
        "reserved_bytes": plan["reserve_after_bytes"], "training_updates_executed": 0,
        "quality_selected": False, "current_mixer_optimizer_retired": False,
        "next_training_arm_launched": False})
    print({"status": "complete", "freed_bytes": freed, "counted_bytes_after": after}, flush=True)


if __name__ == "__main__":
    main()
