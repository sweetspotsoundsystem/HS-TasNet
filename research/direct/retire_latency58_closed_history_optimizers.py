"""Retire two reviewed terminal history Adam files after matched vocal rehearsals."""
from __future__ import annotations

import argparse
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_sdr_history_checkpoint import read_generation, require_space


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Retirement plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-closed-history-optimizer-retirement-plan-v1"
            and set(plan["targets"]) == {"short", "long"}, "Require exactly two terminal history Adam files")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.is_dir() and out.is_relative_to(PHASE) and not (out / "receipt.json").exists(), "Preserve retirement")
    for key in ("closure_review", "closure_execution", "resource_pairing", "resource_pairing_execution", "active_protocol"):
        item = plan[key]
        require(plan["source_bindings"].get(item["path"]) == item["sha256"] == sha(item["path"]),
                "Unbound retirement prerequisite")
    closure, closure_execution = (read(plan[k]["path"]) for k in ("closure_review", "closure_execution"))
    require(closure["schema"] == "latency58-history-500-review-v1" and closure["status"] == "pass"
            and closure["source_bindings_unchanged"] and closure_execution["actual_exit_code"] == 0
            and not closure_execution["timed_out"] and closure_execution["source_bindings_unchanged"]
            and closure_execution["plan_sha256"] == closure["plan_sha256"], "History review did not complete")
    pairing, pairing_execution = (read(plan[k]["path"]) for k in ("resource_pairing", "resource_pairing_execution"))
    require(pairing["schema"] == "latency58-vocal-focus-resource-pairing-v1" and pairing["status"] == "pass"
            and pairing["source_bindings_unchanged"] and pairing["updates_per_arm"] == 2
            and pairing["all_three_pristine_and_original_draws_exact"]
            and pairing["focused_inputs_and_teacher_targets_exact"] and pairing["initial_inherited_gradients_exact"]
            and pairing_execution["actual_exit_code"] == 0 and not pairing_execution["timed_out"]
            and pairing_execution["source_bindings_unchanged"]
            and pairing_execution["plan_sha256"] == pairing["plan_sha256"], "Matched vocal rehearsals have not passed")
    verify_inputs(pairing)
    active = read(plan["active_protocol"]["path"])
    verify_inputs(active)
    require(active["schema"] == "latency58-vocal-focus-protocol-v1" and active["quality_endpoints"] == [250]
            and not active["automatic_continuation"], "Different new pilot protocol")
    protected, paths = {}, []
    for arm in ("short", "long"):
        target, reviewed = plan["targets"][arm], closure["arms"][arm]
        training_binding = target["training_plan"]
        require(sha(training_binding["path"]) == training_binding["sha256"]
                == reviewed["training_plan_sha256"], "Different reviewed history arm")
        training = read(training_binding["path"])
        generation = Path(target["generation"])
        require(training["schema"] == "latency58-sdr-history-training-v1"
                and training["config"]["steps"] == training["continuation_rules"]["maximum_step"] == 500
                and reviewed["training_closed"] and reviewed["completed_step"] == reviewed["original_maximum_step"] == 500
                and reviewed["further_optimizer_updates"] == 0
                and generation == Path(training["run_dir"]) / "checkpoints/step-000500"
                and generation.is_relative_to(PHASE), "History schedule is not closed at its original horizon")
        status = read(Path(training["run_dir"]) / "status.json")
        require(status["status"] == "complete" and status["step"] == 500, "History process did not finish")
        receipt = read_generation(generation, expected_plan_sha=training_binding["sha256"])
        audit, execution = read(target["audit"]), read(target["audit_execution"])
        require(audit["status"] == "pass" and audit["source_bindings_unchanged"]
                and audit["step"] == receipt["step"] == 500
                and audit["plan_sha256"] == execution["plan_sha256"] == training_binding["sha256"]
                and audit["generation_receipt_sha256"] == sha(generation / "receipt.json")
                and audit["model_state_sha256"] == receipt["model_state_sha256"] == reviewed["model_state_sha256"]
                and execution["actual_exit_code"] == 0 and not execution["timed_out"]
                and execution["source_bindings_unchanged"], "Terminal history state lacks its original saved-state audit")
        optimizer = generation / "optimizer.pt"
        require(str(optimizer) == target["path"] and receipt["files"]["optimizer.pt"] == target["file_binding"]
                and str(optimizer) not in active["source_bindings"]
                and str(optimizer) not in plan["source_bindings"]
                and str(optimizer) not in pairing["source_bindings"], "Adam file is still an active input")
        paths.append((optimizer, target["file_binding"]))
        for name in ("model.pt", "rng.pt", "metrics.jsonl", "receipt.json"):
            path = str(generation / name)
            protected[path] = sha(path)
    require(all(plan["source_bindings"].get(p) == digest for p, digest in protected.items()), "Unbound retained state")
    before = require_space(active, 0)
    require(before - sum(item["bytes"] for _, item in paths) + 650_000_000 < active["stop_counted_bytes"],
            "Retiring these terminal Adam files would not reserve enough pilot space")
    write(out / "intent.json", {"plan_sha256": args.plan_sha256, "targets": plan["targets"],
                               "protected_files": protected, "counted_bytes_before": before})
    freed = 0
    for path, expected in paths:
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == expected["bytes"]
                and sha(path) == expected["sha256"], "Adam changed immediately before retirement")
        path.unlink()
        freed += expected["bytes"]
    verify_inputs(plan)
    verify_inputs(active)
    verify_inputs(pairing)
    require(all(not p.exists() for p, _ in paths) and all(sha(p) == s for p, s in protected.items()),
            "Retirement failed or changed retained inference state")
    after = require_space(active, 650_000_000)
    write(out / "receipt.json", {
        "schema": "latency58-closed-history-optimizer-retirement-v1", "status": "complete",
        "plan_sha256": args.plan_sha256, "intent_sha256": sha(out / "intent.json"),
        "retired_paths": [str(p) for p, _ in paths], "freed_bytes": freed, "protected_files": protected,
        "source_bindings_unchanged": True, "active_training_inputs_unchanged": True,
        "counted_bytes_after": after, "headroom_before_stop_bytes": active["stop_counted_bytes"] - after,
        "new_pilot_combined_artifact_reservation_bytes": 650_000_000,
        "reservation_requires": "Serialize pilots and retire each terminal Adam only after its saved-state audit and full quality review close the arm."})
    print({"status": "complete", "freed_bytes": freed, "counted_bytes_after": after}, flush=True)


if __name__ == "__main__":
    main()
