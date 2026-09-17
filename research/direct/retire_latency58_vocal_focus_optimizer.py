"""Retire one reviewed terminal pilot Adam file while retaining inference evidence."""
from __future__ import annotations

import argparse
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.record_latency58_vocal_focus_review import completed_endpoint
from research.direct.latency58_vocal_focus_checkpoint import read_generation, require_space


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Retirement plan or cwd changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-vocal-focus-optimizer-retirement-plan-v1"
            and plan["arm"] in ("original", "focused", "focused_mixer")
            and plan["reserve_after_bytes"] == (0 if plan["arm"] == "focused_mixer" else 350_000_000),
            "Different bounded pilot retirement")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "receipt.json").exists(), "Preserve retirement evidence")
    for key in ("review_plan", "review", "review_execution", "active_protocol"):
        item = plan[key]
        require(plan["source_bindings"].get(item["path"]) == item["sha256"] == sha(item["path"]),
                "Unbound retirement prerequisite")
    review_plan, review, execution = (read(plan[key]["path"]) for key in ("review_plan", "review", "review_execution"))
    verify_inputs(review_plan)
    verify_inputs(review)
    require(review["schema"] == "latency58-vocal-focus-review-v1" and review["status"] == "pass"
            and review["source_bindings_unchanged"] and review["training_closed"]
            and review["completed_step"] == review["original_maximum_step"] == 250
            and review["further_optimizer_updates"] == 0 and review["all_quality_metrics_reviewed"]
            and not review["quality_selected"] and review["review"] == review_plan["review"]
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"]
            and execution["plan_sha256"] == review["plan_sha256"] == plan["review_plan"]["sha256"],
            "Complete quality review has not closed this fixed training schedule")
    require(all(plan["source_bindings"].get(p) == s for p, s in review["source_bindings"].items()),
            "Retirement omitted reviewed evidence")
    training, generation, receipt, _ = completed_endpoint(review_plan)
    require(review["arm"] == plan["arm"] == training["arm"]
            and review["model_state_sha256"] == receipt["model_state_sha256"]
            and review["generation"] == str(generation), "Different reviewed endpoint")
    # Read with the optimizer required before deleting it; the original CPU audit
    # has already checked all saved Adam tensors against this same receipt.
    require(read_generation(generation, expected_plan_sha=review["training_plan"]["sha256"]) == receipt,
            "Terminal Adam or retained generation changed")
    active = read(plan["active_protocol"]["path"])
    verify_inputs(active)
    require(active["schema"] == "latency58-vocal-focus-protocol-v1"
            and active["quality_endpoints"] == [250] and not active["automatic_continuation"]
            and training["matched_protocol"] == plan["active_protocol"], "Different active pilot protocol")
    optimizer = generation / "optimizer.pt"
    expected = plan["optimizer"]
    require(expected["path"] == str(optimizer) and expected["file_binding"] == receipt["files"]["optimizer.pt"],
            "Only this exact terminal Adam file may be retired")
    for document in (plan, review_plan, review, training, active):
        require(str(optimizer) not in document["source_bindings"], "Adam remains an active input")
    protected = {str(generation / name): sha(generation / name)
                 for name in ("model.pt", "rng.pt", "metrics.jsonl", "receipt.json")}
    require(all(plan["source_bindings"].get(p) == s for p, s in protected.items()), "Unbound retained inference state")
    before = require_space(active, 0)
    require(before - expected["file_binding"]["bytes"] + plan["reserve_after_bytes"] < active["stop_counted_bytes"],
            "Retirement would not leave the next pilot checkpoint reserve")
    write(out / "intent.json", {"plan_sha256": args.plan_sha256, "optimizer": expected,
                               "protected_files": protected, "counted_bytes_before": before})
    require(optimizer.is_file() and not optimizer.is_symlink()
            and optimizer.stat().st_size == expected["file_binding"]["bytes"]
            and sha(optimizer) == expected["file_binding"]["sha256"], "Adam changed immediately before retirement")
    optimizer.unlink()
    verify_inputs(plan)
    verify_inputs(review_plan)
    verify_inputs(active)
    require(not optimizer.exists() and all(sha(p) == s for p, s in protected.items()), "Retained state changed")
    after = require_space(active, plan["reserve_after_bytes"])
    write(out / "receipt.json", {
        "schema": "latency58-vocal-focus-optimizer-retirement-v1", "status": "complete",
        "plan_sha256": args.plan_sha256, "intent_sha256": sha(out / "intent.json"),
        "arm": plan["arm"], "review": plan["review"], "retired_path": str(optimizer),
        "freed_bytes": expected["file_binding"]["bytes"], "protected_files": protected,
        "source_bindings_unchanged": True, "active_training_inputs_unchanged": True,
        "counted_bytes_after": after, "headroom_before_stop_bytes": active["stop_counted_bytes"] - after,
        "reserved_bytes": plan["reserve_after_bytes"], "quality_selected": False,
        "training_updates_executed": 0, "next_training_arm_launched": False})
    print({"status": "complete", "arm": plan["arm"], "freed_bytes": expected["file_binding"]["bytes"],
           "counted_bytes_after": after}, flush=True)


if __name__ == "__main__":
    main()
