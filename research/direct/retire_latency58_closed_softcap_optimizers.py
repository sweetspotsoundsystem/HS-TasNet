"""Retire only the audited Adam files of the two formally closed soft-cap trials."""
from __future__ import annotations

import argparse
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_sdr_checkpoint import require_space


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Retirement plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-closed-softcap-optimizer-retirement-plan-v1"
            and len(plan["targets"]) == 2, "Require exactly the two closed soft-cap Adam files")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.is_relative_to(PHASE) and out.is_dir() and not (out / "receipt.json").exists(),
            "Preserve retirement outputs")
    active = read(plan["active_training_plan"]["path"])
    require(sha(plan["active_training_plan"]["path"]) == plan["active_training_plan"]["sha256"],
            "Active training plan changed")
    verify_inputs(active)
    protected, paths, families = {}, [], set()
    for target in plan["targets"]:
        family = target["family"]
        require(family in ("softcap", "softcap-strong") and family not in families, "Duplicate or unsupported trial")
        families.add(family)
        training_binding = target["training_plan"]
        training = read(training_binding["path"])
        require(sha(training_binding["path"]) == training_binding["sha256"]
                and training["schema"] == "latency58-sdr-" + family + "-training-v1", "Different training recipe")
        generation = Path(target["generation"])
        require(generation == Path(training["run_dir"]) / "checkpoints/step-000250"
                and generation.is_relative_to(PHASE), "Unexpected terminal generation")
        receipt, audit = read(generation / "receipt.json"), read(target["audit"])
        audit_execution, closure = read(target["audit_execution"]), read(target["closure_decision"])
        require(receipt["step"] == audit["step"] == 250 and receipt["plan_sha256"] == training_binding["sha256"]
                and audit["status"] == "pass" and audit["source_bindings_unchanged"]
                and audit_execution["actual_exit_code"] == 0 and not audit_execution["timed_out"]
                and audit_execution["source_bindings_unchanged"]
                and audit["plan_sha256"] == audit_execution["plan_sha256"] == receipt["plan_sha256"]
                and audit["generation_receipt_sha256"] == sha(generation / "receipt.json")
                and audit["model_state_sha256"] == receipt["model_state_sha256"]
                and set(receipt["files"]) == {"model.pt", "optimizer.pt", "rng.pt", "metrics.jsonl"},
                "Terminal generation lacks its original independent audit")
        if family == "softcap":
            require(closure["status"] == "stop_both_candidates_without_adoption"
                    and closure["softcap"]["authorized_further_updates"] == 0
                    and closure["softcap"]["training_plan"] == training_binding,
                    "Soft-cap trial is not formally closed")
        else:
            require(closure["status"] == "stop_at_250_without_adoption"
                    and closure["training_plan_sha256"] == training_binding["sha256"],
                    "Stronger soft-cap trial is not formally closed")
        for name, binding in receipt["files"].items():
            path = generation / name
            require(path.is_file() and not path.is_symlink() and path.stat().st_size == binding["bytes"]
                    and sha(path) == binding["sha256"], "Generation bytes changed: " + str(path))
            if name == "optimizer.pt":
                require(str(path) == target["path"] and binding == target["file_binding"]
                        and str(path) not in active["source_bindings"]
                        and str(path) not in plan["source_bindings"], "Adam file is an active or protected input")
                paths.append(path)
            else:
                protected[str(path)] = binding["sha256"]
        protected[str(generation / "receipt.json")] = sha(generation / "receipt.json")
    require(all(plan["source_bindings"].get(p) == s for p, s in protected.items()), "Unbound retained generation")
    before = require_space(active, 0)
    write(out / "intent.json", {"plan_sha256": args.plan_sha256, "targets": plan["targets"],
                               "protected_generations": protected, "counted_bytes_before": before})
    freed = 0
    for path, target in zip(paths, plan["targets"], strict=True):
        require(sha(path) == target["file_binding"]["sha256"], "Adam file changed immediately before retirement")
        freed += path.stat().st_size
        path.unlink()
    verify_inputs(plan)
    verify_inputs(active)
    require(all(not p.exists() for p in paths) and all(sha(p) == s for p, s in protected.items()),
            "Retirement did not preserve every retained generation file")
    after = require_space(active, 350_000_000)
    write(out / "receipt.json", {
        "schema": "latency58-closed-softcap-optimizer-retirement-v1", "status": "complete",
        "plan_sha256": args.plan_sha256, "intent_sha256": sha(out / "intent.json"), "freed_bytes": freed,
        "retired_paths": [str(p) for p in paths], "protected_generations": protected,
        "source_bindings_unchanged": True, "active_training_inputs_unchanged": True,
        "counted_bytes_after": after, "headroom_before_stop_bytes": active["stop_counted_bytes"] - after,
        "accounting_note": "The active training journal can grow between counts; freed bytes are exact file sizes.",
    })
    print({"status": "complete", "freed_bytes": freed, "counted_bytes_after": after}, flush=True)


if __name__ == "__main__":
    main()
