"""Retire one unused synthetic ONNX export, preserving trained and accepted models."""
from __future__ import annotations

import argparse
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_sdr_checkpoint import require_space


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Retirement plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-untrained-mixer-export-retirement-plan-v1"
            and plan["reserve_after_bytes"] == 450_000_000, "Different cleanup scope")
    verify_inputs(plan)
    fixture = PHASE / "sdr-local-mask-mixer-full-onnx-001"
    target = fixture / "active_fixture-full-graph.onnx"
    original_paths = [fixture / name for name in ("plan.json", "result.json", "fixture-execution.json")]
    require(all(plan["source_bindings"].get(str(p)) == sha(p) for p in original_paths), "Unbound fixture evidence")
    original_plan, original, execution = [read(p) for p in original_paths]
    verify_inputs(original_plan)
    require(original["schema"] == "latency58-local-mask-mixer-full-onnx-result-v1" and original["status"] == "pass"
            and original["source_bindings_unchanged"] and original["training_updates_executed"] == 0
            and not original["quality_selected"] and not original["training_parent_selected"]
            and not original["validation_material_used"] and not original["cuda_initialized"]
            and execution["actual_exit_code"] == 0 and not execution["timed_out"] and execution["source_bindings_unchanged"]
            and execution["plan_sha256"] == original["plan_sha256"] == sha(original_paths[0])
            and execution["argv"][execution["argv"].index("-m") + 1] == "research.direct.check_latency58_local_mask_mixer_full_onnx",
            "Target is not the completed untrained export fixture")
    expected = original["graphs"]["active_fixture"]
    require(plan["target"] == expected and expected["path"] == str(target)
            and expected["state_sha256"] == original["fixture_state_sha256"]
            and expected["state_sha256"] != original["preparation_parent_state_sha256"]
            and target.is_file() and not target.is_symlink() and target.stat().st_size == expected["bytes"]
            and sha(target) == expected["sha256"] and str(target) not in plan["source_bindings"], "Different disposable export")
    closure = read(plan["mixer_closure"]["path"])
    require(plan["source_bindings"].get(plan["mixer_closure"]["path"]) == plan["mixer_closure"]["sha256"]
            and closure["schema"] == "latency58-vocal-focus-three-arm-decision-v1"
            and closure["all_three_quality_reviews_complete"] and closure["all_original_pilots_closed_at_updates"] == 250
            and closure["further_updates_to_original_pilots"] == 0 and not closure["quality_selected"],
            "The completed mixer comparison is not closed")
    for binding in plan["retained_input_plans"]:
        require(plan["source_bindings"].get(binding["path"]) == binding["sha256"] == sha(binding["path"]),
                "Unbound retained input plan")
        document = read(binding["path"])
        require(str(target) not in document["source_bindings"], "Synthetic export remains a retained model input")
        verify_inputs(document)
    protected = plan["protected_files"]
    required = {str(fixture / name) for name in ("parent-full-graph.onnx", "untrained-mixer-coefficients.json",
                                                "active_fixture-parity.json", "parent-parity.json")}
    require(required.issubset(protected) and str(target) not in protected
            and all(plan["source_bindings"].get(p) == s == sha(p) for p, s in protected.items()),
            "Incomplete retained evidence or protected model set")
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "receipt.json").exists(), "Preserve retirement record")
    before = require_space(plan, 0)
    require(before - expected["bytes"] + plan["reserve_after_bytes"] < plan["stop_counted_bytes"], "Insufficient planned headroom")
    write(out / "intent.json", {"plan_sha256": args.plan_sha256, "target": expected,
                               "protected_files": protected, "counted_bytes_before": before})
    require(sha(target) == expected["sha256"], "Export changed immediately before retirement")
    target.unlink()
    require(not target.exists() and all(sha(p) == s for p, s in protected.items()), "Protected model or evidence changed")
    verify_inputs(plan)
    for binding in plan["retained_input_plans"]:
        verify_inputs(read(binding["path"]))
    after = require_space(plan, plan["reserve_after_bytes"])
    write(out / "receipt.json", {"schema": "latency58-untrained-mixer-export-retirement-v1", "status": "complete",
          "plan_sha256": args.plan_sha256, "intent_sha256": sha(out / "intent.json"), "retired_export": expected,
          "freed_bytes": expected["bytes"], "protected_files": protected, "source_bindings_unchanged": True,
          "retained_model_inputs_unchanged": True, "counted_bytes_after": after,
          "headroom_before_stop_bytes": plan["stop_counted_bytes"] - after, "reserved_bytes": plan["reserve_after_bytes"],
          "source_audio_deleted": False, "trained_checkpoint_deleted": False, "accepted_model_deleted": False,
          "training_updates_executed": 0,
          "recreation": "The original CPU export implementation, parent state, fixed mixer coefficients and original plan remain retained; re-export requires a fresh output directory and reservation."})
    print({"status": "complete", "freed_bytes": expected["bytes"], "counted_bytes_after": after}, flush=True)


if __name__ == "__main__":
    main()
