"""Freeze two rehearsals changing only participation in the teacher loss."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs, state_sha256
from research.direct.latency58_counterfactual_checkpoint import validate_recipe, load_parent, require_space
from research.direct.latency58_counterfactual_teacher import VERSION


def binding(path):
    return {"path": str(path), "sha256": sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require frozen CPU1 preparation")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-counterfactual-resource-preparation-plan-v1", "Different preparation")
    for item in (plan["decision"], plan["journal_check"], plan["journal_check_execution"]):
        require(plan["source_bindings"].get(item["path"]) == item["sha256"] == sha(item["path"]), "Unbound prerequisite")
    decision = read(plan["decision"]["path"])
    verify_inputs(decision)
    require(decision["status"] == "prepare_selective_distillation" and decision["all_three_quality_reviews_complete"]
            and not decision["quality_selected"] and decision["next_trial"]["resource_updates_per_mode"] == 2
            and decision["next_trial"]["new_production_updates_maximum"] == 250, "Different reviewed experiment")
    checked, execution = read(plan["journal_check"]["path"]), read(plan["journal_check_execution"]["path"])
    require(checked["status"] == "pass" and checked["invalid_journals_rejected"] == 5
            and checked["detached_logging_preserves_outputs_and_gradients"] and checked["source_bindings_unchanged"]
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"] and execution["plan_sha256"] == checked["plan_sha256"],
            "Journal qualification is incomplete")
    control_binding = decision["artifacts"]["focused_control_training"]
    control = read(control_binding["path"])
    verify_inputs(control)
    require(control["config"] == decision["next_trial"]["config"] and control["arm"] == "focused"
            and not control["resource_only"] and not control["local_mask_mixer"], "Different original control")
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "protocol.json").exists(), "Preserve preparation")
    sources = {**plan["source_bindings"], str(args.plan): args.plan_sha256}
    before = require_space(control, 10_000_000)
    protocol = {"schema": "latency58-counterfactual-protocol-v1", "source_bindings": sources,
                "preparation_decision": plan["decision"], "matched_control_training_plan": control_binding,
                "teacher_modes": ["all_views", "ordinary_only"], "resource_updates_per_mode": 2,
                "production_teacher_modes": ["ordinary_only"], "maximum_production_updates": 250,
                "quality_endpoints": [250], "automatic_continuation": False,
                "counterfactual_version": VERSION, "config": control["config"],
                **{k: control[k] for k in ("counted_roots", "stop_counted_bytes")}}
    write(out / "protocol.json", protocol)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rng = torch.get_rng_state().clone()
    prepared = {}
    for mode in ("all_views", "ordinary_only"):
        candidate = copy.deepcopy(control)
        for key in ("resource_plan", "full_resource", "full_resource_execution", "resource_pairing", "resource_pairing_execution"):
            candidate.pop(key, None)
        candidate.update(schema="latency58-counterfactual-training-v1", resource_only=True,
                         teacher_mode=mode, counterfactual_version=VERSION,
                         matched_control_training_plan=control_binding, preparation_decision=plan["decision"],
                         matched_protocol=binding(out / "protocol.json"),
                         run_dir=str(PHASE / ("counterfactual-teacher-" + mode.replace("_", "-") + "-resource-run-001")),
                         source_bindings={**control["source_bindings"], **sources,
                                          str(out / "protocol.json"): sha(out / "protocol.json")})
        candidate["functional_proofs"].append({"result": plan["journal_check"]["path"],
                                              "execution": plan["journal_check_execution"]["path"]})
        validate_recipe(candidate)
        verify_inputs(candidate)
        require(not Path(candidate["run_dir"]).exists(), "Preserve previous rehearsal")
        model = load_parent(candidate)
        require(state_sha256(model.state_dict()) == candidate["initialized_model_state_sha256"], "Parent replay differs")
        del model
        path = out / (mode + "-resource-plan.json")
        write(path, candidate)
        prepared[mode] = binding(path)
    require(torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized(), "Preparation changed RNG or CUDA")
    verify_inputs(plan)
    write(out / "result.json", {"schema": "latency58-counterfactual-resource-preparation-v1", "status": "pass",
          "plan_sha256": args.plan_sha256, "source_bindings": sources, "source_bindings_unchanged": True,
          "protocol": binding(out / "protocol.json"), "plans": prepared, "both_parent_replays_exact": True,
          "cpu_rng_unchanged": True, "cuda_initialized": False, "training_updates_executed": 0,
          "quality_selected": False, "counted_bytes_before": before, "counted_bytes_after": require_space(control, 0)})
    print({"status": "pass", "plans": prepared}, flush=True)


if __name__ == "__main__":
    main()
