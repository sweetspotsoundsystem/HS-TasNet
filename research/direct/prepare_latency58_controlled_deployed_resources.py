"""Freeze zero- and half-weight rehearsals matched to the completed ordinary-only trial."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs, state_sha256
from research.direct.latency58_controlled_deployed_checkpoint import validate_recipe, load_parent, require_space
from research.direct.latency58_controlled_deployed_loss import VERSION


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
    require(plan["schema"] == "latency58-controlled-deployed-resource-preparation-plan-v1", "Different preparation")
    for item in (plan["decision"], plan["model_check"], plan["model_check_execution"]):
        require(plan["source_bindings"].get(item["path"]) == item["sha256"] == sha(item["path"]), "Unbound prerequisite")
    decision = read(plan["decision"]["path"])
    verify_inputs(decision)
    require(decision["status"] == "prepare_matched_controlled_deployed_truth_pilot"
            and decision["additional_loss_version"] == VERSION and decision["additional_loss_weight"] == .5
            and decision["resource_control_weight"] == 0 and decision["maximum_production_updates"] == 250
            and not decision["quality_selected"] and not decision["goal_complete"], "Different reviewed experiment")
    checked, execution = read(plan["model_check"]["path"]), read(plan["model_check_execution"]["path"])
    require(checked["status"] == "pass" and checked["invalid_journals_rejected"] == 7
            and checked["zero_weight_loss_and_all_21_parameter_gradients_exact"]
            and checked["forward_outputs_exact"] and checked["half_weight_changes_parameter_gradients"]
            and checked["detached_logging_preserves_model_gradients"] and checked["source_bindings_unchanged"]
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"] and execution["plan_sha256"] == checked["plan_sha256"],
            "Model and journal qualification incomplete")
    control_binding = decision["matched_control_training_plan"]
    control = read(control_binding["path"])
    verify_inputs(control)
    require(control["config"] == decision["config"] and not control["resource_only"]
            and control["teacher_mode"] == "ordinary_only" and not control["local_mask_mixer"], "Different control")
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "protocol.json").exists(), "Preserve preparation")
    sources = {**plan["source_bindings"], str(args.plan): args.plan_sha256}
    before = require_space(control, 450_000_000)
    protocol = {"schema": "latency58-controlled-deployed-protocol-v1", "source_bindings": sources,
                "preparation_decision": plan["decision"], "matched_ordinary_training_plan": control_binding,
                "additional_loss_weights": {"zero": 0, "half": .5}, "additional_loss_version": VERSION,
                "resource_updates_per_mode": 2, "production_additional_loss_weights": [.5],
                "maximum_production_updates": 250, "quality_endpoints": [250], "automatic_continuation": False,
                "config": control["config"], "reference_resource": decision["reference_resource"],
                **{k: control[k] for k in ("counted_roots", "stop_counted_bytes")}}
    write(out / "protocol.json", protocol)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rng = torch.get_rng_state().clone()
    prepared = {}
    for name, weight in (("zero", 0), ("half", .5)):
        candidate = copy.deepcopy(control)
        for key in ("resource_plan", "full_resource", "full_resource_execution", "resource_pairing", "resource_pairing_execution"):
            candidate.pop(key, None)
        candidate.update(schema="latency58-controlled-deployed-training-v1", resource_only=True,
                         additional_loss_weight=weight, additional_loss_version=VERSION,
                         matched_ordinary_training_plan=control_binding, preparation_decision=plan["decision"],
                         reference_resource=decision["reference_resource"],
                         matched_protocol=binding(out / "protocol.json"),
                         run_dir=str(PHASE / ("controlled-deployed-" + name + "-resource-run-001")),
                         source_bindings={**control["source_bindings"], **sources,
                                          str(out / "protocol.json"): sha(out / "protocol.json")})
        functional = decision["artifacts"]["loss_functional"]
        candidate["functional_proofs"].extend([
            {"result": functional["result.json"]["path"], "execution": functional["functional-execution.json"]["path"]},
            {"result": plan["model_check"]["path"], "execution": plan["model_check_execution"]["path"]}])
        validate_recipe(candidate)
        verify_inputs(candidate)
        require(not Path(candidate["run_dir"]).exists(), "Preserve previous rehearsal")
        model = load_parent(candidate)
        require(state_sha256(model.state_dict()) == candidate["initialized_model_state_sha256"], "Parent replay differs")
        del model
        path = out / (name + "-resource-plan.json")
        write(path, candidate)
        prepared[name] = binding(path)
    require(torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized(), "Preparation changed RNG or CUDA")
    verify_inputs(plan)
    after = require_space(control, 450_000_000)
    write(out / "result.json", {"schema": "latency58-controlled-deployed-resource-preparation-v1", "status": "pass",
          "plan_sha256": args.plan_sha256, "source_bindings": sources, "source_bindings_unchanged": True,
          "protocol": binding(out / "protocol.json"), "plans": prepared, "both_parent_replays_exact": True,
          "cpu_rng_unchanged": True, "cuda_initialized": False, "training_updates_executed": 0,
          "optimizer_instances": 0, "counted_bytes_before": before, "counted_bytes_after": after,
          "reserved_bytes": 450_000_000, "quality_selected": False})
    print({"status": "pass", "plans": prepared, "counted_bytes_after": after}, flush=True)


if __name__ == "__main__":
    main()
