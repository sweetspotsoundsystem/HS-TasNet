"""Freeze a context recipe, then release matched plans after GPU qualification."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.train_latency58 import PRODUCTION, verify_inputs


def freeze_recipe(args, out):
    from research.direct.report_latency58_sdr import load_completed
    from research.direct.latency58_context_checkpoint import load_parent

    require(args.decision is not None, "A recorded next-experiment decision is required")
    decision = read(args.decision)
    require(decision["schema"] == "latency58-context-decision-v1" and not decision["quality_selected"]
            and decision["teacher_kind"] in ("c91", "cropped11")
            and decision["hypothesis"] and decision["rationale"], "Malformed context decision")
    verify_inputs(decision)
    prefix = decision["parent_prefix"]
    require(prefix and all(c.isalnum() or c in "-_" for c in prefix), "Invalid parent prefix")
    bindings = {str(args.decision.resolve()): sha(args.decision), **decision["source_bindings"]}
    parent_plans, states, parent_reports = {}, {}, {}
    for mode in ("full14", "actions60", "probes"):
        plan, report = load_completed(PHASE / (prefix + "-" + mode + "-001"), bindings,
                                      canonical_baseline=prefix == "teacher-half250")
        parent_plans[mode], parent_reports[mode] = plan, report
        if mode == "probes":
            require(report["status"] == "pass" and report["source_bindings_unchanged"], "Parent probes failed")
            states[mode] = report["model_state_sha256"]
        else:
            require(report["inputs_unchanged"], "Parent music inputs changed")
            states[mode] = report["results"][0]["model"]["model_state_sha256"]
        if prefix != "teacher-half250":
            bindings.update(plan["source_bindings"])
    require(len(set(states.values())) == 1, "Parent bundle contains different weights")
    old = read(PHASE / "sdr-teacher-prep-001" / (decision["teacher_kind"] + "-plan.json"))
    verify_inputs(old)
    bindings.update(old["source_bindings"])
    model_identity = parent_reports["full14"]["results"][0]["model"]
    if prefix == "teacher-half250":
        from research.direct.latency58_sdr_teacher import STUDENT_SHA256
        checkpoint = {"path": str(PHASE / "teacher-half-canonical-001/model.pt"), "sha256": STUDENT_SHA256}
        import torch
        provenance = torch.load(checkpoint["path"], map_location="cpu", weights_only=True)["provenance"]
        parent = {"kind": "working_baseline", "checkpoint": checkpoint, "provenance": provenance}
    else:
        plan = parent_plans["full14"]
        require(all(parent_plans[m]["generation"] == plan["generation"]
                    and parent_plans[m]["step"] == plan["step"] for m in parent_plans), "Mixed parent endpoints")
        parent = {"kind": "sdr_candidate", "checkpoint": plan["checkpoint"], "generation": plan["generation"],
                  "training_plan": plan["training_plan"], "provenance": model_identity["provenance"]}
    parent["model_state_sha256"] = states["full14"]
    config = copy.deepcopy(old["config"])
    config.update(steps=500, crop_samples=176128, seed=20260909, data_start=920000)
    common = {key: copy.deepcopy(old[key]) for key in
              ("environment", "torch_version", "precision_policy", "helper_source", "watchdog_source",
               "manifest_sha256", "counted_roots", "stop_counted_bytes", "teacher_kind", "teacher_weight",
               "teacher", "teacher_model_state_sha256")}
    proofs = []
    for name in ("sdr-context-functional-001", "sdr-context-data-functional-001",
                 "sdr-context-augmentation-functional-001", "sdr-context-lineage-functional-001"):
        directory = PHASE / name
        p, result, execution = (directory / n for n in ("plan.json", "result.json", "functional-execution.json"))
        evidence, actual = read(result), read(execution)
        verify_inputs(read(p))
        require(evidence["status"] == "pass" and evidence["source_bindings_unchanged"]
                and actual["actual_exit_code"] == 0 and not actual["timed_out"] and actual["source_bindings_unchanged"]
                and actual["plan_sha256"] == sha(p), "CPU context prerequisite differs")
        proofs.append({"result": str(result), "execution": str(execution)})
        bindings.update(read(p)["source_bindings"])
        bindings.update({str(f): sha(f) for f in (p, result, execution)})
    paths = [Path(__file__).resolve(), ROOT / "research/direct/report_latency58_sdr.py",
             ROOT / "research/direct/run_latency58_quality.py", PRODUCTION / "train_production.py",
             PRODUCTION / "full_config.json", PRODUCTION / "manifests/combined.manifest.json"]
    paths += [ROOT / "research/direct" / name for name in
              ("latency58_context_checkpoint.py", "train_latency58_context.py", "audit_latency58_context.py",
               "run_latency58_context_stage.py", "check_latency58_context_resource.py", "run_latency58_context_resource.py")]
    bindings.update({str(p): sha(p) for p in paths})
    recipe = {"schema": "latency58-context-recipe-v1", **common, "parent": parent, "parent_prefix": prefix,
              "config": config, "warmup_samples": 88064, "scored_samples": 88064,
              "functional_proofs": proofs, "source_bindings": bindings,
              "hypothesis": decision["hypothesis"],
              "matching": "Same parent and fresh Adam, full augmented crop, scored suffix, teacher context, loss and LR. Only student state carry differs; compare every augmented-batch and teacher-target digest.",
              "quality_endpoints": [250, 500], "first_saved_state_check": 2,
              "geometry": {"graph_delay_samples": 128, "queue_samples": 128, "sample_rate": 44100}}
    model = load_parent(recipe)
    require(model.provenance == parent["provenance"], "Parent preparation changed lineage")
    verify_inputs(recipe)
    write(out / "recipe.json", recipe)
    return {"recipe": {"path": str(out / "recipe.json"), "sha256": sha(out / "recipe.json")},
            "parent_model_state_sha256": parent["model_state_sha256"], "training_started": False}


def release_training(args, out):
    from research.direct.latency58_context_checkpoint import require_space, load_parent

    require(args.recipe is not None and args.recipe_sha256 is not None and args.resource_directory is not None
            and sha(args.recipe) == args.recipe_sha256, "Frozen recipe and full GPU resource result are required")
    recipe = read(args.recipe)
    verify_inputs(recipe)
    require(recipe["schema"] == "latency58-context-recipe-v1", "Unknown context recipe")
    directory = args.resource_directory.resolve()
    resource_plan, result, execution = [read(directory / n) for n in ("plan.json", "result.json", "execution.json")]
    verify_inputs(resource_plan)
    monitor_path = Path(execution["monitor_result"])
    monitor = read(monitor_path)
    require(resource_plan["parent"] == recipe["parent"] and resource_plan["teacher"] == recipe["teacher"]
            and resource_plan["source_bindings"][str(args.recipe.resolve())] == args.recipe_sha256
            and result["status"] == "pass" and result["arms"] == [False, True]
            and result["warmup_samples"] == result["scored_samples"] == 88064
            and result["initial_model_state_sha256"] == recipe["parent"]["model_state_sha256"]
            and result["teacher_kind"] == recipe["teacher_kind"]
            and result["teacher_identity"]["model_state_sha256"] == recipe["teacher_model_state_sha256"]
            and execution["actual_exit_code"] == monitor["child_exit_code"] == 0
            and execution["source_bindings_unchanged"] and monitor["status"] == monitor["supervisor_health"] == "pass"
            and monitor["post_exit_quiet_completed"] and result["plan_sha256"] == execution["plan_sha256"]
            == sha(directory / "plan.json"), "Full context GPU prerequisite differs")
    require_space(recipe, 1_600_000_000)
    load_parent(recipe)
    bindings = {**recipe["source_bindings"], **resource_plan["source_bindings"]}
    bindings.update({str(p): sha(p) for p in (args.recipe.resolve(), monitor_path, directory / "plan.json",
                                           directory / "result.json", directory / "execution.json")})
    plans = {}
    for name, carry in (("reset", False), ("warm", True)):
        plan = copy.deepcopy(recipe)
        plan.update(schema="latency58-context-training-v1", carry_state=carry,
                    run_dir=str(PHASE / ("sdr-context-" + name + "-b4-lr3e5-500")), source_bindings=bindings,
                    full_resource={"path": str(directory / "result.json"), "sha256": sha(directory / "result.json")},
                    full_resource_execution={"path": str(directory / "execution.json"), "sha256": sha(directory / "execution.json")})
        require(not Path(plan["run_dir"]).exists(), "Preserve an existing context arm")
        verify_inputs(plan)
        path = out / (name + "-plan.json")
        write(path, plan)
        plans[name] = {"path": str(path), "sha256": sha(path)}
    return {"plans": plans, "training_started": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("recipe", "training"), required=True)
    parser.add_argument("--decision", type=Path)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--recipe-sha256")
    parser.add_argument("--resource-directory", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_directory.absolute()
    require(Path.cwd() == ROOT and out.parent == PHASE and not out.exists()
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Prepare in a fresh phase directory with CUDA hidden and CPU1")
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out.mkdir()
    result = freeze_recipe(args, out) if args.mode == "recipe" else release_training(args, out)
    require(not torch.cuda.is_initialized(), "Context preparation initialized CUDA")
    result.update(status="prepared", cuda_initialized=False)
    write(out / "preparation-result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
