"""Check saved leader-cleanup and retained vocal metadata without exporting ONNX."""
from __future__ import annotations

import argparse
import ast
import copy
import importlib
import os
from pathlib import Path
from types import SimpleNamespace

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.train_latency58 import verify_inputs


def check_export_code():
    from research.direct import export_latency58_sdr_candidate_v6 as old
    from research.direct import export_latency58_sdr_candidate_v7 as new
    previous, current = [ast.parse(Path(module.__file__).read_text()) for module in (old, new)]
    for name in ("main", "write_new"):
        left, right = [copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                                          and n.name == name)) for tree in (previous, current)]
        if name == "main":
            assignments = [n for n in right.body if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == "loader_name" for t in n.targets)]
            require(len(assignments) == 1 and ast.dump(assignments[0].value)
                    == ast.dump(ast.parse("checkpoint_loader(family)", mode="eval").body),
                    "Unexpected checkpoint-loader transformation")
            assignments[0].value = ast.parse('"latency58_" + family + "_checkpoint.py"', mode="eval").body
        require(ast.dump(left) == ast.dump(right).replace("export_latency58_sdr_candidate_v7.py",
                                                        "export_latency58_sdr_candidate_v6.py"),
                "Export, numerical parity or disk checks changed")
    require(new.CANDIDATE_FAMILIES == {**old.CANDIDATE_FAMILIES, "leader_cleanup_candidate": "leader_cleanup"}
            and all(new.checkpoint_loader(family) == "latency58_" + family + "_checkpoint.py"
                    for family in old.CANDIDATE_FAMILIES.values())
            and new.checkpoint_loader("leader_cleanup") == "latency58_leader_cleanup_checkpoint_v2.py",
            "Existing routing changed or leader loader differs")
    return old, new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Metadata check changed")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-leader-export-metadata-plan-v1"
            and set(plan["models"]) == {"counterfactual_candidate", "controlled_deployed_candidate",
                                       "leader_cleanup_candidate"}, "Require all three real saved endpoints")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require CUDA-hidden CPU1")
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "result.json").exists(), "Preserve metadata check")
    from research.direct.latency58_sdr_checkpoint import require_space
    before = require_space(plan, 1_000_000)
    old, new = check_export_code()
    import torch
    from research.direct.latency58_evaluate import model_state_sha256
    from research.direct.latency58_asymmetric_onnx import STATE_SHAPES
    from research.direct.report_latency58_sdr import load_completed
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260907)
    evidence, results = {}, {}
    rejected = 0
    for kind, binding in plan["models"].items():
        directory = Path(binding["path"]).parent
        require(binding["path"] == str(directory / "plan.json")
                and plan["source_bindings"].get(binding["path"]) == binding["sha256"] == sha(binding["path"]),
                "Different or unbound primary endpoint")
        quality, scored = load_completed(directory, evidence)
        module = importlib.import_module("research.direct.evaluate_latency58_" + new.CANDIDATE_FAMILIES[kind])
        model, receipt = module.load_evaluation_model(quality)
        fingerprint = model_state_sha256(model)
        require(receipt["step"] == quality["step"] == 250
                and fingerprint == scored["results"][0]["model"]["model_state_sha256"], "Different saved model")
        rng = torch.get_rng_state().clone()
        request = {**quality, "model_kind": kind}
        metadata = new.metadata(request, model, 250, fingerprint, STATE_SHAPES)
        core = old.original_metadata(quality, model, 250, fingerprint, STATE_SHAPES)
        except_keys = {"hs_tasnet.exporter_sha256", "hs_tasnet.checkpoint_loader_sha256"}
        require(all(metadata[k] == v for k, v in core.items() if k not in except_keys)
                and all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items())
                and metadata["hs_tasnet.intended_total_latency_samples"] == "256"
                and metadata["hs_tasnet.graph_output_delay_samples"] == "128"
                and metadata["hs_tasnet.intended_external_host_queue_samples"] == "128"
                and metadata["hs_tasnet.local_mask_mixer"] == "false"
                and metadata["hs_tasnet.teacher_used_in_inference"] == "false"
                and metadata["hs_tasnet.teacher_mode"] == "ordinary_only", "Deployment metadata contract changed")
        if kind != "leader_cleanup_candidate":
            previous = old.metadata(request, model, 250, fingerprint, STATE_SHAPES)
            require({k: v for k, v in metadata.items() if k != "hs_tasnet.exporter_sha256"}
                    == {k: v for k, v in previous.items() if k != "hs_tasnet.exporter_sha256"},
                    "Retained candidate metadata changed")
        else:
            training = read(quality["training_plan"]["path"])
            require(metadata["hs_tasnet.leader_cleanup_trial_updates"] == "250"
                    and metadata["hs_tasnet.additional_loss_weight"] == "0.5"
                    and metadata["hs_tasnet.additional_loss_used_only_during_training"] == "true"
                    and metadata["hs_tasnet.deployed_truth_divisor"] == "20"
                    and metadata["hs_tasnet.parent_model_state_sha256"] == training["parent"]["model_state_sha256"]
                    and metadata["hs_tasnet.reference_training_plan_sha256"] == training["reference_training_plan"]["sha256"]
                    and metadata["hs_tasnet.comparison_variable"] == "training_parent"
                    and metadata["hs_tasnet.matched_loss_effect_from_leader_claimed"] == "false"
                    and metadata["hs_tasnet.checkpoint_loader_sha256"]
                        == sha(ROOT / "research/direct/latency58_leader_cleanup_checkpoint_v2.py")
                    and "hs_tasnet.controlled_deployed_trial_updates" not in metadata
                    and "hs_tasnet.counterfactual_trial_updates" not in metadata,
                    "Wrong transfer provenance or invented prior trial counts")
            for changed_step, changed in ((249, {}), (250, {"local_mask_mixer": True}),
                                          (250, {"matched_loss_effect_from_leader_claimed": True})):
                invalid = SimpleNamespace(output_source_scales=model.output_source_scales,
                                          provenance={**model.provenance, **changed})
                try:
                    new.metadata(request, invalid, changed_step, fingerprint, STATE_SHAPES)
                except ValueError as error:
                    require(str(error) == "Different leader-cleanup export recipe or endpoint", "Unexpected metadata rejection")
                    rejected += 1
                else:
                    raise RuntimeError("Invalid transfer metadata accepted")
        require(fingerprint == model_state_sha256(model) and torch.equal(rng, torch.get_rng_state())
                and not torch.cuda.is_initialized(), "Metadata mutated model or RNG, or initialized CUDA")
        results[kind] = {"model_state_sha256": fingerprint, "metadata": metadata}
        del model
    require(rejected == 3 and all(plan["source_bindings"].get(p) == v for p, v in evidence.items()),
            "Incomplete negative cases or unbound completed primary evidence")
    verify_inputs(plan)
    write(out / "result.json", {
        "schema": "latency58-leader-export-metadata-v1", "status": "pass", "plan_sha256": args.plan_sha256,
        "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
        "models": results, "invalid_transfer_metadata_rejected": rejected,
        "original_export_and_parity_code_preserved": True, "retained_candidate_metadata_exact": True,
        "all_original_deployment_metadata_preserved": True, "counted_bytes_before": before,
        "onnx_export_executed": False, "inference_executed": False, "training_updates_executed": 0,
        "cuda_initialized": False, "quality_selected": False, "native_host_qualified": False,
        "limitations": ["Metadata checks do not qualify actual ONNX, continuous native streaming, latency or M4 runtime."]})
    print({"status": "pass", "models": list(results), "onnx_export_executed": False}, flush=True)


if __name__ == "__main__":
    main()
