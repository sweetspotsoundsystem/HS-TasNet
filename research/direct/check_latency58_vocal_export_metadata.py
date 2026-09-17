"""Check the real vocal endpoints' export metadata without writing an ONNX graph."""
from __future__ import annotations

import argparse
import ast
import importlib
import os
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.train_latency58 import verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Preflight plan or cwd differs")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-vocal-export-metadata-plan-v1"
            and set(plan["models"]) == {"counterfactual_candidate", "controlled_deployed_candidate"},
            "Different metadata scope")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require CUDA-hidden CPU1")
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "result.json").exists(), "Preserve preflight")
    from research.direct.latency58_sdr_checkpoint import require_space
    before = require_space(plan, 1_000_000)
    from research.direct import export_latency58_sdr_candidate_v5 as old
    from research.direct import export_latency58_sdr_candidate_v6 as new
    for name in ("main", "write_new"):
        bodies = [next(n for n in ast.parse(Path(m.__file__).read_text()).body
                       if isinstance(n, ast.FunctionDef) and n.name == name) for m in (old, new)]
        require(ast.dump(bodies[0]) == ast.dump(bodies[1]).replace(
            "export_latency58_sdr_candidate_v6.py", "export_latency58_sdr_candidate_v5.py"),
            "Export, parity or disk checks changed")
    require(all(new.CANDIDATE_FAMILIES[k] == v for k, v in old.CANDIDATE_FAMILIES.items()),
            "Existing candidate routing changed")
    import torch
    from research.direct.latency58_evaluate import model_state_sha256
    from research.direct.latency58_asymmetric_onnx import STATE_SHAPES
    from research.direct.report_latency58_sdr import load_completed
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260907)
    evidence, results = {}, {}
    for kind, binding in plan["models"].items():
        directory = Path(binding["path"]).parent
        require(binding["path"] == str(directory / "plan.json") and binding["sha256"] == sha(binding["path"]),
                "Different primary endpoint")
        quality, scored = load_completed(directory, evidence)
        module = importlib.import_module("research.direct.evaluate_latency58_" + new.CANDIDATE_FAMILIES[kind])
        model, receipt = module.load_evaluation_model(quality)
        fingerprint = model_state_sha256(model)
        require(receipt["step"] == quality["step"] == 250
                and fingerprint == scored["results"][0]["model"]["model_state_sha256"], "Different saved model")
        rng = torch.get_rng_state().clone()
        metadata = new.metadata({**quality, "model_kind": kind}, model, 250, fingerprint, STATE_SHAPES)
        core = old.original_metadata(quality, model, 250, fingerprint, STATE_SHAPES)
        except_keys = {"hs_tasnet.exporter_sha256", "hs_tasnet.checkpoint_loader_sha256"}
        require(all(metadata[k] == v for k, v in core.items() if k not in except_keys)
                and metadata["hs_tasnet.intended_total_latency_samples"] == "256"
                and metadata["hs_tasnet.graph_output_delay_samples"] == "128"
                and metadata["hs_tasnet.intended_external_host_queue_samples"] == "128"
                and metadata["hs_tasnet.local_mask_mixer"] == "false"
                and metadata["hs_tasnet.teacher_used_in_inference"] == "false"
                and metadata["hs_tasnet.teacher_mode"] == "ordinary_only"
                and fingerprint == model_state_sha256(model) and torch.equal(rng, torch.get_rng_state())
                and not torch.cuda.is_initialized(), "Metadata changed the deployment contract or model")
        if kind == "controlled_deployed_candidate":
            require(metadata["hs_tasnet.additional_loss_weight"] == "0.5"
                    and metadata["hs_tasnet.additional_loss_used_only_during_training"] == "true"
                    and metadata["hs_tasnet.deployed_truth_divisor"] == "20", "Missing deployed-training provenance")
        results[kind] = {"model_state_sha256": fingerprint, "metadata": metadata}
        del model
    require(all(plan["source_bindings"].get(p) == v for p, v in evidence.items()), "Unbound completed primary evidence")
    verify_inputs(plan)
    write(out / "result.json", {
        "schema": "latency58-vocal-export-metadata-v1", "status": "pass", "plan_sha256": args.plan_sha256,
        "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
        "models": results, "original_export_and_parity_code_preserved": True,
        "all_original_deployment_metadata_preserved": True, "counted_bytes_before": before,
        "onnx_export_executed": False, "inference_executed": False, "training_updates_executed": 0,
        "cuda_initialized": False, "quality_selected": False, "native_host_qualified": False,
        "limitations": ["Actual ONNX, continuous native streaming and M4 runtime remain unverified for these endpoints."]})
    print({"status": "pass", "models": list(results), "onnx_export_executed": False}, flush=True)


if __name__ == "__main__":
    main()
