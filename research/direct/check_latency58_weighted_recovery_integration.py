"""Check the production storage adapter and selected-device recovery on CPU."""
from datetime import datetime, timezone
import ast
import json
import os
from pathlib import Path
import resource
import time

import torch

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_branch_memory_checkpoint import load_model
from research.direct.latency58_lossless_recovery_codec_v2 import policy as packed_policy
from research.direct.latency58_weighted_storage import NAME, policy as storage_policy, snapshot as storage_snapshot
from research.direct.latency58_weighted_vocal_auxiliary import VERSION, policy as loss_policy
from research.direct.latency58_weighted_vocal_canonical import policy as accumulation_policy
from research.direct.latency58_weighted_recovery_check import exercise_recovery


def main():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(key) == "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require CUDA-hidden CPU1")
    root = PHASE / NAME
    out = root / "cpu-integration"
    require(root.is_dir() and set(root.iterdir()) == {root / "cpu-integration-stage"}
            and not out.exists(), "Prepare CPU integration beside its sole fresh launch stage")
    source_path = PHASE / "branch-grouped-vocal-013/plan.json"
    source = read(source_path)
    retired = PHASE / "weighted-vocal-quarter-storage-001"
    receipt, execution = read(retired / "retirement-receipt.json"), read(retired / "retirement-execution.json")
    require(receipt["status"] == "complete" and execution["actual_exit_code"] == 0
            and execution["receipt_sha256"] == sha(retired / "retirement-receipt.json")
            and receipt["all_models_optimizers_plugin_binaries_sources_tests_and_logs_preserved"], "Storage reclamation is incomplete")
    previous = PHASE / "lossless-recovery-cpu-003"
    old, result, completed = (read(previous / name) for name in ("plan.json", "result.json", "execution.json"))
    require(result["status"] == "pass" and completed["actual_exit_code"] == 0
            and completed["result_sha256"] == sha(previous / "result.json"), "Packed CPU qualification is incomplete")
    bindings = dict(old["source_bindings"])
    paths = [Path(__file__).resolve(), source_path]
    paths.extend(previous / name for name in ("plan.json", "result.json", "execution.json"))
    paths.extend(retired / name for name in ("allocation-qualification.json", "allocation-qualification-execution.json",
                 "retirement-intent.json", "retirement-receipt.json", "retirement-execution.json"))
    paths.extend(ROOT / "research/direct" / name for name in (
        "latency58_weighted_storage.py", "latency58_lossless_recovery_files_v3.py", "latency58_weighted_recovery_check.py"))
    bindings.update({str(path): sha(path) for path in paths})
    baseline = ROOT / "research/direct/latency58_lossless_recovery_files_v2.py"
    adapted = ROOT / "research/direct/latency58_lossless_recovery_files_v3.py"
    def functions(path):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef)}
    first, second = functions(baseline), functions(adapted)
    require(first.keys() == second.keys() and all(value == second[name] for name, value in first.items()
            if name != "production_preflight"), "Production adapter changed qualified atomic or loading operations")
    fixture_source = {**source, "packed_recovery": packed_policy(), "weighted_storage": storage_policy(),
        "objective_version": VERSION, "grouped_vocal_loss": {key: value for key, value in loss_policy().items()
            if key != "production_recipe_selected"}, "accumulation_policy": accumulation_policy()}
    budget = storage_snapshot(fixture_source)
    plan = {"schema": "latency58-weighted-packed-production-adapter-cpu-v1", "source_bindings": bindings,
            "storage_budget": source["storage_budget"], "weighted_storage": storage_policy(),
            "fixture_parent": source["parent_checkpoint"], "full_schedule_steps": 1000,
            "synthetic_disk_destination": str(root / "production-run"),
            "destination_absent_at_entry_and_removed_after_fixture": True, "budget_before": budget,
            "qualified_atomic_and_load_function_bodies_unchanged": True, "gpu_used": False, "quality_measured": False}
    verify_inputs(plan); out.mkdir(parents=True); write(out / "plan.json", plan)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    parent, _ = load_model(source["parent_checkpoint"]); parent.training_precision = "fp32"
    began = time.monotonic()
    def progress(row):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        require(rss < 12_000_000_000, "Recovery integration exceeded its CPU memory bound")
        print(json.dumps({**row, "peak_rss_bytes": rss}), flush=True)
    result = exercise_recovery(parent, fixture_source, out, disk_directory=root / "production-run", progress=progress)
    verify_inputs(plan)
    require(result["status"] == "pass" and result["tensor_count"] == 216
            and result["training_schedule_steps"] == 1000 and result["production_rng_restored"]
            and result["production_filesystem_wrapper"]["all_tensor_and_metadata_values_exact"]
            and not torch.cuda.is_initialized() and not (root / "production-run").exists(),
            "CPU production-adapter integration is incomplete")
    result.update(schema=plan["schema"], plan_sha256=sha(out / "plan.json"), source_bindings_unchanged=True,
                  qualified_atomic_and_load_function_bodies_unchanged=True,
                  production_storage_preflight_exercised_with_real_packed_file=True,
                  gpu_used=False, quality_measured=False, elapsed_seconds=time.monotonic() - began,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  budget_after=storage_snapshot(fixture_source), completed_utc=datetime.now(timezone.utc).isoformat())
    write(out / "result.json", result)
    print(json.dumps({"status": "pass", "tensor_count": result["tensor_count"],
                      "complete_pack_and_audit_seconds": result["complete_pack_and_audit_seconds"],
                      "complete_production_wrapper_save_seconds": result["production_filesystem_wrapper"]["complete_production_wrapper_save_seconds"],
                      "elapsed_seconds": result["elapsed_seconds"], "peak_rss_bytes": result["peak_rss_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
