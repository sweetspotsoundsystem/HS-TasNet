"""Check CLI imports and current-trial metadata without constructing a candidate."""
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import json
import os
import subprocess
import time

from research.direct.run_latency58_quality import ROOT, read, require, sha, write
from research.direct.train_latency58 import verify_inputs

out = Path(__file__).resolve().parent
plan = read(out / "plan.json")
verify_inputs(plan)
require(os.environ["CUDA_VISIBLE_DEVICES"] == "" and all(os.environ[k] == "1" for k in
        ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "CPU-only source preflight")
began = time.monotonic()
results = []
for path in plan["sources"]:
    path = Path(path)
    tree = ast.parse(path.read_text())
    has_main = any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body)
    if has_main:
        completed = subprocess.run([plan["python"], "-B", "-m", "research.direct." + path.stem, "--help"],
                                   capture_output=True, text=True, timeout=30, check=False)
        require(completed.returncode == 0 and "usage:" in completed.stdout and not completed.stderr,
                "CLI source check failed: " + str(path))
    results.append({"path": str(path), "syntax_pass": True, "cli_help_pass": True if has_main else None})

from research.direct.latency58_sdr_drum_v3_checkpoint import expected_provenance
from research.direct.export_latency58_sdr_candidate_v4 import CANDIDATE_FAMILIES, metadata
from research.direct.latency58_asymmetric_onnx import STATE_SHAPES
import torch

fixture = read(plan["parent_fixture"]["path"])
require(sha(plan["parent_fixture"]["path"]) == plan["parent_fixture"]["sha256"], "Parent fixture changed")
recipe = {**copy.deepcopy(plan["recipe_metadata"]), "parent": copy.deepcopy(fixture["parent"])}
parent = copy.deepcopy(recipe["parent"]["provenance"])
cases = []
for step in (2, 250, 500, 1000):
    provenance = expected_provenance(recipe, step, "0" * 64)
    require(provenance["parent_provenance"] == parent == recipe["parent"]["provenance"]
            and provenance["training_updates"] == parent["training_updates"] + step
            and provenance["accum_trial_updates"] == parent["accum_trial_updates"]
            and provenance["drum_emphasis_trial_updates"] == step
            and provenance["gradient_accumulation_steps"] == 1
            and provenance["training_microbatch_size"] == 4
            and provenance["trial_augmented_examples"] == step * 4
            and provenance["trial_microbatches"] == step, "Current trial inherits obsolete batch metadata")
    # This is a metadata object, not trained weights or a candidate checkpoint.
    model = SimpleNamespace(provenance=provenance,
                            output_source_scales=SimpleNamespace(tolist=lambda: [1.0] * 4))
    values = metadata({"checkpoint": {"sha256": "0" * 64}, "model_kind": "drum_candidate"},
                      model, step, "0" * 64, STATE_SHAPES)
    expected = {"gradient_accumulation_steps": "1", "training_microbatch_size": "4",
                "trial_augmented_examples": str(step * 4), "trial_microbatches": str(step),
                "drum_emphasis_trial_updates": str(step), "drum_weight": "2",
                "intended_total_latency_samples": "256", "graph_qualified": "false",
                "native_host_timing_qualified": "false"}
    require(all(values["hs_tasnet." + key] == value for key, value in expected.items()),
            "Export metadata differs from the current trial or overstates qualification")
    cases.append({"constructed_metadata_only": True, "step": step, "expected_fields": expected})
require(CANDIDATE_FAMILIES["drum_candidate"] == "sdr_drum_v3"
        and all((ROOT / "research/direct" / (prefix + family + suffix)).is_file()
                for family in CANDIDATE_FAMILIES.values()
                for prefix, suffix in (("latency58_", "_checkpoint.py"), ("evaluate_latency58_", ".py")))
        and not torch.cuda.is_initialized(), "Export family mapping or CPU scope differs")
verify_inputs(plan)
write(out / "result.json", {"schema": "latency58-sdr-drum-source-preflight-v3", "status": "pass",
    "plan_sha256": sha(out / "plan.json"), "source_bindings": plan["source_bindings"],
    "source_bindings_unchanged": True, "results": results, "current_trial_metadata_cases": cases,
    "parent_provenance_unchanged": True, "training_updates_executed": 0, "checkpoint_written": False,
    "onnx_export_executed": False, "cuda_initialized": False, "quality_selected": False,
    "gpu_rehearsal_passed": False, "elapsed_seconds": time.monotonic() - began,
    "limitations": ["CLI and constructed metadata checks only; no trained candidate or ONNX parity is claimed.",
                    "The unchanged objective has independent real-parent gradient evidence; full-crop GPU use remains unqualified."]})
print(json.dumps({"status": "pass", "sources": len(results), "metadata_cases": len(cases)}), flush=True)
