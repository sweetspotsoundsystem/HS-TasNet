"""Screen one unadapted gate transform on the existing Actions excerpt, on CPU."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import read, require, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Gate screen plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-gate-bias-actions-screen-v1"
            and plan["mode"] == "actions60" and plan["offset"] == 0.6931471805599453
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
            and all(sha(p) == s for p, s in plan["source_bindings"].items()),
            "Use the bound single-offset Actions screen with CUDA hidden and CPU1")
    proof, execution = read(plan["functional_proof"]["path"]), read(plan["functional_execution"]["path"])
    require(proof["status"] == "pass" and proof["source_bindings_unchanged"]
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"] and execution["plan_sha256"] == proof["plan_sha256"]
            and all(sha(p) == s for p, s in proof["source_bindings"].items()),
            "Gate transform lacks its actual functional qualification")
    parent_binding = plan["parent_training_plan"]
    require(sha(parent_binding["path"]) == parent_binding["sha256"], "Parent loading plan changed")
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve any prior screen")
    import torch
    from research.direct.latency58_gate_bias import with_update_gate_bias
    from research.direct.latency58_log_relative_checkpoint import load_parent
    from research.direct.latency58_evaluate import evaluate_latency58_music, model_state_sha256

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260907)
    torch.use_deterministic_algorithms(True)
    parent = load_parent(read(parent_binding["path"])).eval().requires_grad_(False)
    model = with_update_gate_bias(parent, offset=plan["offset"])
    fingerprint = model_state_sha256(model)
    require(model_state_sha256(parent) == proof["parent_model_state_sha256"]
            and fingerprint == proof["transformed_model_state_sha256"], "Screen transform differs from the functional model")
    identity = {"label": plan["label"], "state_kind": "untrained_initialization", "training_updates": 0,
                "checkpoint": None, "model_state_sha256": fingerprint,
                "provenance": {"initialization": "unadapted_update_gate_bias_transform",
                               "training_updates_after_transform": 0,
                               "parent_cumulative_training_updates": parent.provenance["training_updates"],
                               "parameter_transform_provenance": model.provenance}}
    began = time.monotonic()
    result = evaluate_latency58_music(model, identity=identity, track_indices=[1], excerpt_starts=[60.0])
    require(all(sha(p) == s for p, s in plan["source_bindings"].items())
            and model_state_sha256(model) == fingerprint and not torch.cuda.is_initialized(),
            "Screen input, transformed state or CPU scope changed")
    result.update(plan_sha256=args.plan_sha256, root_source_bindings=plan["source_bindings"],
                  inputs_unchanged=True, total_elapsed_seconds=time.monotonic() - began,
                  checkpoint_written=False, audio_written=False, training_updates_executed=0,
                  quality_selected=False, screen_scope="One existing Actions excerpt; no full-panel quality conclusion")
    write(out / "result.json", result)
    print(str(result["results"][0]["aggregate"]), flush=True)


if __name__ == "__main__":
    main()
