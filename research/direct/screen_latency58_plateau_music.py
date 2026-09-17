"""Screen the untrained plateau carrier on the fixed Actions music excerpt."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Evaluation plan changed")
    plan = json.loads(args.plan.read_text())
    require(plan["schema"] == "latency58-plateau-music-plan-v1" and plan["mode"] == "actions60",
            "Unexpected evaluation plan")
    require(all(sha(path) == digest for path, digest in plan["source_bindings"].items()),
            "Evaluation source/input changed")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(name) == "1" for name in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CPU1 with CUDA hidden")
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve existing result")
    import torch
    from research.direct.latency58 import Latency58Model
    from research.direct.latency58_plateau import Latency58PlateauModel
    from research.direct.latency58_checkpoint import load_model_state
    from research.direct.latency58_evaluate import evaluate_latency58_music, model_state_sha256

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260907)
    torch.use_deterministic_algorithms(True)
    parent = Latency58Model.from_accepted().eval().requires_grad_(False)
    parent_step = load_model_state(parent, plan["parent_checkpoint"])
    model = Latency58PlateauModel.from_hann_model(parent).eval().requires_grad_(False)
    checkpoint, state_kind, updates = None, "untrained_initialization", 0
    functional = plan["functional_check"]
    require(sha(functional["path"]) == functional["sha256"], "Functional evidence changed")
    proof = json.loads(Path(functional["path"]).read_text())
    require(proof["status"] == "pass" and proof["parent_step"] == parent_step
            and proof["parent_model_state_sha256"] == model_state_sha256(parent)
            and proof["model_state_sha256"] == model_state_sha256(model), "Functional model identity differs")
    identity = {"label": plan["label"], "state_kind": state_kind, "training_updates": updates,
                "provenance": model.provenance, "checkpoint": checkpoint,
                "parent_checkpoint": plan["parent_checkpoint"], "parent_step": parent_step,
                "model_state_sha256": model_state_sha256(model)}
    progress_file = (out / "progress.jsonl").open("x", buffering=1)
    def progress(row):
        progress_file.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)

    started = time.monotonic()
    options = {"track_indices": [1], "excerpt_starts": [60.0], "audio_dir": out / "audio"} \
        if plan["mode"] == "actions60" else {}
    result = evaluate_latency58_music(model, identity=identity, progress=progress, **options)
    require(all(sha(path) == digest for path, digest in plan["source_bindings"].items()),
            "An evaluation input changed during execution")
    result.update(plan_sha256=args.plan_sha256, root_source_bindings=plan["source_bindings"],
                  inputs_unchanged=True, total_elapsed_seconds=time.monotonic() - started)
    with (out / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    progress_file.close()
    print(json.dumps({"result": str(out / "result.json"),
                      "aggregate": result["results"][0]["aggregate"],
                      "elapsed_seconds": result["total_elapsed_seconds"]}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
