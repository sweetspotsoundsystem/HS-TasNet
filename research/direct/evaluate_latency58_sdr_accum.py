"""Score an audited accumulated-gradient context generation with the preserved streamer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import read, require, sha, write


def load_evaluation_model(plan):
    from research.direct.latency58_sdr_accum_checkpoint import load_model
    training_binding = plan["training_plan"]
    require(sha(training_binding["path"]) == training_binding["sha256"], "Training plan changed")
    training = read(training_binding["path"])
    model, receipt = load_model(plan["generation"], training, expected_plan_sha=training_binding["sha256"])
    require(receipt["step"] == plan["step"]
            and receipt["files"]["model.pt"]["sha256"] == plan["checkpoint"]["sha256"],
            "Loaded a different quality endpoint")
    return model.eval().requires_grad_(False), receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Evaluation plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-sdr-accum-music-plan-v1"
            and plan["mode"] in ("actions60", "full14"), "Unexpected music protocol")
    require(all(sha(p) == s for p, s in plan["source_bindings"].items()), "Evaluation inputs changed")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CUDA-hidden CPU1")
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve existing result")
    import torch
    from research.direct.latency58_evaluate import evaluate_latency58_music, model_state_sha256
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260907)
    torch.use_deterministic_algorithms(True)
    model, receipt = load_evaluation_model(plan)
    identity = {"label": plan["label"], "state_kind": "checkpoint",
                "training_updates": model.provenance["training_updates"],
                "accum_trial_updates": receipt["step"], "provenance": model.provenance,
                "checkpoint": plan["checkpoint"], "model_state_sha256": model_state_sha256(model)}
    started = time.monotonic()
    with (out / "progress.jsonl").open("x", buffering=1) as stream:
        def progress(row):
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row, allow_nan=False), flush=True)
        options = {"track_indices": [1], "excerpt_starts": [60.0], "audio_dir": out / "audio"} \
            if plan["mode"] == "actions60" else {}
        result = evaluate_latency58_music(model, identity=identity, progress=progress, **options)
    require(all(sha(p) == s for p, s in plan["source_bindings"].items()), "Evaluation inputs changed during scoring")
    result.update(plan_sha256=args.plan_sha256, root_source_bindings=plan["source_bindings"],
                  inputs_unchanged=True, total_elapsed_seconds=time.monotonic() - started)
    write(out / "result.json", result)
    print(json.dumps({"result": str(out / "result.json"), "aggregate": result["results"][0]["aggregate"],
                      "elapsed_seconds": result["total_elapsed_seconds"]}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
