"""Create one inference snapshot from a separately audited asymmetric-window resume.

This does not update the trainer's checkpoint pointer or instantiate Adam.
The output is published only after a complete serialization/readback check.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

from research.direct.latency58_asymmetric_checkpoint import require, sha

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Snapshot plan changed")
    plan = json.loads(args.plan.read_text())
    require(plan["schema"] == "latency58-asymmetric-snapshot-plan-v1"
            and plan["checkpoint"]["kind"] == "audited_resume"
            and all(sha(path) == digest for path, digest in plan["source_bindings"].items()),
            "Snapshot source or audited input identity differs")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(name) == "1" for name in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CUDA-hidden CPU1")
    output = Path(plan["output"])
    phase = ROOT / "research/direct/runs/latency58"
    receipt = output.with_suffix(".snapshot.json")
    require(output.is_absolute() and output.is_relative_to(phase) and output.parent.is_dir()
            and output.suffix == ".pt" and not output.exists() and not receipt.exists(), "Use a fresh snapshot path")
    from research.direct.train_latency58 import disk_bytes, state_sha256
    require(disk_bytes(phase) + 230_000_000 < plan["artifact_allowance_bytes"],
            "Insufficient temporary and published snapshot allowance")
    import torch
    from research.direct.latency58_asymmetric_checkpoint import load_model_state, make_model
    from research.direct.latency58_evaluate import model_state_sha256

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260907)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    model = make_model(plan["parent_checkpoint"])
    step = load_model_state(model, plan["checkpoint"])
    fingerprint = model_state_sha256(model)
    require(step == plan["step"] and fingerprint == plan["model_state_sha256"], "Snapshot model identity differs")
    rng = torch.get_rng_state().clone()
    payload = {"schema": "latency58-asymmetric-inference-v1", "step": step, "model": model.state_dict(),
               "model_state_sha256": fingerprint, "provenance": model.provenance,
               "architecture": model.architecture_metadata,
               "plan_sha256": model.provenance["training_plan_sha256"]}
    descriptor, name = tempfile.mkstemp(prefix=output.stem + ".", suffix=".pending.pt", dir=output.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        saved = torch.load(temporary, map_location="cpu", weights_only=True)
        require(set(saved) == set(payload) and saved["schema"] == payload["schema"] and saved["step"] == step
                and saved["architecture"] == payload["architecture"] and saved["provenance"] == payload["provenance"]
                and saved["plan_sha256"] == payload["plan_sha256"]
                and saved["model_state_sha256"] == state_sha256(saved["model"]) == fingerprint,
                "Inference snapshot serialization did not preserve the audited model")
        require(all(sha(path) == digest for path, digest in plan["source_bindings"].items())
                and sha(plan["checkpoint"]["path"]) == plan["checkpoint"]["sha256"]
                and torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized(),
                "Snapshot source, checkpoint, RNG or CPU scope changed")
        result = {"status": "pass", "step": step, "model_state_sha256": fingerprint,
                  "input": plan["checkpoint"], "output": {"path": str(output), "sha256": sha(temporary),
                                                          "bytes": temporary.stat().st_size},
                  "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"],
                  "source_bindings_unchanged": True, "round_trip_model_state_exact": True,
                  "optimizer_instances": 0, "training_updates_executed": 0, "cuda_initialized": False,
                  "elapsed_seconds": time.monotonic() - started}
        os.link(temporary, output)
        with receipt.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(result, allow_nan=False))
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
