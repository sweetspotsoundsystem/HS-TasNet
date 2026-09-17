"""Read-only CPU audit of a context trial's model, Adam state, RNG and journal."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random

from research.direct.train_latency58 import load_source, read, require, sha, state_sha256, verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256 and not args.output.exists(), "Audit plan or output differs")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Audit requires CPU1 with CUDA hidden")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-sdr-drum-training-v2", "Unknown training plan")
    from research.direct.latency58_sdr_drum_v2_checkpoint import load_model, read_generation, validate_journal
    receipt = read_generation(args.generation, expected_plan_sha=args.plan_sha256)
    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(torch.__version__ == plan["torch_version"] and not torch.cuda.is_initialized(), "Audit runtime differs")
    model, loaded = load_model(args.generation, plan, expected_plan_sha=args.plan_sha256)
    require(loaded == receipt and 0 < receipt["step"] <= plan["config"]["steps"], "Loaded generation differs")
    helpers = load_source("latency58_sdr_audit_helpers", plan["helper_source"])
    step, config = receipt["step"], plan["config"]
    journal = (args.generation / "metrics.jsonl").read_bytes()
    rows = validate_journal(journal, step, plan, helpers)
    require(len(journal) == receipt["metrics_bytes"] and hashlib.sha256(journal).hexdigest() == receipt["metrics_sha256"]
            and receipt["next_sample_index"] == config["data_start"] + step * config["batch_size"],
            "Journal identity or data endpoint differs")
    names = [name for name, _ in model.named_parameters()]
    require(len(names) == 21 and receipt["optimizer_parameter_names"] == names, "Parameter order differs")
    optimizer = torch.load(args.generation / "optimizer.pt", map_location="cpu", weights_only=True)
    require(set(optimizer) == {"state", "param_groups"} and len(optimizer["param_groups"]) == 1
            and set(optimizer["state"]) == set(range(21)), "Adam inventory differs")
    helpers.validate_adam_group(optimizer["param_groups"][0], list(range(21)), step, config)
    for index, parameter in enumerate(model.parameters()):
        state = optimizer["state"][index]
        require(set(state) == {"step", "exp_avg", "exp_avg_sq"} and state["step"].shape == ()
                and state["step"].dtype == torch.float32 and float(state["step"]) == step, "Adam step differs")
        for key in ("exp_avg", "exp_avg_sq"):
            value = state[key]
            require(value.shape == parameter.shape and value.dtype == torch.float32 and value.device.type == "cpu"
                    and bool(torch.isfinite(value).all()), "Malformed FP32 Adam moment")
        require(bool((state["exp_avg_sq"] >= 0).all()), "Negative second moment")
    rng = torch.load(args.generation / "rng.pt", map_location="cpu", weights_only=False)
    require(set(rng) == {"python", "numpy", "torch_cpu", "torch_cuda"}
            and isinstance(rng["torch_cuda"], list) and len(rng["torch_cuda"]) == 1, "RNG inventory differs")
    random.Random().setstate(rng["python"])
    np.random.RandomState().set_state(rng["numpy"])
    for value in (rng["torch_cpu"], rng["torch_cuda"][0]):
        require(value.dtype == torch.uint8 and value.device.type == "cpu" and value.ndim == 1 and value.numel() > 0,
                "Malformed RNG bytes")
    require(rng["torch_cpu"].shape == torch.get_rng_state().shape, "CPU RNG shape differs")
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        audio = torch.linspace(-.1, .1, 1024).reshape(1, 2, 512)
        first, second = model.render(audio), model.render(audio)
        closure = float((first.deployed.sum(dim=1) - first.delayed_mixture).abs().max())
        require(torch.equal(first.deployed, second.deployed) and bool(torch.isfinite(first.deployed).all())
                and closure <= 1e-6 and all(torch.equal(a, b) for a, b in zip(first.state, second.state, strict=True)),
                "Saved inference reset or reconstruction failed")
    verify_inputs(plan)
    require(read_generation(args.generation, expected_plan_sha=args.plan_sha256) == receipt
            and not torch.cuda.is_initialized(), "Audit changed generation or initialized CUDA")
    result = {"schema": "latency58-sdr-drum-audit-v2", "status": "pass", "step": step,
              "plan_sha256": args.plan_sha256, "generation": str(args.generation.resolve()),
              "generation_receipt_sha256": sha(args.generation / "receipt.json"),
              "checkpoint": {"path": str((args.generation / "model.pt").resolve()),
                             "sha256": receipt["files"]["model.pt"]["sha256"]},
              "model_state_sha256": state_sha256(model.state_dict()), "teacher_kind": plan["teacher_kind"],
              "parameter_tensors": 21, "buffer_tensors": 6, "adam_state_pairs": 21,
              "batch_identity_rows": len(rows), "fixed_buffers_unchanged": True,
              "exact_reset_replay": True, "closure_max_abs": closure,
              "source_bindings_unchanged": True, "cumulative_training_updates": model.provenance["training_updates"],
              "carry_state": plan["carry_state"], "warmup_samples": plan["warmup_samples"],
              "scored_samples": plan["scored_samples"],
              "drum_weight": plan["drum_weight"], "objective_version": plan["objective_version"],
              "normalized_drum_objective_journal_verified": True,
              "training_updates_executed": 0, "optimizer_instances": 0, "cuda_initialized": False}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
