"""Read-only CPU audit of the atomic model/optimizer/RNG teacher endpoint."""
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
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists() and sha(args.plan) == args.plan_sha256, "Plan or output differs")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and all(os.environ.get(k) == "1" for k in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CPU1 with CUDA hidden")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-teacher-training-plan-v1", "Wrong training plan")
    verify_inputs(plan)
    run, config = Path(plan["run_dir"]), plan["config"]
    endpoint = run / "endpoint"
    pointer, receipt = read(run / "latest.json"), read(endpoint / "receipt.json")
    require(pointer["endpoint"] == str(endpoint) and pointer["receipt_sha256"] == sha(endpoint / "receipt.json")
            and pointer["step"] == receipt["step"] == config["steps"] == 250
            and pointer["plan_sha256"] == receipt["plan_sha256"] == args.plan_sha256
            and receipt["schema"] == "latency58-teacher-endpoint-v1"
            and set(receipt["files"]) == {"model.pt", "optimizer.pt", "rng.pt"}, "Endpoint inventory or identity differs")
    for name, binding in receipt["files"].items():
        require(sha(endpoint / name) == binding["sha256"] and (endpoint / name).stat().st_size == binding["bytes"],
                "Endpoint file differs")
    import numpy as np
    import torch
    from research.direct.latency58_teacher_checkpoint import make_model, load_model_state

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(torch.__version__ == plan["torch_version"] and not torch.cuda.is_initialized(), "Audit runtime differs")
    helpers = load_source("latency58_teacher_audit_helpers", plan["helper_source"])
    checkpoint = {"kind": "inference", "path": str(endpoint / "model.pt"), "sha256": receipt["files"]["model.pt"]["sha256"]}
    model = make_model(plan["hann_parent_checkpoint"])
    require(load_model_state(model, checkpoint) == 250 and state_sha256(model.state_dict()) == receipt["model_state_sha256"]
            and model.provenance["training_plan_sha256"] == args.plan_sha256
            and model.provenance["teacher_weight"] == plan["teacher_weight"]
            and model.provenance["training_objective"] == plan["objective"], "Model differs from plan and receipt")
    journal = (run / "metrics.jsonl").read_bytes()
    helpers.validate_journal(journal, 250, config)
    require(hashlib.sha256(journal).hexdigest() == receipt["metrics_sha256"] and len(journal) == receipt["metrics_bytes"]
            and receipt["next_sample_index"] == config["data_start"] + 1000, "Journal or sample count differs")
    rows = [json.loads(line) for line in journal.splitlines()]
    for row in rows:
        require(row["objective"] == plan["objective"] and row["teacher_weight"] == plan["teacher_weight"]
                and row["data_hops"] == 688 and row["flush_hops"] == 1
                and len(row["augmented_batch_sha256"]) == 64
                and all(c in "0123456789abcdef" for c in row["augmented_batch_sha256"])
                and all(math.isfinite(row[k]) for k in ("loss", "supervised_loss", "teacher_l1", "grad_norm"))
                and abs(row["loss"] - row["supervised_loss"] - plan["teacher_weight"] * row["teacher_l1"]) < 1e-7,
                "Malformed or inconsistent matched objective journal")
    names = [name for name, _ in model.named_parameters()]
    require(receipt["optimizer_parameter_names"] == names and len(names) == 21, "Parameter order differs")
    optimizer = torch.load(endpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    require(set(optimizer) == {"state", "param_groups"} and len(optimizer["param_groups"]) == 1
            and set(optimizer["state"]) == set(range(21)), "Adam inventory differs")
    helpers.validate_adam_group(optimizer["param_groups"][0], list(range(21)), 250, config)
    for index, parameter in enumerate(model.parameters()):
        state = optimizer["state"][index]
        require(set(state) == {"step", "exp_avg", "exp_avg_sq"} and state["step"].shape == ()
                and state["step"].dtype == torch.float32 and float(state["step"]) == 250, "Adam step differs")
        for name in ("exp_avg", "exp_avg_sq"):
            value = state[name]
            require(value.shape == parameter.shape and value.dtype == torch.float32 and value.device.type == "cpu"
                    and bool(torch.isfinite(value).all()), "Adam moments malformed")
        require(bool((state["exp_avg_sq"] >= 0).all()), "Negative squared Adam moment")
    # The RNG file is a SHA-authenticated local training artifact containing NumPy state.
    rng = torch.load(endpoint / "rng.pt", map_location="cpu", weights_only=False)
    require(set(rng) == {"python", "numpy", "torch_cpu", "torch_cuda"}
            and isinstance(rng["torch_cuda"], list) and len(rng["torch_cuda"]) == 1, "RNG inventory differs")
    random.Random().setstate(rng["python"])
    np.random.RandomState().set_state(rng["numpy"])
    for value in (rng["torch_cpu"], rng["torch_cuda"][0]):
        require(value.dtype == torch.uint8 and value.device.type == "cpu" and value.ndim == 1 and value.numel() > 0,
                "RNG bytes malformed")
    require(rng["torch_cpu"].shape == torch.get_rng_state().shape, "CPU RNG geometry differs")
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        audio = torch.linspace(-0.1, 0.1, 512).reshape(1, 2, 256)
        left, right = model.render(audio), model.render(audio)
        closure = float((left.deployed.sum(1) - left.delayed_mixture).abs().max())
        require(torch.equal(left.deployed, right.deployed) and torch.isfinite(left.deployed).all().item()
                and closure <= 1e-6 and all(torch.equal(a, b) for a, b in zip(left.state, right.state, strict=True)),
                "Saved model replay or closure failed")
    verify_inputs(plan)
    require(all(sha(endpoint / n) == v["sha256"] for n, v in receipt["files"].items())
            and not torch.cuda.is_initialized(), "Audit changed endpoint or used CUDA")
    result = {"status": "pass", "step": 250, "plan_sha256": args.plan_sha256,
              "checkpoint": checkpoint, "endpoint_receipt_sha256": sha(endpoint / "receipt.json"),
              "model_state_sha256": receipt["model_state_sha256"], "source_bindings_unchanged": True,
              "parameter_tensors": 21, "buffer_tensors": 6, "adam_state_pairs": 21,
              "fixed_buffers_unchanged": True, "exact_reset_replay": True, "closure_max_abs": closure,
              "batch_identity_rows": 250, "objective": plan["objective"], "teacher_weight": plan["teacher_weight"],
              "cumulative_training_updates": 5000, "cumulative_asymmetric_updates": 750,
              "training_updates_executed": 0, "optimizer_instances": 0, "cuda_initialized": False}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
