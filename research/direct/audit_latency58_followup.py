"""Read-only CPU audit of a completed hop128 resume file; no optimizer or CUDA."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--resume-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and sha(args.plan) == args.plan_sha256, "Plan/output identity differs")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(name) == "1" for name in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CPU1 with CUDA hidden")
    plan = json.loads(args.plan.read_text())
    require(all(sha(path) == digest for path, digest in plan["source_bindings"].items()), "Training sources changed")
    run = Path(plan["run_dir"])
    resume = run / "resume.pt"
    require(sha(resume) == args.resume_sha256, "Saved resume identity differs")
    import torch
    from research.direct.latency58 import Latency58Model, Latency58State
    from research.direct.train_latency58 import load_source, state_sha256

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(), "Saved-state audit must stay on CPU")
    helpers = load_source("latency58_audit_helpers", plan["helper_source"])
    payload = torch.load(resume, map_location="cpu", weights_only=False)
    pointer = json.loads((run / "latest.json").read_text())
    require(payload["schema"] == "latency58-resume-v1" and type(payload["step"]) is int
            and 0 < payload["step"] <= plan["config"]["steps"]
            and payload["plan_sha256"] == args.plan_sha256
            and pointer["step"] == payload["step"] and pointer["resume_sha256"] == args.resume_sha256
            and sha(pointer["receipt_path"]) == pointer["receipt_sha256"], "Saved receipt or schema differs")
    step, config = payload["step"], plan["config"]
    provenance = payload["provenance"]
    require(plan["schema"] == "latency58-followup-training-plan-v1"
            and provenance["training_updates"] == plan["parent_training_updates"] + step
            and provenance["pilot_updates"] == plan["parent_hop128_updates"] + step
            and provenance["tail_updates"] == step
            and provenance["parent_checkpoint"] == plan["parent_checkpoint"]
            and provenance["parent_model_state_sha256"] == plan["initial_model_state_sha256"]
            and provenance["training_objective"] == plan["objective"], "Saved follow-up lineage differs")
    require(payload["next_sample_index"] == config["data_start"] + step * config["batch_size"],
            "Saved data counter differs")
    journal = (run / "metrics.jsonl").read_bytes()
    helpers.validate_journal(journal, step, config)
    rows = [json.loads(line) for line in journal.decode().splitlines()]
    require(all(row["objective"] == plan["objective"]
                and len(row["augmented_batch_sha256"]) == 64
                and all(c in "0123456789abcdef" for c in row["augmented_batch_sha256"])
                for row in rows), "Missing matched augmented-batch identities")
    require(hashlib.sha256(journal).hexdigest() == payload["metrics_sha256"] == pointer["metrics_sha256"]
            and len(journal) == payload["metrics_bytes"], "Saved journal prefix differs")
    model = Latency58Model.from_accepted()
    frozen = {name: value.clone() for name, value in model.named_buffers()}
    require(payload["architecture"] == model.architecture_metadata, "Saved framing differs")
    require(set(payload["model"]) == set(model.state_dict())
            and all(value.dtype == torch.float32 and value.device.type == "cpu"
                    and torch.isfinite(value).all().item() for value in payload["model"].values()),
            "Saved weights are not the complete finite FP32 model")
    model.load_state_dict(payload["model"], strict=True)
    require(state_sha256(model.state_dict()) == payload["model_state_sha256"] == pointer["model_state_sha256"]
            and all(torch.equal(value, frozen[name]) for name, value in model.named_buffers()),
            "Saved model hash or fixed buffers differ")
    names = [name for name, _ in model.named_parameters()]
    require(payload["optimizer_parameter_names"] == names and len(names) == 21,
            "Saved optimizer parameter ordering differs")
    optimizer = payload["optimizer"]
    require(set(optimizer) == {"state", "param_groups"} and len(optimizer["param_groups"]) == 1
            and set(optimizer["state"]) == set(range(21)), "Saved Adam inventory differs")
    helpers.validate_adam_group(optimizer["param_groups"][0], list(range(21)), step, config)
    for index, (_, parameter) in enumerate(model.named_parameters()):
        state = optimizer["state"][index]
        require(set(state) == {"step", "exp_avg", "exp_avg_sq"}
                and state["step"].shape == () and state["step"].dtype == torch.float32
                and float(state["step"]) == step, "Adam update count differs")
        for name in ("exp_avg", "exp_avg_sq"):
            value = state[name]
            require(value.shape == parameter.shape and value.dtype == torch.float32
                    and value.device.type == "cpu" and torch.isfinite(value).all().item(),
                    "Saved Adam moment shape/dtype/values differ")
        require((state["exp_avg_sq"] >= 0).all().item(), "Adam squared moment is negative")
    rng = payload["rng"]
    require(set(rng) == {"python", "numpy", "torch_cpu", "torch_cuda"}
            and isinstance(rng["torch_cuda"], list) and len(rng["torch_cuda"]) == 1,
            "Saved random-state inventory differs")
    for value in (rng["torch_cpu"], rng["torch_cuda"][0]):
        require(value.dtype == torch.uint8 and value.device.type == "cpu" and value.ndim == 1
                and value.numel() > 0, "Random-state bytes malformed")
    require(rng["torch_cpu"].shape == torch.get_rng_state().shape, "CPU RNG shape differs")
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        audio = torch.linspace(-0.1, 0.1, 512).reshape(1, 2, 256)
        a, b = model.render(audio), model.render(audio)
        require(torch.equal(a.deployed, b.deployed)
                and all(torch.equal(x, y) for x, y in zip(a.state, b.state, strict=True)),
                "Saved model reset replay differs")
        require(isinstance(a.state, Latency58State) and torch.isfinite(a.deployed).all().item()
                and float((a.deployed.sum(dim=1) - a.delayed_mixture).abs().max()) <= 1e-6,
                "Saved model output ABI or reconstruction failed")
    if pointer["inference"] is not None:
        snapshot = pointer["inference"]
        require(sha(snapshot["path"]) == snapshot["sha256"], "Inference snapshot changed")
        saved = torch.load(snapshot["path"], map_location="cpu", weights_only=True)
        require(saved["schema"] == "latency58-inference-v1" and saved["step"] == step
                and state_sha256(saved["model"]) == payload["model_state_sha256"],
                "Inference snapshot differs from resumable weights")
    require(sha(resume) == args.resume_sha256 and not torch.cuda.is_initialized(),
            "Audit changed checkpoint or initialized CUDA")
    result = {"status": "pass", "step": step, "resume_sha256": args.resume_sha256,
              "model_state_sha256": payload["model_state_sha256"], "next_sample_index": payload["next_sample_index"],
              "parameter_tensors": 21, "buffer_tensors": 5, "adam_state_pairs": 21,
              "fixed_buffers_unchanged": True, "exact_reset_replay": True,
              "cpu_cuda_rng_bytes_present": True, "source_bindings_unchanged": True,
              "training_updates_executed": 0, "optimizer_instances": 0, "cuda_initialized": False,
              "plan_sha256": args.plan_sha256, "auditor_sha256": sha(__file__),
              "parent_checkpoint": plan["parent_checkpoint"], "objective": plan["objective"],
              "cumulative_training_updates": provenance["training_updates"],
              "cumulative_hop128_updates": provenance["pilot_updates"], "batch_identity_rows": len(rows)}
    require(all(sha(path) == digest for path, digest in plan["source_bindings"].items()), "An audit input changed")
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
