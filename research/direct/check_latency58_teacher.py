"""Check frozen-teacher physical alignment and student-only gradients on CPU."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from research.direct.latency58_checkpoint import require, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Check plan changed")
    plan = json.loads(args.plan.read_text())
    require(plan["schema"] == "latency58-teacher-functional-plan-v1"
            and all(sha(p) == h for p, h in plan["source_bindings"].items()), "Check inputs changed")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CPU1 with CUDA hidden")
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve prior result")
    import torch
    import torch.nn.functional as F
    from research.direct.latency58_teacher import load_frozen_teacher, physical_teacher_targets, deployed_teacher_l1
    from research.direct.latency58_asymmetric_checkpoint import load_model_state, make_model
    from research.direct.latency58_evaluate import model_state_sha256

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260907)
    torch.use_deterministic_algorithms(True)
    began = time.monotonic()
    teacher, identity = load_frozen_teacher(plan["teacher_export_plan"])
    student = make_model(plan["parent_checkpoint"])
    require(load_model_state(student, plan["checkpoint"]) == plan["step"]
            and model_state_sha256(student) == plan["model_state_sha256"], "Student identity differs")
    teacher_sha = model_state_sha256(teacher)
    require(teacher_sha == identity["model_state_sha256"], "Teacher tensor identity differs")
    rows = []
    for samples in (1, 127, 128, 129, 255, 256, 257, 1025):
        x = torch.randn(1, 2, samples) * .03
        x[..., -1] = torch.tensor([.3125, -.21875])
        rng = torch.get_rng_state().clone()
        target = physical_teacher_targets(teacher, x)
        state = teacher.initial_state(1)
        padded = F.pad(x, (0, (-samples) % 256))
        pieces = []
        with torch.no_grad():
            for chunk in padded.split(256, dim=-1):
                piece, state = teacher.forward_chunk(chunk, state)
                pieces.append(piece)
            piece, state = teacher.flush(state)
            pieces.append(piece)
        literal = torch.cat(pieces, dim=-1)[..., 256:256 + samples]
        parity = float((literal - target).abs().max())
        closure = float((target.sum(1) - x).abs().max())
        require(parity <= 1e-6 and closure <= 1e-6 and torch.equal(rng, torch.get_rng_state()),
                "Teacher literal/grouped, last-sample closure or RNG failed")
        require(torch.equal(target, physical_teacher_targets(teacher, x)), "Teacher reset replay differs")
        rows.append({"samples": samples, "literal_max_abs": parity, "closure_max_abs": closure,
                     "one_flush": True, "last_real_sample_recovered": True, "reset_exact": True})
    student.train().requires_grad_(True)
    x = (torch.randn(2, 2, 1025) * .04).requires_grad_(True)
    target = physical_teacher_targets(teacher, x)
    output = student.render(F.pad(x, (0, (-x.shape[-1]) % 128 + 128)))
    deployed = output.deployed[..., 128:128 + x.shape[-1]]
    require(torch.equal(output.delayed_mixture[..., 128:128 + x.shape[-1]], x), "Student physical alignment differs")
    loss = deployed_teacher_l1(deployed, target)
    loss.backward()
    gradients = {name: float(p.grad.abs().max()) if p.grad is not None else None for name, p in student.named_parameters()}
    require(len(gradients) == 21 and all(v is not None and v > 0 for v in gradients.values())
            and all(bool(torch.isfinite(p.grad).all()) for p in student.parameters())
            and x.grad is not None and bool(torch.isfinite(x.grad).all()) and bool((x.grad[..., -1].abs() > 0).all())
            and all(not p.requires_grad and p.grad is None for p in teacher.parameters()), "Student/frozen-teacher gradients failed")
    require(model_state_sha256(teacher) == teacher_sha and model_state_sha256(student) == plan["model_state_sha256"]
            and not torch.cuda.is_initialized() and all(sha(p) == h for p, h in plan["source_bindings"].items()),
            "Teacher/student tensors or CPU/source scope changed")
    result = {"status": "pass", "plan_sha256": args.plan_sha256, "source_bindings_unchanged": True,
              "teacher_identity": identity, "student_checkpoint": plan["checkpoint"], "cases": rows,
              "batch2_student_only_gradient_max_abs": gradients, "auxiliary_l1": float(loss.detach()),
              "final_real_input_gradient_nonzero": True, "teacher_parameters_and_input_target_detached": True,
              "no_optimizer_or_cuda_or_saved_weights": True, "elapsed_seconds": time.monotonic() - began}
    with (out / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
