"""CPU checks for both SDR teachers, including independent streaming oracles."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from research.direct.latency58_checkpoint import require, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Functional plan changed")
    plan = json.loads(args.plan.read_text())
    require(plan["schema"] == "latency58-sdr-teacher-functional-v1"
            and all(sha(p) == h for p, h in plan["source_bindings"].items()), "Functional inputs changed")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CPU1 with CUDA hidden")
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve previous evidence")
    import torch
    import torch.nn.functional as F
    from research.direct.latency58_sdr_teacher import (
        STUDENT_STATE_SHA256, load_initial_student, load_teacher, physical_targets)
    from research.direct.latency58_teacher import deployed_teacher_l1
    from research.direct.train_latency58 import state_sha256

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260908)
    torch.use_deterministic_algorithms(True)
    began = time.monotonic()
    rng_before = torch.get_rng_state().clone()
    student = load_initial_student()
    require(torch.equal(torch.get_rng_state(), rng_before), "Student loader changed RNG")
    teacher_results = {}
    for kind in ("cropped11", "c91"):
        rng_before = torch.get_rng_state().clone()
        teacher, identity = load_teacher(kind, plan["teachers"][kind])
        require(torch.equal(torch.get_rng_state(), rng_before), "Teacher loader changed RNG")
        hop = identity["graph_alignment_samples"]
        rows = []
        for samples in (1, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1025, 4097):
            x = torch.randn(2, 2, samples) * .03
            x[..., -1] = torch.tensor([[.3125, -.21875], [-.0625, .1875]])
            original = x.clone()
            rng = torch.get_rng_state().clone()
            target = physical_targets(teacher, x, kind=kind)
            references = []
            with torch.no_grad():
                for example in x:
                    padded = F.pad(example, (0, (-samples) % hop))
                    pieces = []
                    if kind == "c91":
                        transform = teacher.init_stateful_transform_fn(device=torch.device("cpu"))
                        pieces = [transform(chunk) for chunk in padded.split(hop, dim=-1)]
                        pieces.append(transform(torch.zeros_like(padded[..., :hop])))
                        raw = torch.cat(pieces, dim=-1)[..., hop:hop + samples]
                        dbv = raw[:3]
                        expected = torch.cat((dbv, example[None] - dbv.sum(dim=0, keepdim=True)), dim=0)
                    else:
                        state = teacher.initial_state(1)
                        for chunk in padded.split(hop, dim=-1):
                            piece, state = teacher.forward_chunk(chunk[None], state)
                            pieces.append(piece)
                        piece, state = teacher.flush(state)
                        pieces.append(piece)
                        expected = torch.cat(pieces, dim=-1)[0, ..., hop:hop + samples]
                    references.append(expected)
            literal = torch.stack(references)
            error = float((literal - target).abs().max())
            closure = float((target.sum(dim=1) - x).abs().max())
            require(error <= 1e-5 and closure <= 1e-6, "Independent physical target or closure mismatch")
            require(torch.equal(target, physical_targets(teacher, x, kind=kind))
                    and torch.equal(original, x) and torch.equal(rng, torch.get_rng_state()),
                    "Reset, input immutability or RNG isolation failed")
            rows.append({"samples": samples, "batch": 2, "oracle_max_abs": error,
                         "closure_max_abs": closure, "reset_exact": True, "final_real_sample_retained": True})

        x = torch.randn(1, 2, 8192) * .03
        full = physical_targets(teacher, x, kind=kind)
        prefix = physical_targets(teacher, x[..., :3072], kind=kind)
        prefix_error = float((full[..., :3072 - hop] - prefix[..., :3072 - hop]).abs().max())
        require(prefix_error <= 1e-5,
                "Teacher uses samples beyond its declared streaming delay")
        student.train().requires_grad_(True)
        student.zero_grad(set_to_none=True)
        mixture = (torch.randn(2, 2, 1025) * .04).requires_grad_()
        target = physical_targets(teacher, mixture, kind=kind)
        output = student.render(F.pad(mixture, (0, (-mixture.shape[-1]) % 128 + 128)))
        deployed = output.deployed[..., 128:128 + mixture.shape[-1]]
        loss = deployed_teacher_l1(deployed, target)
        loss.backward()
        gradients = {name: float(p.grad.abs().max()) if p.grad is not None else None
                     for name, p in student.named_parameters()}
        require(len(gradients) == 21 and all(v is not None and v > 0 for v in gradients.values())
                and all(bool(torch.isfinite(p.grad).all()) for p in student.parameters())
                and mixture.grad is not None and bool(torch.isfinite(mixture.grad).all())
                and bool((mixture.grad[..., -1].abs() > 0).all()), "Student gradient coverage failed")
        require(all(not p.requires_grad and p.grad is None for p in teacher.parameters())
                and state_sha256(teacher.state_dict()) == identity["model_state_sha256"]
                and state_sha256(student.state_dict()) == STUDENT_STATE_SHA256,
                "A frozen teacher or initial student tensor changed")
        teacher_results[kind] = {"identity": identity, "cases": rows,
                                 "prefix_causality_with_declared_delay": True,
                                 "prefix_max_abs": prefix_error,
                                 "student_gradient_max_abs": gradients,
                                 "teacher_and_target_detached": True,
                                 "final_real_input_gradient_nonzero": True}
        student.zero_grad(set_to_none=True)
        del teacher, target, output, deployed, loss
    require(not torch.cuda.is_initialized() and all(sha(p) == h for p, h in plan["source_bindings"].items()),
            "CPU scope or functional source binding changed")
    result = {"schema": "latency58-sdr-teacher-functional-result-v1", "status": "pass",
              "plan_sha256": args.plan_sha256, "teachers": teacher_results,
              "student_model_state_sha256": STUDENT_STATE_SHA256, "source_bindings_unchanged": True,
              "no_optimizer_or_cuda_or_saved_weights": True, "elapsed_seconds": time.monotonic() - began}
    with (out / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "pass", "teacher_count": 2, "cases_per_teacher": 12,
                      "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
