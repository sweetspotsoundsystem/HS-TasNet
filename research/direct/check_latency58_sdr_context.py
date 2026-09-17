"""Bounded CPU proof for real-prefix warmup and its matched reset control."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import read, require, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Context fixture plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-sdr-context-functional-v1"
            and all(sha(p) == s for p, s in plan["source_bindings"].items()), "Context fixture inputs changed")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use a CUDA-hidden CPU1 fixture")
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve previous context fixture")
    import torch
    import torch.nn.functional as F
    from research.direct.latency58_sdr_context import render_scored_context, physical_context_teacher
    from research.direct.latency58_sdr_teacher import load_initial_student, load_teacher, STUDENT_STATE_SHA256
    from research.direct.latency58_teacher import deployed_teacher_l1
    from research.direct.train_latency58 import state_sha256

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260909)
    torch.use_deterministic_algorithms(True)
    began = time.monotonic()
    model = load_initial_student().train().requires_grad_(True)
    rows = []
    for warmup, count in plan["geometry_cases"]:
        x = torch.randn(2, 2, warmup + count) * .03
        x[..., -1] = torch.tensor([[.3125, -.21875], [-.0625, .1875]])
        original, rng = x.clone(), torch.get_rng_state().clone()
        with torch.no_grad():
            padded = F.pad(x, (0, (-count) % 128 + 128))
            state, pieces = model.initial_state(2), []
            for chunk in padded.split(128, dim=-1):
                value = model.render(chunk, state)
                state = value.state
                pieces.append(value)
            literal_raw = torch.cat([p.raw for p in pieces], dim=-1)[..., warmup + 128:warmup + 128 + count]
            literal_deployed = torch.cat([p.deployed for p in pieces], dim=-1)[..., warmup + 128:warmup + 128 + count]
            warm = render_scored_context(model, x, warmup_samples=warmup, carry_state=True)
            replay = render_scored_context(model, x, warmup_samples=warmup, carry_state=True)
            reset = render_scored_context(model, x, warmup_samples=warmup, carry_state=False)
            reset_reference = model.render(F.pad(x[..., warmup:], (0, (-count) % 128 + 128)))
        raw_error = float((warm.raw - literal_raw).abs().max())
        deployed_error = float((warm.deployed - literal_deployed).abs().max())
        require(raw_error <= 1e-6 and deployed_error <= 1e-6
                and torch.equal(reset.raw, reset_reference.raw[..., 128:128 + count])
                and torch.equal(reset.deployed, reset_reference.deployed[..., 128:128 + count])
                and torch.equal(warm.raw, replay.raw) and torch.equal(warm.deployed, replay.deployed)
                and torch.equal(warm.physical_mixture, x[..., warmup:])
                and torch.equal(reset.physical_mixture, x[..., warmup:])
                and float((warm.deployed.sum(dim=1) - x[..., warmup:]).abs().max()) <= 1e-6
                and torch.equal(original, x) and torch.equal(rng, torch.get_rng_state()),
                "Context streaming, exact reset control, input immutability or physical alignment failed")
        rows.append({"warmup_samples": warmup, "scored_samples": count, "batch": 2,
                     "literal_raw_max_abs": raw_error, "literal_deployed_max_abs": deployed_error,
                     "reset_control_exact": True, "reset_replay_exact": True, "physical_alignment_exact": True,
                     "warm_vs_reset_max_abs": float((warm.deployed - reset.deployed).abs().max())})
        del pieces, warm, reset, replay, reset_reference, literal_raw, literal_deployed, value
    require(any(r["warm_vs_reset_max_abs"] > 1e-5 for r in rows), "Prefix never affects the student")
    teachers = {}
    for kind in ("cropped11", "c91"):
        teacher, identity = load_teacher(kind, plan["teachers"][kind])
        x = (torch.randn(2, 2, 1024 + 1025) * .03).requires_grad_()
        original, rng = x.detach().clone(), torch.get_rng_state().clone()
        target = physical_context_teacher(teacher, x, kind=kind, warmup_samples=1024)
        hop = identity["graph_alignment_samples"]
        references = []
        with torch.no_grad():
            for example in x.detach():
                padded = F.pad(example, (0, (-example.shape[-1]) % hop))
                if kind == "c91":
                    transform = teacher.init_stateful_transform_fn(device=torch.device("cpu"))
                    pieces = [transform(chunk) for chunk in padded.split(hop, dim=-1)]
                    pieces.append(transform(torch.zeros_like(padded[..., :hop])))
                    raw = torch.cat(pieces, dim=-1)[..., hop:hop + example.shape[-1]]
                    dbv = raw[:3]
                    physical = torch.cat((dbv, example[None] - dbv.sum(dim=0, keepdim=True)), dim=0)
                else:
                    state, pieces = teacher.initial_state(1), []
                    for chunk in padded.split(hop, dim=-1):
                        piece, state = teacher.forward_chunk(chunk[None], state)
                        pieces.append(piece)
                    piece, state = teacher.flush(state)
                    pieces.append(piece)
                    physical = torch.cat(pieces, dim=-1)[0, ..., hop:hop + example.shape[-1]]
                references.append(physical[..., 1024:])
        error = float((target - torch.stack(references)).abs().max())
        require(error <= 1e-6 and torch.equal(rng, torch.get_rng_state()), "Teacher context oracle or RNG differs")
        gradient_rows = []
        for carry in (False, True):
            model.zero_grad(set_to_none=True)
            x.grad = None
            scored = render_scored_context(model, x, warmup_samples=1024, carry_state=carry)
            loss = scored.raw.square().mean() + .5 * deployed_teacher_l1(scored.deployed, target)
            loss.backward()
            require(len(list(model.parameters())) == 21
                    and all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                            and bool((p.grad != 0).any()) for p in model.parameters())
                    and x.grad is not None and bool(torch.isfinite(x.grad).all())
                    and torch.count_nonzero(x.grad[..., :1024]).item() == 0
                    and bool((x.grad[..., -1] != 0).all()) and scored.initial_state_detached,
                    "Warmup gradient cutoff, parameter gradients or final real sample failed")
            del scored, loss
            model.zero_grad(set_to_none=True)
            x.grad = None
            scored = render_scored_context(model, x, warmup_samples=1024, carry_state=carry)
            scored.deployed.sum(dim=1)[..., -1].sum().backward()
            expected = torch.zeros_like(x)
            expected[..., -1] = 1
            gradient_error = float((x.grad - expected).abs().max())
            require(gradient_error <= 1e-6, "Final flush fails analytic mixture gradient")
            gradient_rows.append({"carry_state": carry, "finite_nonzero_parameter_gradients": 21,
                                  "prefix_gradient_exactly_zero": True, "final_sample_gradient_nonzero": True,
                                  "analytic_final_flush_gradient_max_abs": gradient_error})
            del scored
        require(torch.equal(original, x) and all(not p.requires_grad and p.grad is None for p in teacher.parameters())
                and state_sha256(teacher.state_dict()) == identity["model_state_sha256"], "Teacher or input changed")
        teachers[kind] = {"identity": identity, "context_oracle_max_abs": error, "gradients": gradient_rows}
        del teacher, target, x, references, pieces
    model.zero_grad(set_to_none=True)
    require(state_sha256(model.state_dict()) == STUDENT_STATE_SHA256 and not torch.cuda.is_initialized()
            and all(sha(p) == s for p, s in plan["source_bindings"].items()), "Fixture changed model or inputs")
    result = {"schema": "latency58-sdr-context-functional-result-v1", "status": "pass",
              "plan_sha256": args.plan_sha256, "geometry": rows, "teachers": teachers,
              "model_state_sha256": STUDENT_STATE_SHA256, "source_bindings_unchanged": True,
              "optimizer_updates": 0, "cuda_initialized": False, "saved_weights": False,
              "scope": "Functional preparation only; no context training or quality improvement is established",
              "elapsed_seconds": time.monotonic() - began}
    write(out / "result.json", result)
    print(json.dumps({"status": "pass", "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
