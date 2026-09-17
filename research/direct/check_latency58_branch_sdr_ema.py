"""Check that the production Adam/EMA hook leaves CPU raw training unchanged."""
import argparse
import copy
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs


def check(plan):
    import torch
    from research.direct.latency58_branch_memory_checkpoint import load_model, audit_live
    from research.direct.latency58_branch_memory_context import render_scored_context
    from research.direct.latency58_branch_sdr_blend import objective, VERSION, SDR_WEIGHT
    from research.direct.latency58_branch_ema import BranchParameterEMA
    from research.direct.train_latency58_branch_sdr_ema import advance_with_ema, compare_ema_to_fp64
    parent, _ = load_model(plan["parent_checkpoint"])
    parent_sha = state_sha256(parent.state_dict())
    require(parent_sha == plan["parent_model_state_sha256"] and plan["objective_version"] == VERSION
            and plan["direct_sdr_weight"] == SDR_WEIGHT == .2,
            "Raw/EMA control parent or objective changed")
    models = [copy.deepcopy(parent).train().requires_grad_(True) for _ in range(2)]
    for model in models:
        model.training_precision = "fp32"
    optimizers = [torch.optim.Adam(model.parameters(), lr=6e-5, foreach=False) for model in models]
    frozen = {name: value.clone() for name, value in parent.named_buffers()}
    ema = BranchParameterEMA(models[1], decay=plan["ema_decay"], base_state_sha256=parent_sha)
    reference = {name: parameter.detach().double().clone() for name, parameter in parent.named_parameters()}
    generator = torch.Generator().manual_seed(202610271)
    rows, arithmetic = [], []
    for step in range(1, 4):
        truth = .03 * torch.randn(1, 4, 2, 132224, generator=generator)
        if step > 1:
            truth[:, step - 1] = 0
        mixture = truth.sum(1)
        targets = truth[..., 88064:]
        before, updates = [], []
        for index, (model, optimizer) in enumerate(zip(models, optimizers, strict=True)):
            optimizer.param_groups[0]["lr"] = step * 6e-7
            optimizer.zero_grad(set_to_none=True)
            output = render_scored_context(model, mixture, warmup_samples=88064, carry_state=True)
            require(output.initial_state_detached and output.scored_samples == 44160
                    and torch.equal(output.physical_mixture, mixture[..., 88064:]),
                    "Control lost full training context or physical alignment")
            terms = objective(output.raw, output.deployed, targets, output.physical_mixture)
            terms.total.backward()
            gradients = {name: parameter.grad for name, parameter in model.named_parameters()}
            require(len(gradients) == 40 and all(value is not None and bool(torch.isfinite(value).all())
                    and float(value.norm()) > 0 for value in gradients.values()),
                    "Control lacks a finite nonzero parameter gradient")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True, foreach=False)
            before.append({"loss": float(terms.total.detach()), "negative_sdr_db": float(terms.negative_sdr_db.detach()),
                           "grad_norm": float(norm), "clipped_gradient_sha256": state_sha256(gradients),
                           "outputs_sha256": state_sha256({"raw": output.raw, "deployed": output.deployed}),
                           "active_windows": terms.active_window_counts.tolist(),
                           "absent_windows": terms.absent_window_counts.tolist()})
            del gradients, output, terms
            rng_before = torch.get_rng_state()
            if index == 0:
                optimizer.step()
            else:
                updates.append(advance_with_ema(model, optimizer, ema, step=step))
            require(torch.equal(rng_before, torch.get_rng_state()), "Adam/EMA hook changed CPU RNG state")
            audit_live(model, optimizer, step, frozen)
        require(before[0] == before[1] and state_sha256(models[0].state_dict()) == state_sha256(models[1].state_dict()),
                "EMA changed raw outputs, gradients, objective or Adam weights")
        for p, q in zip(models[0].parameters(), models[1].parameters(), strict=True):
            require(all(torch.equal(optimizers[0].state[p][key], optimizers[1].state[q][key])
                        for key in ("step", "exp_avg", "exp_avg_sq")), "EMA changed raw Adam state")
        arithmetic.append(compare_ema_to_fp64(models[1], ema, reference))
        rows.append({"step": step, "lr": step * 6e-7, "input_sha256": state_sha256({"mixture": mixture, "truth": truth}),
                     **before[0], **updates[0], "raw_control_bit_exact": True})
        print(json.dumps({"event": "raw_ema_control_update", "step": step, "raw_control_bit_exact": True,
                          "ema_fp64_max_abs_error": arithmetic[-1]["maximum_absolute_error"]}), flush=True)
    require(rows[1]["absent_windows"] == [0, 1, 0, 0] and rows[2]["absent_windows"] == [0, 0, 1, 0],
            "CPU control did not exercise the planned absent-source terms")
    with torch.inference_mode():
        audio = mixture[..., :8 * 128]
        first, second = (model.eval().render(audio) for model in models)
        require(len(first.state) == len(second.state) == 8
                and all(torch.equal(a.view(torch.int32), b.view(torch.int32))
                        for a, b in zip(first.state, second.state, strict=True)), "Raw control stream states differ")
    require(state_sha256(parent.state_dict()) == parent_sha and not torch.cuda.is_initialized(),
            "CPU control modified the parent or used GPU")
    return {"status": "pass", "objective_version": VERSION, "direct_sdr_weight": SDR_WEIGHT,
            "updates": rows, "ema_arithmetic": arithmetic, "ema_decay": ema.decay,
            "production_optimizer_ema_hook_exercised": True, "all_40_raw_gradients_and_adam_states_bit_exact": True,
            "raw_control_audio_and_eight_states_bit_exact": True, "cpu_rng_unchanged_by_optimizer_ema_hook": True,
            "warmup_samples": 88064, "scored_samples": 44160, "fixture_batch_size": 1,
            "fixture_precision": "fp32", "absent_bass_and_vocal_terms_exercised": True,
            "parent_unchanged": True, "gpu_used": False, "checkpoint_files_written": False, "quality_measured": False,
            "limitation": "Synthetic CPU B1 FP32 control using the production objective, full context and optimizer/EMA hook. The B16 BF16 recorded-data loader, GPU arithmetic checks, monitor and paired full14 launcher still require the future GPU rehearsal and actual endpoint execution."}


def main():
    import torch
    from research.direct.latency58_sdr_checkpoint import require_space
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require frozen inputs and CUDA-hidden CPU1 control")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    plan = read(args.plan)
    out = Path(plan["output_directory"])
    require(plan["schema"] == "latency58-branch-sdr-ema-control-plan-v1"
            and out.resolve() == args.plan.parent.resolve() and out.resolve().is_relative_to(PHASE)
            and not (out / "result.json").exists(), "Preserve the raw/EMA control result")
    verify_inputs(plan)
    counted = require_space(plan, 450_000_000)
    started = time.monotonic()
    result = check(plan)
    verify_inputs(plan)
    require_space(plan, 450_000_000)
    result.update(plan_sha256=args.plan_sha256, source_bindings=plan["source_bindings"],
                  elapsed_seconds=time.monotonic() - started, counted_bytes_before=counted,
                  forecast_bytes=counted + 450_000_000 + plan["outside_roots_reservation_bytes"])
    write(out / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("source_bindings", "ema_arithmetic", "updates")}), flush=True)


if __name__ == "__main__":
    main()
