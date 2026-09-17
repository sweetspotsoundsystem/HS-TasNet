"""Replay ordinary gradients and the selected quarter-vocal auxiliary scalar.

Capturing detached output coordinates costs an extra forward pass. No neural
activation graph spans microbatches, and only one Adam/EMA update follows both
groups. Ordinary accumulation delegates to its unchanged implementation.
"""
import torch

from research.direct.run_latency58_quality import require
from research.direct.latency58_branch_ema import BranchParameterEMA
from research.direct.latency58_branch_memory_context import render_scored_context
from research.direct.latency58_grouped_vocal_auxiliary import source_views, prepare_groups, AUXILIARY_WEIGHT
from research.direct.latency58_weighted_vocal_auxiliary import objective as auxiliary_objective, VIEW_WEIGHTS
from research.direct.latency58_grouped_vocal_canonical import accumulate_group as original_accumulate_group
from research.direct.train_latency58_branch_sdr_ema import advance_with_ema

VERSION = "latency58-quarter-vocal-whole-group-output-vjp-replay-v1"


def policy():
    return {"version": VERSION, "loss": "Unchanged ordinary16 plus 0.1 joint source-view2 with view multipliers [1,0.25]",
            "capture": "Grad-enabled B4/B2 renders, detached output coordinates only; release each neural graph",
            "output_derivatives": "One FP32 complete-group scalar per group; auxiliary view weights retain complete-group denominators",
            "backward": "Replay each B4/B2 render with its exact whole-group output derivatives",
            "replay_outputs": "Require bit-exact raw and deployed outputs before every backward",
            "optimizer": "Clear gradients once, complete both groups, clip once, Adam once, EMA once",
            "inference_changed": False, "extra_forward_pass_per_update": True}


def accumulate_group(model, mixture, targets, *, group, microbatch, warmup_samples,
                     check_continue=None, verify_input_gradients=False, progress=None):
    if group == "ordinary":
        return original_accumulate_group(model, mixture, targets, group=group, microbatch=microbatch,
            warmup_samples=warmup_samples, check_continue=check_continue,
            verify_input_gradients=verify_input_gradients, progress=progress)
    require(group == "auxiliary" and len(mixture) == 2, "Require a complete auxiliary source group")
    require(type(microbatch) is int and 0 < microbatch <= len(mixture)
            and mixture.device == targets.device and not mixture.requires_grad and not targets.requires_grad,
            "Require fixed group inputs and a valid microbatch")
    captured = [[], []]
    for offset in range(0, len(mixture), microbatch):
        if check_continue is not None:
            check_continue()
        output = render_scored_context(model, mixture[offset:offset + microbatch],
                                       warmup_samples=warmup_samples, carry_state=True)
        require(output.initial_state_detached and output.flush_hops == 1, "Capture context contract changed")
        captured[0].append(output.raw.detach().clone())
        captured[1].append(output.deployed.detach().clone())
        del output
        if progress is not None:
            progress("canonical_capture", group, offset)
    raw, deployed = (torch.cat(values).requires_grad_() for values in captured)
    del captured
    terms = auxiliary_objective(raw, deployed, targets[..., warmup_samples:], mixture[..., warmup_samples:])
    value = AUXILIARY_WEIGHT * terms.total
    loss = float(value.detach())
    view_losses = (AUXILIARY_WEIGHT * terms.weighted_view_contributions).detach().cpu().tolist()
    unweighted_view_losses = (AUXILIARY_WEIGHT * terms.unweighted_view_contributions).detach().cpu().tolist()
    active, absent = terms.active_window_counts.cpu().tolist(), terms.absent_window_counts.cpu().tolist()
    derivatives = tuple(v.detach() for v in torch.autograd.grad(value, (raw, deployed)))
    expected = raw.detach(), deployed.detach()
    del raw, deployed, value, terms
    rows = []
    for offset in range(0, len(mixture), microbatch):
        if check_continue is not None:
            check_continue()
        end = offset + microbatch
        audio = mixture[offset:end]
        if verify_input_gradients:
            audio = audio.detach().clone().requires_grad_()
        output = render_scored_context(model, audio, warmup_samples=warmup_samples, carry_state=True)
        require(output.initial_state_detached and output.flush_hops == 1
                and torch.equal(output.raw, expected[0][offset:end])
                and torch.equal(output.deployed, expected[1][offset:end]), "Canonical replay outputs differ")
        torch.autograd.backward((output.raw, output.deployed), tuple(v[offset:end] for v in derivatives))
        if verify_input_gradients:
            require(audio.grad is not None and bool(torch.isfinite(audio.grad).all())
                    and torch.count_nonzero(audio.grad[..., :warmup_samples]) == 0
                    and torch.count_nonzero(audio.grad[..., warmup_samples:]) > 0,
                    "Canonical warmup or scored-input gradients differ")
        rows.append({"offset": offset, "examples": len(audio), "whole_group_output_derivatives_replayed": True})
        del audio, output
        if progress is not None:
            progress("canonical_backward", group, offset)
    return {"examples": len(mixture), "weighted_loss": loss, "active_windows": active,
            "absent_windows": absent, "microbatches": rows, "replay_outputs_bit_exact": True,
            "whole_group_objective_evaluations": 1, "view_contribution_multipliers": list(VIEW_WEIGHTS),
            "weighted_view_losses": view_losses, "original_weight_view_losses": unweighted_view_losses}


def accumulate_groups(model, mixture_cpu, targets_cpu, *, warmup_samples, ordinary_microbatch=4,
                      auxiliary_microbatch=2, check_continue=None, after_group=None,
                      verify_input_gradients=False, progress=None):
    auxiliary_mix, auxiliary_targets = source_views(mixture_cpu, targets_cpu)
    device = next(model.parameters()).device
    inputs = {"ordinary": (mixture_cpu.to(device), targets_cpu.to(device)),
              "auxiliary": (auxiliary_mix.to(device), auxiliary_targets.to(device))}
    groups = prepare_groups(inputs["ordinary"][1][..., warmup_samples:], inputs["auxiliary"][1][..., warmup_samples:])
    rows = {}
    for group, microbatch in (("ordinary", ordinary_microbatch), ("auxiliary", auxiliary_microbatch)):
        row = accumulate_group(model, *inputs[group], group=group, microbatch=microbatch,
            warmup_samples=warmup_samples, check_continue=check_continue,
            verify_input_gradients=verify_input_gradients, progress=progress)
        reduction = getattr(groups, group)
        require(row["active_windows"] == reduction.active.cpu().tolist()
                and row["absent_windows"] == reduction.absent.cpu().tolist(), "Canonical group activity differs")
        rows[group] = row
        if after_group is not None:
            after_group(group, row)
    return rows


def grouped_update(model, optimizer, ema, mixture_cpu, targets_cpu, *, step,
                   warmup_samples, ordinary_microbatch=4, auxiliary_microbatch=2,
                   check_continue=None, after_group=None):
    """Accumulate both independently normalized groups, then clip/update once.

    Optional callbacks run only before the final update, and may raise to stop
    accumulation without changing weights, Adam moments or EMA. A later call
    clears any partial gradients. An exception during Adam/EMA is fatal: do
    not retry against a potentially partially updated endpoint.
    """
    parameters = list(model.parameters())
    require(type(ema) is BranchParameterEMA and type(step) is int and step == ema.updates + 1,
            "Grouped update must follow the contiguous EMA endpoint")
    require(type(optimizer) is torch.optim.Adam and len(parameters) == 40
            and model.training and all(p.requires_grad and p.dtype == torch.float32 for p in parameters)
            and len(optimizer.param_groups) == 1
            and [id(p) for p in optimizer.param_groups[0]["params"]] == [id(p) for p in parameters]
            and set(optimizer.state) == (set() if step == 1 else set(parameters))
            and all(state["step"].item() == step - 1 for state in optimizer.state.values()),
            "Grouped update requires all 40 parameters at one Adam endpoint")
    require(type(warmup_samples) is int and warmup_samples > 0 and warmup_samples % 128 == 0
            and mixture_cpu.device.type == targets_cpu.device.type == "cpu"
            and type(ordinary_microbatch) is int and 0 < ordinary_microbatch <= 16
            and type(auxiliary_microbatch) is int and 0 < auxiliary_microbatch <= 2,
            "Invalid grouped update input or microbatch geometry")
    if check_continue is not None:
        check_continue()
    optimizer.zero_grad(set_to_none=True)
    rows = accumulate_groups(model, mixture_cpu, targets_cpu, warmup_samples=warmup_samples,
        ordinary_microbatch=ordinary_microbatch, auxiliary_microbatch=auxiliary_microbatch,
        check_continue=check_continue, after_group=after_group)
    gradient_norms = {}
    for name, parameter in model.named_parameters():
        require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()),
                "Missing or nonfinite grouped gradient: " + name)
        gradient_norms[name] = float(parameter.grad.norm())
    if check_continue is not None:
        check_continue()
    norm = torch.nn.utils.clip_grad_norm_(parameters, 5., foreach=False, error_if_nonfinite=True)
    endpoint = advance_with_ema(model, optimizer, ema, step=step)
    require(ema.updates == step and len(optimizer.state) == 40
            and all(state["step"].item() == step for state in optimizer.state.values()),
            "Grouped update advanced Adam or EMA incorrectly")
    return {"step": step, "accumulation_policy": policy(), "groups": rows, "weighted_loss": sum(r["weighted_loss"] for r in rows.values()),
            "gradient_norm_before_clip": float(norm), "parameter_gradient_norms": gradient_norms, **endpoint}
