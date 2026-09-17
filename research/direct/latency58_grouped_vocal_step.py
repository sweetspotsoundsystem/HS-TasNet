"""One prospective ordinary-plus-source-view Adam/EMA update.

Inputs are the complete, already augmented ordinary crop. The auxiliary
history is derived before warmup; every rendered microbatch starts fresh.
This helper is not wired into the live or a future production trainer.
"""
import torch

from research.direct.run_latency58_quality import require
from research.direct.latency58_branch_ema import BranchParameterEMA
from research.direct.latency58_branch_memory_context import render_scored_context
from research.direct.latency58_grouped_vocal_auxiliary import source_views, prepare_groups, contribution
from research.direct.train_latency58_branch_sdr_ema import advance_with_ema


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
    auxiliary_mix, auxiliary_targets = source_views(mixture_cpu, targets_cpu)
    device = parameters[0].device
    inputs = {"ordinary": (mixture_cpu.to(device), targets_cpu.to(device)),
              "auxiliary": (auxiliary_mix.to(device), auxiliary_targets.to(device))}
    groups = prepare_groups(inputs["ordinary"][1][..., warmup_samples:],
                            inputs["auxiliary"][1][..., warmup_samples:])
    optimizer.zero_grad(set_to_none=True)
    rows = {}
    for group, microbatch in (("ordinary", ordinary_microbatch), ("auxiliary", auxiliary_microbatch)):
        mixture, targets = inputs[group]
        active = torch.zeros(4, dtype=torch.int64, device=device)
        absent = torch.zeros_like(active)
        micro_rows, total = [], 0.
        for offset in range(0, mixture.shape[0], microbatch):
            if check_continue is not None:
                check_continue()
            audio, truth = mixture[offset:offset + microbatch], targets[offset:offset + microbatch, ..., warmup_samples:]
            output = render_scored_context(model, audio, warmup_samples=warmup_samples, carry_state=True)
            require(output.initial_state_detached and output.flush_hops == 1
                    and output.scored_samples == truth.shape[-1], "Grouped context contract changed")
            value, terms = contribution(group, output.raw, output.deployed, truth, output.physical_mixture, groups)
            value.backward()
            active += terms.active_window_counts
            absent += terms.absent_window_counts
            loss = float(value.detach())
            total += loss
            micro_rows.append({"offset": offset, "examples": audio.shape[0], "weighted_loss": loss})
            del audio, truth, output, value, terms
        reduction = getattr(groups, group)
        require(torch.equal(active, reduction.active) and torch.equal(absent, reduction.absent),
                "Grouped accumulation missed or duplicated activity windows")
        rows[group] = {"examples": mixture.shape[0], "weighted_loss": total,
                       "active_windows": active.cpu().tolist(), "absent_windows": absent.cpu().tolist(),
                       "microbatches": micro_rows}
        if after_group is not None:
            after_group(group, rows[group])
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
    return {"step": step, "groups": rows, "weighted_loss": sum(r["weighted_loss"] for r in rows.values()),
            "gradient_norm_before_clip": float(norm), "parameter_gradient_norms": gradient_norms, **endpoint}
