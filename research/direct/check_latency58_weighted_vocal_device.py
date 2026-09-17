"""Check quarter-vocal canonical accumulation and guarded device restart.

The original parameter, scalar-loss and warmup-gradient tolerances are retained.
Device resource entry still requires CUDA BF16 and the complete scored context.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import io
import json
import torch

from research.direct.run_latency58_quality import require
from research.direct.train_latency58 import state_sha256
from research.direct.latency58_branch_memory_context import render_scored_context
from research.direct.latency58_grouped_vocal_auxiliary import source_views, prepare_groups, AUXILIARY_WEIGHT
from research.direct.latency58_branch_sdr_blend import objective as whole_objective
from research.direct.latency58_weighted_vocal_auxiliary import objective as weighted_objective
from research.direct.latency58_weighted_vocal_canonical import accumulate_groups, policy as accumulation_policy


def compare_group_gradients(model, mixture_cpu, targets_cpu, *, warmup_samples, progress=None):
    device = next(model.parameters()).device
    fingerprint = state_sha256(model.state_dict())
    auxiliary = source_views(mixture_cpu, targets_cpu)
    inputs = {"ordinary": tuple(v.to(device) for v in (mixture_cpu, targets_cpu)),
              "auxiliary": tuple(v.to(device) for v in auxiliary)}
    groups = prepare_groups(inputs["ordinary"][1][..., warmup_samples:], inputs["auxiliary"][1][..., warmup_samples:])
    outputs, derivatives, losses, ordinary_reference = {}, {}, {}, {}
    for group, microbatch in (("ordinary", 4), ("auxiliary", 2)):
        audio, truth = inputs[group]
        raw, deployed = [], []
        for offset in range(0, len(audio), microbatch):
            result = render_scored_context(model, audio[offset:offset + microbatch], warmup_samples=warmup_samples, carry_state=True)
            raw.append(result.raw.detach().clone()); deployed.append(result.deployed.detach().clone())
            del result
            if progress is not None:
                progress("reference_capture", group, offset)
        raw, deployed = torch.cat(raw).requires_grad_(), torch.cat(deployed).requires_grad_()
        objective = whole_objective if group == "ordinary" else weighted_objective
        value = objective(raw, deployed, truth[..., warmup_samples:], audio[..., warmup_samples:]).total
        if group == "auxiliary":
            value = value * AUXILIARY_WEIGHT
        losses[group] = float(value.detach())
        derivatives[group] = tuple(v.detach() for v in torch.autograd.grad(value, (raw, deployed)))
        outputs[group] = raw.detach(), deployed.detach()
        del raw, deployed, value
    try:
        model.zero_grad(set_to_none=True)
        for group, microbatch in (("ordinary", 4), ("auxiliary", 2)):
            audio, truth = inputs[group]
            for offset in range(0, len(audio), microbatch):
                end = offset + microbatch
                physical = audio[offset:end].detach().clone().requires_grad_()
                result = render_scored_context(model, physical, warmup_samples=warmup_samples, carry_state=True)
                require(torch.equal(result.raw, outputs[group][0][offset:end]) and torch.equal(result.deployed, outputs[group][1][offset:end]),
                        "Reference output replay changed")
                torch.autograd.backward((result.raw, result.deployed), tuple(v[offset:end] for v in derivatives[group]))
                require(physical.grad is not None and bool(torch.isfinite(physical.grad).all())
                        and torch.count_nonzero(physical.grad[..., :warmup_samples]) == 0
                        and torch.count_nonzero(physical.grad[..., warmup_samples:]) > 0, "Reference warmup gradients differ")
                del result, physical
                if progress is not None:
                    progress("independent_whole_group_vjp", group, offset)
            if group == "ordinary":
                ordinary_reference.update({name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()})
        expected = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()}
        del outputs, derivatives, inputs, groups
        model.zero_grad(set_to_none=True)

        def verify_ordinary(group, row):
            if group == "ordinary":
                require(len(ordinary_reference) == 40 and all(
                    torch.equal(p.grad.detach().cpu(), ordinary_reference[name])
                    for name, p in model.named_parameters()), "Ordinary gradients changed")
                ordinary_reference.clear()

        actual_groups = accumulate_groups(model, mixture_cpu, targets_cpu, warmup_samples=warmup_samples,
                                         verify_input_gradients=True, progress=progress, after_group=verify_ordinary)
        errors = {}
        for name, parameter in model.named_parameters():
            require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), "Missing finite canonical gradient: " + name)
            actual, reference = parameter.grad.detach().cpu(), expected[name]
            require(float(actual.norm()) > 0 and float(reference.norm()) > 0, "Unexercised canonical gradient: " + name)
            relative_l2 = float((actual - reference).double().norm() / reference.double().norm())
            errors[name] = {"maximum_absolute_error": float((actual - reference).abs().max()),
                           "relative_l2_error": relative_l2, "reference_norm": float(reference.norm()),
                           "bitwise_equal": torch.equal(actual, reference)}
            require(torch.allclose(actual, reference, atol=1e-7, rtol=1e-4) and relative_l2 < 5e-5,
                    "Canonical parameter gradient differs from whole-group VJP: " + name)
        require(len(errors) == 40 and state_sha256(model.state_dict()) == fingerprint, "Canonical parent or parameter inventory changed")
        for group in actual_groups:
            require(abs(actual_groups[group]["weighted_loss"] - losses[group]) < 3e-6
                    and actual_groups[group]["replay_outputs_bit_exact"], "Canonical whole-group loss differs")
        return {"status": "pass", "model_state_sha256": fingerprint, "device": str(device),
            "precision": model.training_precision, "ordinary_microbatch": 4, "auxiliary_microbatch": 2,
            "warmup_samples": warmup_samples, "scored_samples": mixture_cpu.shape[-1] - warmup_samples,
            "all_40_gradients": errors, "reference_losses": losses,
            "actual_losses": {g: row["weighted_loss"] for g, row in actual_groups.items()},
            "activity": {g: {"active": row["active_windows"], "absent": row["absent_windows"]} for g, row in actual_groups.items()},
            "absolute_gradient_tolerance": 1e-7, "relative_gradient_tolerance": 1e-4,
            "relative_l2_tolerance": 5e-5, "loss_absolute_tolerance": 3e-6,
            "warmup_input_gradients_zero": True, "weights_unchanged": True, "optimizer_updates": 0,
            "ordinary_all_40_gradients_bit_exact_against_unmodified_reference": True,
            "accumulation_policy": accumulation_policy(), "canonical_replay_outputs_bit_exact": True}
    finally:
        model.zero_grad(set_to_none=True)


def check_restart(parent, plan, *, progress=None):
    """Short device-local update/restart fixture on a private model copy."""
    from research.direct.check_latency58_weighted_vocal_cpu import exercise
    from research.direct.check_latency58_grouped_vocal_restart import fingerprint
    from research.direct.latency58_branch_memory_checkpoint import audit_live
    from research.direct.latency58_branch_ema import BranchParameterEMA
    from research.direct.latency58_branch_ema_checkpoint import make_payloads, audit_payloads
    from research.direct.latency58_weighted_vocal_auxiliary import VERSION
    device = next(parent.parameters()).device
    parent_sha = state_sha256(parent.state_dict())
    model = copy.deepcopy(parent).train().requires_grad_(True)
    model.provenance = {**model.provenance, "branch_memory_previous_provenance": copy.deepcopy(model.provenance),
        "branch_memory_parent_model_state_sha256": parent_sha,
        "branch_memory_current_stage_corpus": "Synthetic device restart fixture; no recorded examples",
        "branch_memory_current_stage_augmentation": "Fixed synthetic batch and two derived source views",
        "branch_memory_current_stage_training_context": {"warmup_samples": 512, "scored_samples": 44160}}
    frozen = {name: value.clone() for name, value in model.named_buffers()}
    optimizer = torch.optim.Adam(model.parameters(), lr=6e-5, foreach=False)
    ema = BranchParameterEMA(model, decay=plan["ema"]["decay"], base_state_sha256=parent_sha)
    generator = torch.Generator().manual_seed(202610309)
    truth = .02 * torch.randn(16, 4, 2, 512 + 44160, generator=generator)
    truth[:3, 2] = 0; truth[4:6, 1] = 0; truth[8:9, 3] = 0
    inputs = truth.sum(1), truth
    initial = fingerprint(model, optimizer, ema)
    from research.direct.latency58_weighted_vocal_canonical import grouped_update
    try:
        grouped_update(model, optimizer, ema, *inputs, step=2, warmup_samples=512)
    except RuntimeError as error:
        require(str(error) == "Grouped update must follow the contiguous EMA endpoint", "Unexpected invalid-step rejection")
    else:
        raise RuntimeError("Noncontiguous canonical update accepted")
    require(fingerprint(model, optimizer, ema) == initial and all(p.grad is None for p in model.parameters()),
            "Noncontiguous update changed its endpoint")
    interrupted = [exercise(model, optimizer, ema, inputs, step=1, stop_after=group)
                   for group in ("ordinary", "auxiliary")]
    for step in (1, 2):
        exercise(model, optimizer, ema, inputs, step=step)
        audit_live(model, optimizer, step, frozen)
        if progress is not None:
            progress("restart_fixture_update", "both", step)
    fixture = {"schema": "latency58-grouped-device-restart-fixture-v1",
        "config": {"steps": 2, "batch_size": 16, "data_start": 0},
        "parent_checkpoint": plan["parent_checkpoint"], "parent_model_state_sha256": parent_sha,
        "parent_training_updates": plan["parent_training_updates"], "ema": plan["ema"],
        "precision_policy": model.training_precision, "objective_version": VERSION,
        "fixed_buffers_sha256": state_sha256(frozen), "warmup_samples": 512, "scored_samples": 44160}
    fixture_sha = hashlib.sha256(json.dumps(fixture, sort_keys=True, allow_nan=False).encode()).hexdigest()
    values = make_payloads(model, optimizer, ema, 2, fixture, fixture_sha)
    with io.BytesIO() as stream:
        torch.save(values, stream)
        data = stream.getbuffer()
        serialized = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        del data
        stream.seek(0)
        raw, resume, average, metadata = torch.load(stream, map_location="cpu", weights_only=True)
    recovered, resume, averaged_model, recovered_ema = audit_payloads(raw, resume, average, metadata, fixture, fixture_sha)
    recovered_ema_payload = recovered_ema.state_dict(recovered)
    recovered.to(device).train().requires_grad_(True)
    recovered.training_precision = model.training_precision
    # Reconstruct EMA on the final training device, then load Adam against the raw parameters.
    recovered_ema = BranchParameterEMA.from_state_dict(recovered, recovered_ema_payload, expected_step=2,
        decay=ema.decay, base_state_sha256=parent_sha)
    restarted = torch.optim.Adam(recovered.parameters(), lr=6e-5, foreach=False)
    restarted.load_state_dict(resume["optimizer"])
    require(fingerprint(model, optimizer, ema) == fingerprint(recovered, restarted, recovered_ema),
            "Serialized device endpoint differs")
    del values, raw, resume, average, metadata, averaged_model, recovered_ema_payload
    gc.collect()
    continuous = exercise(model, optimizer, ema, inputs, step=3)
    replay = exercise(recovered, restarted, recovered_ema, inputs, step=3)
    require(continuous == replay and fingerprint(model, optimizer, ema) == fingerprint(recovered, restarted, recovered_ema),
            "Device restart changed the next grouped update")
    audit_live(model, optimizer, 3, frozen); audit_live(recovered, restarted, 3, frozen)
    require(state_sha256(parent.state_dict()) == parent_sha, "Restart fixture changed its training parent")
    return {"status": "pass", "parent_model_state_sha256": parent_sha, "device": str(device),
        "precision": model.training_precision, "warmup_samples": 512, "scored_samples": 44160,
        "serialized_in_memory": serialized, "all_40_adam_states_checked": True,
        "third_update_raw_adam_ema_and_accounting_bit_exact": True, "parent_weights_unchanged": True,
        "checkpoint_files_written": False, "quality_measured": False,
        "interrupted_accumulations": interrupted, "noncontiguous_step_rejected_before_gradients": True,
        "accumulation_policy": accumulation_policy()}


def check_gpu(model, plan, *, progress=None):
    from research.direct.check_latency58_branch_long_context_gpu_b4 import check_gpu as original_check, compare_context
    from research.direct.latency58_long_context_data import CROP_SAMPLES, WARMUP_SAMPLES
    parameter = next(model.parameters())
    require(parameter.is_cuda and model.training and model.training_precision == "bf16"
            and all(p.grad is None for p in model.parameters()), "Require the selected BF16 parent before updates")
    original = state_sha256(model.state_dict())
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    with torch.random.fork_rng(devices=[parameter.device.index]):
        ordinary = original_check(model)
        generator = torch.Generator().manual_seed(202610310)
        truth = .02 * torch.randn(16, 4, 2, CROP_SAMPLES, generator=generator)
        truth[:3, 2] = 0; truth[4:6, 1] = 0; truth[8:9, 3] = 0
        auxiliary_mix, auxiliary_truth = source_views(truth.sum(1), truth)
        auxiliary_context = compare_context(model, auxiliary_mix.to(parameter.device), WARMUP_SAMPLES)
        del auxiliary_mix, auxiliary_truth
        gradients = compare_group_gradients(model, truth.sum(1), truth, warmup_samples=WARMUP_SAMPLES, progress=progress)
        del truth
        gc.collect()
        restart = check_restart(model, plan, progress=progress)
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.synchronize()
    require(state_sha256(model.state_dict()) == original and torch.equal(cpu_rng, torch.get_rng_state())
            and all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)),
            "Grouped resource checks changed the parent or RNG")
    return {"status": "pass", "original_model_state_sha256": original,
        "ordinary_context": ordinary, "auxiliary_context": auxiliary_context,
        "grouped_parameter_gradients": gradients, "grouped_restart": restart,
        "trained_parent_weights_unchanged": True, "cpu_and_cuda_rng_restored": True,
        "training_optimizer_updates": 0, "peak_vram_gib_including_check": torch.cuda.max_memory_allocated() / 2**30}
