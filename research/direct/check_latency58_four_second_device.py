"""Qualify the selected four-second B2 context, gradients and device restart."""
import copy
import gc
import hashlib
import io
import json

import torch

from research.direct.run_latency58_quality import require
from research.direct.train_latency58 import state_sha256
from research.direct.latency58_four_second_data import WARMUP_SAMPLES, SCORED_SAMPLES, CROP_SAMPLES
from research.direct.latency58_branch_sdr_blend import objective as full_objective
from research.direct.latency58_logical_batch_loss import objective, prepare_reduction
from research.direct.latency58_grouped_vocal_auxiliary import source_views
from research.direct.latency58_weighted_vocal_canonical import policy as accumulation_policy
from research.direct.check_latency58_four_second_model import compare_group_gradients
from research.direct.check_latency58_branch_long_context_gpu_b4 import compare_context

def check_loss(device, generator):
    target = .03 * torch.randn(16, 4, 2, SCORED_SAMPLES, device=device, generator=generator)
    target[:8, 1] = 0
    target[8:, 2] = 0
    target[0, 3, :, :44100] = .0001
    mixture = target.sum(1)
    noise = .008 * torch.randn(target.shape, device=device, generator=generator)
    noise[8:] *= 3
    raw, deployed = (target + noise).requires_grad_(), (target + .7 * noise).requires_grad_()
    full = full_objective(raw, deployed, target, mixture)
    expected = torch.autograd.grad(full.total, (raw, deployed))
    reduction = prepare_reduction(target)
    pieces = [objective(raw[i:i + 2], deployed[i:i + 2], target[i:i + 2], mixture[i:i + 2], reduction)
              for i in range(0, 16, 2)]
    combined = sum(p.total for p in pieces)
    actual = torch.autograd.grad(combined, (raw, deployed))
    errors = [float((a - b).abs().max()) for a, b in zip(actual, expected, strict=True)]
    require(torch.allclose(combined, full.total, atol=3e-6, rtol=3e-6)
            and all(torch.allclose(a, b, atol=2e-9, rtol=3e-5) for a, b in zip(actual, expected, strict=True))
            and torch.equal(sum(p.active_window_counts for p in pieces), reduction.active)
            and torch.equal(sum(p.absent_window_counts for p in pieces), reduction.absent),
            "CUDA microbatch reduction differs from the original full-batch objective")
    return {"status": "pass", "logical_batch_size": 16, "microbatch_size": 2,
            "scored_samples": SCORED_SAMPLES, "loss_absolute_error": float((combined - full.total).detach().abs()),
            "raw_and_deployed_gradient_max_abs_errors": errors,
            "active_windows": reduction.active.cpu().tolist(), "absent_windows": reduction.absent.cpu().tolist(),
            "scope": "Loss and gradients in audio-output coordinates; no full-B16 neural activation allocation"}

def check_restart(parent, plan, *, progress=None):
    """Short device-local update/restart fixture on a private model copy."""
    from research.direct.check_latency58_four_second_restart import exercise
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
        grouped_update(model, optimizer, ema, *inputs, step=2, warmup_samples=512, ordinary_microbatch=2, auxiliary_microbatch=2)
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
        "ordinary_microbatch": 2, "auxiliary_microbatch": 2,
        "interrupted_accumulations": interrupted, "noncontiguous_step_rejected_before_gradients": True,
        "accumulation_policy": accumulation_policy()}


def check_gpu(model, plan, *, progress=None):
    parameter = next(model.parameters())
    require(parameter.is_cuda and model.training and model.training_precision == 'bf16'
            and all(p.grad is None for p in model.parameters()), 'Require the selected untouched BF16 parent')
    original = state_sha256(model.state_dict())
    require(original == plan['initialized_model_state_sha256'], 'Four-second resource parent differs')
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    generator = torch.Generator(device=parameter.device).manual_seed(202611041)
    with torch.random.fork_rng(devices=[parameter.device.index]):
        audio = .03 * torch.randn(2, 2, CROP_SAMPLES, generator=generator, device=parameter.device)
        ordinary = compare_context(model, audio, WARMUP_SAMPLES)
        del audio
        ordinary.update(microbatch_size=2, logical_batch_loss=check_loss(parameter.device, generator))
        if progress is not None:
            progress('ordinary_context_complete', 'ordinary', 0)
        truth = .02 * torch.randn(16, 4, 2, CROP_SAMPLES,
                                 generator=torch.Generator().manual_seed(202611042))
        truth[:3, 2] = 0; truth[4:6, 1] = 0; truth[8:9, 3] = 0
        auxiliary_mix, auxiliary_truth = source_views(truth.sum(1), truth)
        auxiliary = compare_context(model, auxiliary_mix.to(parameter.device), WARMUP_SAMPLES)
        del auxiliary_mix, auxiliary_truth
        if progress is not None:
            progress('auxiliary_context_complete', 'auxiliary', 0)
        gradients = compare_group_gradients(model, truth.sum(1), truth,
                                            warmup_samples=WARMUP_SAMPLES, progress=progress)
        del truth
        gc.collect()
        restart = check_restart(model, plan, progress=progress)
    model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.synchronize()
    require(state_sha256(model.state_dict()) == original and torch.equal(cpu_rng, torch.get_rng_state())
            and all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)),
            'Four-second GPU qualification changed weights or RNG')
    return {'status': 'pass', 'ordinary_context': ordinary, 'auxiliary_context': auxiliary,
            'grouped_parameter_gradients': gradients, 'grouped_restart': restart,
            'trained_parent_weights_unchanged': True, 'cpu_and_cuda_rng_restored': True,
            'training_optimizer_updates': 0, 'original_model_state_sha256': original,
            'ordinary_microbatch': 2, 'auxiliary_microbatch': 2,
            'warmup_samples': WARMUP_SAMPLES, 'scored_samples': SCORED_SAMPLES,
            'inference_architecture_changed': False,
            'peak_vram_gib_including_check': torch.cuda.max_memory_allocated() / 2**30}
