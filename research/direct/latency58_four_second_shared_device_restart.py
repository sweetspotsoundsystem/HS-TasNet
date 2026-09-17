"""Selected-geometry weighted CUDA restart with exact saved-weight sharing."""
import copy
import gc
import hashlib
import io
import json
import torch
from research.direct.run_latency58_quality import require
from research.direct.train_latency58 import state_sha256
from research.direct.latency58_weighted_vocal_canonical import policy as accumulation_policy
from research.direct.latency58_bf16_saved_gru_weights import grouped_update

def check_restart(parent, plan, *, ordinary_microbatch, auxiliary_microbatch, progress=None):
    """Short device-local update/restart fixture on a private model copy."""
    from functools import partial
    from research.direct.latency58_four_second_geometry_restart import exercise as geometry_exercise
    exercise = partial(geometry_exercise, ordinary_microbatch=ordinary_microbatch,
        auxiliary_microbatch=auxiliary_microbatch, update_impl=grouped_update)
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
    try:
        grouped_update(model, optimizer, ema, *inputs, step=2, warmup_samples=512, ordinary_microbatch=ordinary_microbatch, auxiliary_microbatch=auxiliary_microbatch)
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
        "ordinary_microbatch": ordinary_microbatch, "auxiliary_microbatch": auxiliary_microbatch,
        "interrupted_accumulations": interrupted, "noncontiguous_step_rejected_before_gradients": True,
        "accumulation_policy": accumulation_policy()}
