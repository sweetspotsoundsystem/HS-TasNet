"""Check longer context and a fresh-Adam raw/EMA restart from the selected EMA."""
import copy
import hashlib
import io

import torch

from research.direct.run_latency58_quality import require, sha, write
from research.direct.train_latency58 import state_sha256
from research.direct.latency58_branch_memory_checkpoint import audit_live, audit_resume
from research.direct.latency58_branch_ema_checkpoint import make_payloads, audit_payloads, policy
from research.direct.latency58_branch_ema import BranchParameterEMA
from research.direct.train_latency58_branch_sdr_ema import advance_with_ema
from research.direct.check_latency58_branch_memory import compare_context
from research.direct.latency58_long_context_data import WARMUP_SAMPLES, SCORED_SAMPLES, CROP_SAMPLES


def check(parent, source, out):
    parent_sha = state_sha256(parent.state_dict())
    require(parent.provenance.get("checkpoint_weight_role") == "averaged_inference"
            and source["parent_checkpoint"]["sha256"] == sha(source["parent_checkpoint"]["path"])
            and parent_sha == source["parent_model_state_sha256"]
            and source["optimizer_initialization"] == "fresh_adam", "Require the selected EMA and a fresh optimizer")
    model = copy.deepcopy(parent).train().requires_grad_(True)
    model.training_precision = "fp32"
    model.provenance = {**model.provenance, "branch_memory_previous_provenance": copy.deepcopy(model.provenance),
                        "branch_memory_parent_model_state_sha256": parent_sha}
    generator = torch.Generator().manual_seed(202610293)
    context = compare_context(model, .03 * torch.randn(1, 2, CROP_SAMPLES, generator=generator), WARMUP_SAMPLES)
    frozen = {name: value.clone() for name, value in model.named_buffers()}
    optimizer = torch.optim.Adam(model.parameters(), lr=6e-5, foreach=False)
    require(len(optimizer.state) == 0, "Fresh EMA-parent optimizer inherited Adam moments")
    ema = BranchParameterEMA(model, decay=.995, base_state_sha256=parent_sha)
    audio = .03 * torch.randn(1, 2, 8 * 128, generator=generator)
    target = .02 * torch.randn(1, 4, 2, 8 * 128, generator=generator)

    def update(candidate, adam, average, step):
        adam.zero_grad(set_to_none=True)
        output = candidate.render(audio)
        loss = (output.deployed - target).square().mean() + .25 * (output.raw - target).square().mean()
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.norm()) > 0
                    for p in candidate.parameters()), "Fresh-optimizer fixture missed a parameter gradient")
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), 5., foreach=False, error_if_nonfinite=True)
        advance_with_ema(candidate, adam, average, step=step)
        audit_live(candidate, adam, step, frozen)

    audit_live(model, optimizer, 0, frozen)
    for step in (1, 2):
        update(model, optimizer, ema, step)
    fixture = {**source, "config": {**source["config"], "steps": 2}, "ema": policy(.995),
               "fixed_buffers_sha256": state_sha256(frozen), "fixture_loss": "short synthetic MSE for restart only"}
    write(out / "fixture-plan.json", fixture)
    fixture_sha = sha(out / "fixture-plan.json")
    raw, resume, averaged, metadata = make_payloads(model, optimizer, ema, 2, fixture, fixture_sha)
    serialized, loaded = {}, {}
    for name, value in (("raw", raw), ("optimizer", resume), ("averaged", averaged)):
        with io.BytesIO() as stream:
            torch.save(value, stream)
            data = stream.getbuffer()
            serialized[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            del data
            stream.seek(0)
            loaded[name] = torch.load(stream, map_location="cpu", weights_only=True)
    restored, recovered, averaged_model, restored_ema = audit_payloads(
        loaded["raw"], loaded["optimizer"], loaded["averaged"], metadata, fixture, fixture_sha)
    require("weight_averaging" not in loaded["raw"]["provenance"]
            and loaded["raw"]["provenance"]["initial_parent_weight_averaging"] == parent.provenance["weight_averaging"]
            and loaded["averaged"]["provenance"]["weight_averaging"]["base_state_sha256"] == parent_sha,
            "Averaged-parent history was confused with current raw or averaged weights")
    with torch.inference_mode():
        live_average = ema.inference_copy(model)
        a, b = live_average.render(audio), averaged_model.render(audio)
        require(all(torch.equal(getattr(a, k).view(torch.int32), getattr(b, k).view(torch.int32))
                    for k in ("raw", "deployed", "native_raw", "spectral", "waveform", "delayed_mixture"))
                and all(torch.equal(x.view(torch.int32), y.view(torch.int32))
                        for x, y in zip(a.state, b.state, strict=True)), "Serialized EMA replay differs")
        del live_average
    try:
        audit_resume(averaged_model, loaded["averaged"], loaded["optimizer"], fixture, fixture_sha)
    except RuntimeError as error:
        require(str(error) == "Saved optimizer belongs to another endpoint", "Unexpected EMA/Adam rejection")
    else:
        raise RuntimeError("Raw Adam moments were accepted as belonging to averaged weights")
    restored.train().requires_grad_(True)
    restored.training_precision = "fp32"
    restarted = torch.optim.Adam(restored.parameters(), lr=6e-5, foreach=False)
    restarted.load_state_dict(recovered["optimizer"])
    audit_live(restored, restarted, 2, frozen)
    update(model, optimizer, ema, 3)
    update(restored, restarted, restored_ema, 3)
    require(state_sha256(model.state_dict()) == state_sha256(restored.state_dict())
            and state_sha256(ema.state_dict(model)["parameters"]) ==
                state_sha256(restored_ema.state_dict(restored)["parameters"]), "Resumed raw or EMA update differs")
    for first, second in zip(model.parameters(), restored.parameters(), strict=True):
        require(all(torch.equal(optimizer.state[first][k], restarted.state[second][k])
                    for k in ("step", "exp_avg", "exp_avg_sq")), "Resumed raw Adam state differs")
    require(state_sha256(parent.state_dict()) == parent_sha and not torch.cuda.is_initialized(),
            "CPU parent qualification changed its parent or initialized CUDA")
    return {"status": "pass", "parent_model_state_sha256": parent_sha, "context": context,
            "warmup_samples": WARMUP_SAMPLES, "scored_samples": SCORED_SAMPLES,
            "fresh_adam_initial_state_empty": True, "all_40_optimizer_states_checked": True,
            "previous_ema_history_preserved_in_provenance": True, "raw_adam_rejected_for_ema_weights": True,
            "serialized_ema_outputs_and_eight_states_bit_exact": True,
            "resumed_third_raw_and_ema_update_and_adam_states_bit_exact": True,
            "serialized_in_memory": serialized, "parent_unchanged": True, "gpu_used": False,
            "checkpoint_files_written": False, "quality_measured": False,
            "scope": "B1 FP32 full two-second scored-context outputs and all 40 gradients; short synthetic restart fixture. GPU B8 context and recorded B16 accumulated updates require the resource rehearsal."}
