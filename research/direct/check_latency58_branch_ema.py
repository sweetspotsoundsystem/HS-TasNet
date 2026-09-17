"""CPU-only EMA recurrence, checkpoint restart and unchanged-graph qualification."""
import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs


def check(plan, out):
    import torch
    from research.direct.latency58_branch_ema import BranchParameterEMA
    from research.direct.latency58_branch_memory_checkpoint import (
        audit_live, audit_resume, load_model, load_payload, make_payloads,
    )
    from research.direct.check_latency58_branch_memory import compare_context
    parent, _ = load_model(plan["parent_checkpoint"])
    parent_sha = state_sha256(parent.state_dict())
    require(parent_sha == plan["parent_model_state_sha256"], "EMA fixture parent changed")
    model = copy.deepcopy(parent).train().requires_grad_(True)
    model.training_precision = "fp32"
    model.provenance = {**model.provenance,
                        "branch_memory_previous_provenance": dict(model.provenance),
                        "branch_memory_parent_model_state_sha256": parent_sha}
    frozen = {name: b.clone() for name, b in model.named_buffers()}
    decay = plan["ema_decay"]
    ema = BranchParameterEMA(model, decay=decay, base_state_sha256=parent_sha)
    zero = BranchParameterEMA(model, decay=0., base_state_sha256=parent_sha)
    independent = {name: p.detach().double().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=6e-5, foreach=False)
    generator = torch.Generator().manual_seed(202610251)
    audio = .03 * torch.randn(1, 2, 8 * 128, generator=generator)
    target = .02 * torch.randn(1, 4, 2, 8 * 128, generator=generator)

    def compare_outputs(first, second):
        with torch.inference_mode():
            a, b = first.eval().render(audio), second.eval().render(audio)
        require(all(torch.equal(getattr(a, k).view(torch.int32), getattr(b, k).view(torch.int32))
                    for k in ("raw", "deployed", "native_raw", "spectral", "waveform", "delayed_mixture"))
                and len(a.state) == len(b.state) == 8
                and all(torch.equal(x.view(torch.int32), y.view(torch.int32))
                        for x, y in zip(a.state, b.state, strict=True)), "EMA inference replay differs")
        require(first.algorithmic_latency_samples == second.algorithmic_latency_samples == 256,
                "EMA inference delay changed")

    initial = ema.inference_copy(model)
    compare_outputs(parent, initial)
    require(state_sha256(initial.state_dict()) == parent_sha, "EMA initialization changed weights")
    del initial
    serialized = {}
    def roundtrip(name, value):
        with io.BytesIO() as stream:
            torch.save(value, stream)
            data = stream.getbuffer()
            serialized[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                                "saved_to_file": False}
            del data
            stream.seek(0)
            return torch.load(stream, map_location="cpu", weights_only=True)

    def update(candidate, adam, step):
        candidate.train()
        adam.zero_grad(set_to_none=True)
        output = candidate.render(audio)
        loss = (output.deployed - target).square().mean() + .25 * (output.raw - target).square().mean()
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in candidate.parameters()),
                "EMA fixture lacks finite parameter gradients")
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), 5., error_if_nonfinite=True, foreach=False)
        adam.step()
        audit_live(candidate, adam, step, frozen)

    errors = []
    for step in (1, 2):
        update(model, optimizer, step)
        ema.update(model, step=step)
        zero.update(model, step=step)
        for name, p in model.named_parameters():
            independent[name] = decay * independent[name] + (1 - decay) * p.detach().double()
            require(torch.equal(zero.parameters[name].view(torch.int32), p.detach().view(torch.int32)),
                    "Zero-decay EMA must equal raw weights exactly")
        error = max(float((ema.parameters[name].double() - value).abs().max())
                    for name, value in independent.items())
        errors.append(error)
        require(error < 2e-6, "EMA differs from independent FP64 recurrence")
    fixture = {**plan["fixture_training_plan"], "parent_checkpoint": plan["parent_checkpoint"],
               "parent_model_state_sha256": parent_sha, "parent_training_updates": parent.provenance["training_updates"],
               "fixed_buffers_sha256": state_sha256(frozen),
               "objective_version": "ema-cpu-functional-mse-fixture-v1", "precision_policy": "fp32",
               "config": {**plan["fixture_training_plan"]["config"], "steps": 2}}
    write(out / "fixture-plan.json", fixture)
    fixture_sha = sha(out / "fixture-plan.json")
    raw_payload, raw_resume = make_payloads(model, optimizer, 2, fixture, fixture_sha)
    loaded_raw = roundtrip("raw_model", raw_payload)
    loaded_optimizer = roundtrip("raw_optimizer", raw_resume)
    original_ema = ema.state_dict(model)
    loaded_ema = roundtrip("ema", original_ema)
    restored, recovered = load_payload(loaded_raw)
    audit_resume(restored, recovered, loaded_optimizer, fixture, fixture_sha)
    restored_ema = BranchParameterEMA.from_state_dict(restored, loaded_ema, expected_step=2,
                                                     decay=decay, base_state_sha256=parent_sha)
    compare_outputs(model, restored)
    averaged, restored_average = ema.inference_copy(model), restored_ema.inference_copy(restored)
    compare_outputs(averaged, restored_average)
    require(all(ema.parameters[name].data_ptr() != p.data_ptr() for name, p in model.named_parameters())
            and all(restored_ema.parameters[name].data_ptr() != loaded_ema["parameters"][name].data_ptr()
                    for name in restored_ema.parameters), "EMA tensors alias raw or serialized state")
    first_name = next(iter(ema.parameters))
    before = ema.parameters[first_name].clone()
    original_ema["parameters"][first_name].add_(1)
    with torch.no_grad():
        dict(averaged.named_parameters())[first_name].add_(1)
    require(torch.equal(before, ema.parameters[first_name])
            and state_sha256(model.state_dict()) == recovered["model_state_sha256"],
            "Modifying EMA exports changed the live EMA or raw model")
    del averaged, restored_average, raw_payload, raw_resume, loaded_raw, original_ema, zero
    rejected = []
    def reject(name, operation):
        try:
            operation()
        except RuntimeError as error:
            rejected.append({"case": name, "reason": str(error)})
        else:
            raise RuntimeError("Invalid EMA case accepted: " + name)

    def restore(payload, **kwargs):
        return BranchParameterEMA.from_state_dict(restored, payload, expected_step=2, decay=decay,
                                                  base_state_sha256=parent_sha, **kwargs)
    for name, changed in (
        ("wrong_schema", {"schema": "unknown"}), ("wrong_update", {"updates": 3}),
        ("wrong_decay", {"decay": .99}), ("wrong_base", {"base_state_sha256": "0" * 64}),
        ("wrong_raw_endpoint", {"raw_state_sha256": "0" * 64}),
        ("wrong_inventory", {"parameter_names": loaded_ema["parameter_names"][:-1]}),
        ("wrong_fingerprint", {"ema_parameters_sha256": "0" * 64}),
        ("wrong_dtype", {"parameters": {**loaded_ema["parameters"], first_name:
                                         loaded_ema["parameters"][first_name].double()}}),
        ("wrong_shape", {"parameters": {**loaded_ema["parameters"], first_name:
                                         loaded_ema["parameters"][first_name].flatten()[:1]}}),
    ):
        reject(name, lambda changed=changed: restore({**loaded_ema, **changed}))
    for decay_case in (float("nan"), 1., -.1, True):
        reject("invalid_decay_" + str(decay_case), lambda d=decay_case:
               BranchParameterEMA(restored, decay=d, base_state_sha256=state_sha256(restored.state_dict())))
    before_rejection = state_sha256(ema.parameters)
    reject("repeated_update", lambda: ema.update(model, step=2))
    reject("skipped_update", lambda: ema.update(model, step=4))
    buffer_name = next(iter(frozen))
    with torch.no_grad():
        dict(model.named_buffers())[buffer_name].add_(1)
    reject("modified_fixed_buffer", lambda: ema.update(model, step=3))
    with torch.no_grad():
        dict(model.named_buffers())[buffer_name].copy_(frozen[buffer_name])
    require(state_sha256(ema.parameters) == before_rejection and ema.updates == 2,
            "Rejected EMA update mutated averages")
    restored.train().requires_grad_(True)
    restored.training_precision = "fp32"
    restarted = torch.optim.Adam(restored.parameters(), lr=6e-5, foreach=False)
    restarted.load_state_dict(loaded_optimizer["optimizer"])
    update(model, optimizer, 3)
    reject("raw_advanced_without_ema", lambda: ema.state_dict(model))
    reject("actual_wrong_raw_model", lambda: BranchParameterEMA.from_state_dict(
        model, loaded_ema, expected_step=2, decay=decay, base_state_sha256=parent_sha))
    ema.update(model, step=3)
    update(restored, restarted, 3)
    restored_ema.update(restored, step=3)
    require(state_sha256(model.state_dict()) == state_sha256(restored.state_dict())
            and state_sha256(ema.parameters) == state_sha256(restored_ema.parameters),
            "Resumed raw or averaged third update differs")
    for p, q in zip(model.parameters(), restored.parameters(), strict=True):
        require(all(torch.equal(optimizer.state[p][key], restarted.state[q][key])
                    for key in ("step", "exp_avg", "exp_avg_sq")), "Resumed raw Adam moments differ")
    for name, p in model.named_parameters():
        independent[name] = decay * independent[name] + (1 - decay) * p.detach().double()
    errors.append(max(float((ema.parameters[name].double() - value).abs().max())
                      for name, value in independent.items()))
    require(max(errors) < 2e-6, "Resumed EMA differs from independent FP64 recurrence")
    averaged, restored_average = ema.inference_copy(model), restored_ema.inference_copy(restored)
    compare_outputs(averaged, restored_average)
    averaged.train().requires_grad_(True)
    averaged.training_precision = "fp32"
    context = compare_context(averaged, .03 * torch.randn(1, 2, 88064 + 8 * 128, generator=generator), 88064)
    with torch.inference_mode():
        output = averaged.eval().render(audio)
        closure = float((output.deployed.sum(1) - output.delayed_mixture).abs().max())
    require(closure < 1e-6 and state_sha256(parent.state_dict()) == parent_sha
            and not torch.cuda.is_initialized(), "EMA parent, closure or CPU scope changed")
    return {"status": "pass", "ema_decay": decay, "updates": 3, "parameter_tensor_count": 40,
            "initialized_parent_outputs_and_eight_states_bit_exact": True,
            "zero_decay_raw_parameters_bit_exact": True,
            "fp64_recurrence_max_abs_errors": errors, "fp64_recurrence_abs_tolerance": 2e-6,
            "serialized_in_memory": serialized, "raw_and_ema_checkpoint_inference_bit_exact": True,
            "resumed_raw_third_update_and_all_40_adam_states_bit_exact": True,
            "resumed_ema_third_update_outputs_and_eight_states_bit_exact": True,
            "raw_and_ema_storage_independent": True, "fixed_buffers_unchanged": True,
            "ema_context": context, "closure_max_abs": closure, "algorithmic_latency_samples": 256,
            "rejected_invalid_cases": rejected, "parent_unchanged": True, "gpu_used": False,
            "checkpoint_files_written": False, "disk_publication_exercised": False,
            "production_objective_exercised": False, "quality_measured": False,
            "limitation": "CPU synthetic MSE fixture only; no quality gain, GPU parity, disk publication, native timing or equivalence to the upstream EMA package schedule is claimed."}


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
            "Require a frozen plan and CUDA-hidden CPU1 fixture")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    plan = read(args.plan)
    require(plan["schema"] == "latency58-branch-ema-functional-plan-v1", "Wrong EMA fixture plan")
    out = Path(plan["output_directory"])
    require(out.resolve() == args.plan.parent.resolve() and out.resolve().is_relative_to(PHASE)
            and not (out / "result.json").exists(), "Preserve earlier EMA results")
    verify_inputs(plan)
    counted = require_space(plan, 450_000_000)
    started = time.monotonic()
    result = check(plan, out)
    verify_inputs(plan)
    require_space(plan, 450_000_000)
    result.update(plan_sha256=args.plan_sha256, source_bindings=plan["source_bindings"],
                  elapsed_seconds=time.monotonic() - started, counted_bytes_before=counted,
                  forecast_with_concurrent_training=counted + 450_000_000 + plan["outside_roots_reservation_bytes"],
                  fixture_plan_sha256=sha(out / "fixture-plan.json"))
    write(out / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("source_bindings", "ema_context")}), flush=True)


if __name__ == "__main__":
    main()
