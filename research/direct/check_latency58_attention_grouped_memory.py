"""Check two-group attention Adam serialization, endpoint rates and exact restart in RAM."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs


def check(source, evidence):
    import numpy as np
    import random
    import torch
    from research.direct.latency58_temporal_attention import Latency58TemporalAttentionModel
    from research.direct.latency58_attention_grouped_checkpoint import (
        make_payloads, load_payload, audit_resume, audit_live, make_optimizer, set_learning_rate,
        scheduled_backbone_lr, OPTIMIZER_SCHEMA, SCHEDULE_SCHEMA,
    )
    from research.direct.latency58_temporal_attention_checkpoint import load_model as load_parent
    parent, _ = load_parent(source["parent_checkpoint"])
    parent_sha = state_sha256(parent.state_dict())
    model = copy.deepcopy(parent).train().requires_grad_(True)
    model.provenance = {**model.provenance,
                        "temporal_attention_previous_provenance": dict(model.provenance),
                        "temporal_attention_parent_model_state_sha256": parent_sha}
    model.training_precision = "fp32"
    previous_context = PHASE / "attention-continuation-memory-functional-001/result.json"
    context = read(previous_context)
    require(context["status"] == "pass" and context["source_bindings_unchanged"]
            and context["trained_parent_context"]["status"] == "pass"
            and len(context["trained_parent_context"]["all_30_gradients"]) == 30,
            "Require the existing unchanged-architecture context proof")
    frozen = {name: value.clone() for name, value in model.named_buffers()}
    plan = {**source, "schema": "latency58-attention-grouped-checkpoint-fixture-plan-v1",
            "config": {**source["config"], "steps": 2, "lr": 1e-5, "min_lr": 1e-6, "warmup": 0},
            "fixed_buffers_sha256": state_sha256(frozen), "parent_model_state_sha256": parent_sha,
            "parent_training_updates": parent.provenance["training_updates"],
            "optimizer_schema": OPTIMIZER_SCHEMA, "optimizer_schedule": SCHEDULE_SCHEMA,
            "attention_lr_multiplier": 100., "objective_version": "synthetic-eight-hop-deployed-and-raw-mse-fixture-v1",
            "precision_policy": "CPU FP32 fixture; no production recipe selected"}
    write(evidence / "fixture-plan.json", plan)
    plan_sha = sha(evidence / "fixture-plan.json")
    optimizer = make_optimizer(model, lr=plan["config"]["lr"], attention_lr_multiplier=100.)
    observed_rates = []
    generator = torch.Generator().manual_seed(202609145)
    audio = .03 * torch.randn(1, 2, 8 * 128, generator=generator)
    target = .02 * torch.randn(1, 4, 2, 8 * 128, generator=generator)

    def update(candidate, adam, step):
        if step <= 2:
            set_learning_rate(adam, scheduled_backbone_lr(plan["config"], step - 1))
        # The third fixture update deliberately uses the saved endpoint rates.
        # It exercises resume mechanics after the declared two-update endpoint.
        observed_rates.append({"step": step, "rates": [group["lr"] for group in adam.param_groups]})
        adam.zero_grad(set_to_none=True)
        output = candidate.render(audio)
        loss = (output.deployed - target).square().mean() + .25 * (output.raw - target).square().mean()
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in candidate.parameters()),
                "A fixture parameter lacks a finite gradient")
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), 5., error_if_nonfinite=True, foreach=False)
        adam.step()
        audit_live(candidate, adam, step, frozen, attention_lr_multiplier=100.)

    audit_live(model, optimizer, 0, frozen, attention_lr_multiplier=100.)
    for step in (1, 2):
        update(model, optimizer, step)
    payload, resume = make_payloads(model, optimizer, 2, plan, plan_sha)
    serialized, loaded = {}, {}
    for name, value in (("model", payload), ("optimizer", resume)):
        with io.BytesIO() as stream:
            torch.save(value, stream)
            data = stream.getbuffer()
            serialized[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "saved_to_file": False}
            del data
            stream.seek(0)
            loaded[name] = torch.load(stream, map_location="cpu", weights_only=True)
    restored, recovered = load_payload(loaded["model"])
    audit_resume(restored, recovered, loaded["optimizer"], plan, plan_sha)
    from research.direct.latency58_temporal_attention_checkpoint import load_payload as original_inference_loader
    compatible, _ = original_inference_loader(loaded["model"])
    require(recovered["model_state_sha256"] == state_sha256(model.state_dict()), "Model serialization changed tensors")
    with torch.inference_mode():
        first, second, third = model.eval().render(audio), restored.render(audio), compatible.render(audio)
        require(all(torch.equal(getattr(first, key).view(torch.int32), getattr(second, key).view(torch.int32))
                    for key in ("raw", "deployed", "native_raw", "spectral", "waveform", "delayed_mixture"))
                and all(torch.equal(a.view(torch.int32), b.view(torch.int32))
                        for a, b in zip(first.state, second.state, strict=True)), "Serialized inference replay differs")
        require(torch.equal(second.deployed, third.deployed)
                and all(torch.equal(a, b) for a, b in zip(second.state, third.state, strict=True)),
                "The original inference loader changed outputs or states")
    del compatible
    rejected = []
    invalid = {**recovered, "model": {**recovered["model"],
               "temporal_output.weight": recovered["model"]["temporal_output.weight"].double()}}
    wrong_step = {**loaded["optimizer"], "step": 3}
    def with_groups(groups):
        return {**loaded["optimizer"], "optimizer": {**loaded["optimizer"]["optimizer"], "param_groups": groups}}
    groups = loaded["optimizer"]["optimizer"]["param_groups"]
    wrong_membership = with_groups([{**groups[0], "params": [1, *groups[0]["params"][1:]]}, groups[1]])
    wrong_ratio = with_groups([groups[0], {**groups[1], "lr": groups[1]["lr"] * 2}])
    wrong_absolute_rates = with_groups([{**group, "lr": group["lr"] * 2} for group in groups])
    wrong_betas = with_groups([{**groups[0], "betas": (.8, .999)}, groups[1]])
    wrong_group_name = with_groups([groups[0], {**groups[1], "group_name": "backbone"}])
    wrong_group_count = with_groups([{**groups[0], "params": list(range(30))}])
    for name, operation, message in (
        ("wrong_tensor_dtype", lambda: load_payload(invalid), "Invalid temporal_attention inference tensors"),
        ("wrong_optimizer_step", lambda: audit_resume(restored, recovered, wrong_step, plan, plan_sha),
         "Saved optimizer belongs to another endpoint"),
        ("wrong_training_plan", lambda: audit_resume(restored, recovered, loaded["optimizer"], plan, "0" * 64),
         "Saved optimizer belongs to another endpoint"),
        ("wrong_group_membership", lambda: audit_resume(restored, recovered, wrong_membership, plan, plan_sha),
         "Adam group membership, rate or hyperparameters changed"),
        ("wrong_attention_rate_ratio", lambda: audit_resume(restored, recovered, wrong_ratio, plan, plan_sha),
         "Attention-to-backbone learning-rate ratio changed"),
        ("wrong_absolute_endpoint_rates", lambda: audit_resume(restored, recovered, wrong_absolute_rates, plan, plan_sha),
         "Saved backbone rate differs from the declared schedule endpoint"),
        ("wrong_adam_betas", lambda: audit_resume(restored, recovered, wrong_betas, plan, plan_sha),
         "Adam group membership, rate or hyperparameters changed"),
        ("wrong_group_name", lambda: audit_resume(restored, recovered, wrong_group_name, plan, plan_sha),
         "Adam group membership, rate or hyperparameters changed"),
        ("wrong_group_count", lambda: audit_resume(restored, recovered, wrong_group_count, plan, plan_sha),
         "Require two ordered Adam groups"),
    ):
        try:
            operation()
        except RuntimeError as error:
            require(str(error) == message, "Unexpected payload rejection reason")
            rejected.append(name)
        else:
            raise RuntimeError("Invalid serialized payload was accepted")
    restored.train().requires_grad_(True)
    restored.training_precision = "fp32"
    restarted = make_optimizer(restored, lr=7e-5, attention_lr_multiplier=100.)
    require([group["lr"] for group in restarted.param_groups] != [group["lr"] for group in optimizer.param_groups],
            "Start the reload check with deliberately different constructor rates")
    restarted.load_state_dict(loaded["optimizer"]["optimizer"])
    audit_live(restored, restarted, 2, frozen, attention_lr_multiplier=100.)
    require([{k: v for k, v in group.items() if k != "params"} for group in restarted.param_groups]
            == [{k: v for k, v in group.items() if k != "params"} for group in optimizer.param_groups],
            "Reload changed group rates or hyperparameters")
    model.train()
    update(model, optimizer, 3)
    update(restored, restarted, 3)
    require(state_sha256(model.state_dict()) == state_sha256(restored.state_dict()), "Resumed next update differs")
    for original, replay in zip(model.parameters(), restored.parameters(), strict=True):
        require(all(torch.equal(optimizer.state[original][key], restarted.state[replay][key])
                    for key in ("step", "exp_avg", "exp_avg_sq")), "Resumed Adam moments differ")
    # Exercise the stored CPU RNG formats in private generators only.
    rng = loaded["optimizer"]
    original_rng = torch.Generator().set_state(resume["torch_rng"])
    restored_rng = torch.Generator().set_state(rng["torch_rng"])
    require(torch.equal(torch.rand(8, generator=original_rng), torch.rand(8, generator=restored_rng)), "Torch RNG changed")
    python_rngs = [random.Random(), random.Random()]
    numpy_rngs = [np.random.RandomState(), np.random.RandomState()]
    for record, prng, nrng in zip((resume, rng), python_rngs, numpy_rngs, strict=True):
        prng.setstate(record["python_rng"])
        state = record["numpy_rng"]
        nrng.set_state((state[0], state[1].numpy().astype(np.uint32), *state[2:]))
    require(python_rngs[0].random() == python_rngs[1].random()
            and np.array_equal(numpy_rngs[0].random_sample(8), numpy_rngs[1].random_sample(8)), "CPU RNG replay changed")
    require(state_sha256(parent.state_dict()) == parent_sha and not torch.cuda.is_initialized(), "Parent or CPU scope changed")
    return {"status": "pass", "unchanged_architecture_context_proof_reused": {
                "path": str(previous_context), "sha256": sha(previous_context)},
            "serialized_in_memory": serialized, "all_30_optimizer_states_checked": True,
            "optimizer_group_sizes": [26, 4], "fixture_attention_lr_multiplier": 100.,
            "observed_update_rates": observed_rates, "constructor_rates_replaced_by_saved_rates": True,
            "third_fixture_update_uses_saved_endpoint_rates": True,
            "inference_outputs_and_states_bit_exact": True, "resumed_third_update_and_adam_moments_bit_exact": True,
            "original_inference_loader_outputs_and_states_bit_exact": True,
            "cpu_rng_formats_replay_exact": True, "rejected_invalid_cases": rejected,
            "parent_unchanged": True, "gpu_used": False, "checkpoint_files_written": False,
            "disk_generation_path_exercised": False, "quality_measured": False,
            "production_recipe_selected": False,
            "limitation": "Uses the production payload builder and loader in RAM. Atomic disk publication and GPU RNG restoration are not exercised by this CPU fixture."}



def main():
    import torch
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use the CPU1 workspace")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    parent_root = PHASE / "temporal-attention-001"
    source_path = parent_root / "plan.json"
    source = read(source_path)
    review_path = parent_root / "selection-review.json"
    review = read(review_path)
    terminal = read(parent_root / "result.json")
    require(review["status"] == "selected_for_research" and review["actual_root_exit_code"] == 0
            and terminal["status"] == "training_audit_and_full14_complete"
            and terminal["checkpoint"] == review["best_research_checkpoint"]
            and terminal["full_sdr_db"] == 4.288099064999147 < 5.0,
            "Use the selected completed attention parent")
    verify_inputs(source)
    verify_inputs(review)
    paths = [Path(__file__).resolve(), source_path, review_path]
    paths.extend(parent_root / name for name in (
        "result.json", "production-stage/execution.json", "full14/execution.json", "full14/result.json",
        "production-run/checkpoint/model.pt", "production-run/checkpoint/optimizer.pt"))
    paths.extend(ROOT / "research/direct" / name for name in (
        "check_latency58_temporal_attention.py", "latency58_temporal_attention.py",
        "latency58_temporal_attention_context.py", "latency58_temporal_attention_checkpoint.py",
        "latency58_attention_grouped_checkpoint.py"))
    paths.extend(PHASE / name for name in (
        "attention-continuation-memory-functional-001/plan.json",
        "attention-continuation-memory-functional-001/result.json",
        "attention-continuation-memory-functional-stage-001/execution.json",
        "attention-parent-ablation-002/analysis.json"))
    memory_plan = read(PHASE / "attention-continuation-memory-functional-001/plan.json")
    verify_inputs(memory_plan)
    require(memory_plan["parent_checkpoint"] == terminal["checkpoint"], "Reused context proof belongs to another parent")
    memory_execution = read(PHASE / "attention-continuation-memory-functional-stage-001/execution.json")
    require(memory_execution["actual_exit_code"] == 0 and memory_execution["source_bindings_unchanged"]
            and not memory_execution["timed_out"], "Reused context proof did not close successfully")
    for name in ("production-stage/execution.json", "full14/execution.json"):
        execution = read(parent_root / name)
        require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"],
                "Parent training/scoring did not close")
    bindings = {**source["source_bindings"], **{str(path): sha(path) for path in paths}}
    verify_inputs({"source_bindings": bindings})
    source = {**source, "parent_checkpoint": terminal["checkpoint"],
              "parent_model_state_sha256": "d976397247dd5ad79d09b14ab62ea96a4acaeb0554e256d4f290bd7edf4c8127",
              "parent_training_updates": 21250, "parent_kind": "saved_temporal_attention",
              "optimizer_initialization": "fresh_adam"}
    require_space(source, 385_000_000)
    out = PHASE / "attention-grouped-memory-functional-001"
    require(not out.exists(), "Preserve earlier checkpoint checks")
    out.mkdir()
    write(out / "plan.json", {"source_bindings": bindings, "parent_checkpoint": terminal["checkpoint"],
          "scope": "Two-group Adam rates, in-memory serialization and exact next update at the saved endpoint rates",
          "checkpoint_files_written": False, "production_recipe_selected": False})
    began = time.monotonic()
    result = check(source, out)
    verify_inputs({"source_bindings": bindings})
    result.update(source_bindings_unchanged=True, elapsed_seconds=time.monotonic() - began,
                  plan_sha256=sha(out / "plan.json"), counted_bytes_after=require_space(source, 385_000_000))
    write(out / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
