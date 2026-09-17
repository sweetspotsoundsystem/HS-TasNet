"""Exercise the temporal-attention serializer and optimizer restart in RAM only."""
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
    from research.direct.latency58_temporal_attention_checkpoint import make_payloads, load_payload, audit_resume, audit_live
    from research.direct.latency58_fusion_refinement_checkpoint import load_model as load_parent
    parent, _ = load_parent(source["parent_checkpoint"])
    parent_sha = state_sha256(parent.state_dict())
    model = Latency58TemporalAttentionModel.from_parent(parent).train().requires_grad_(True)
    model.training_precision = "fp32"
    frozen = {name: value.clone() for name, value in model.named_buffers()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, foreach=False)
    generator = torch.Generator().manual_seed(202609145)
    audio = .03 * torch.randn(1, 2, 8 * 128, generator=generator)
    target = .02 * torch.randn(1, 4, 2, 8 * 128, generator=generator)

    def update(candidate, adam, step):
        adam.zero_grad(set_to_none=True)
        output = candidate.render(audio)
        loss = (output.deployed - target).square().mean() + .25 * (output.raw - target).square().mean()
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in candidate.parameters()),
                "A fixture parameter lacks a finite gradient")
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), 5., error_if_nonfinite=True, foreach=False)
        adam.step()
        audit_live(candidate, adam, step, frozen)

    audit_live(model, optimizer, 0, frozen)
    for step in (1, 2):
        update(model, optimizer, step)
    plan = {**source, "schema": "latency58-temporal-attention-training-plan-v1",
            "config": {**source["config"], "steps": 2}, "fixed_buffers_sha256": state_sha256(frozen),
            "parent_model_state_sha256": parent_sha, "parent_training_updates": parent.provenance["training_updates"]}
    write(evidence / "fixture-plan.json", plan)
    plan_sha = sha(evidence / "fixture-plan.json")
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
    require(recovered["model_state_sha256"] == state_sha256(model.state_dict()), "Model serialization changed tensors")
    with torch.inference_mode():
        first, second = model.eval().render(audio), restored.render(audio)
        require(all(torch.equal(getattr(first, key).view(torch.int32), getattr(second, key).view(torch.int32))
                    for key in ("raw", "deployed", "native_raw", "spectral", "waveform", "delayed_mixture"))
                and all(torch.equal(a.view(torch.int32), b.view(torch.int32))
                        for a, b in zip(first.state, second.state, strict=True)), "Serialized inference replay differs")
    rejected = []
    invalid = {**recovered, "model": {**recovered["model"],
               "temporal_output.weight": recovered["model"]["temporal_output.weight"].double()}}
    wrong_step = {**loaded["optimizer"], "step": 3}
    for name, operation, message in (
        ("wrong_tensor_dtype", lambda: load_payload(invalid), "Invalid temporal_attention inference tensors"),
        ("wrong_optimizer_step", lambda: audit_resume(restored, recovered, wrong_step, plan, plan_sha),
         "Saved optimizer belongs to another endpoint"),
        ("wrong_training_plan", lambda: audit_resume(restored, recovered, loaded["optimizer"], plan, "0" * 64),
         "Saved optimizer belongs to another endpoint"),
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
    restarted = torch.optim.Adam(restored.parameters(), lr=1e-4, foreach=False)
    restarted.load_state_dict(loaded["optimizer"]["optimizer"])
    audit_live(restored, restarted, 2, frozen)
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
    return {"status": "pass", "serialized_in_memory": serialized, "all_30_optimizer_states_checked": True,
            "inference_outputs_and_states_bit_exact": True, "resumed_third_update_and_adam_moments_bit_exact": True,
            "cpu_rng_formats_replay_exact": True, "rejected_invalid_cases": rejected,
            "parent_unchanged": True, "gpu_used": False, "checkpoint_files_written": False,
            "disk_generation_path_exercised": False, "quality_measured": False,
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
    source_path = PHASE / "fusion-refinement-001/plan.json"
    source = read(source_path)
    review_path = PHASE / "fusion-refinement-001/selection-review.json"
    review = read(review_path)
    require(review["status"] == "selected_for_research" and review["actual_root_exit_code"] == 0,
            "Use the completed selected fusion-refinement parent")
    verify_inputs(review)
    source = {**source, "parent_checkpoint": review["best_research_checkpoint"],
              "parent_model_state_sha256": "437179545299aac9a079da9e4eac6f29dd138ac68bfddb573696f77c110d8bf0"}
    require_space(source, 5_000_000)
    functional = read(PHASE / "temporal-attention-functional-001/result.json")
    execution = read(PHASE / "temporal-attention-functional-stage-001/execution.json")
    require(functional["status"] == "pass" and functional["source_bindings_unchanged"]
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"], "Functional model check must close successfully")
    functional_plan = read(PHASE / "temporal-attention-functional-001/plan.json")
    require(functional_plan["parent_checkpoint"] == source["parent_checkpoint"]
            and functional["parent_model_state_sha256"] == source["parent_model_state_sha256"],
            "Serializer must exercise the functionally checked current parent")
    paths = [source_path, review_path, Path(__file__).resolve(), PHASE / "temporal-attention-functional-001/result.json",
             PHASE / "temporal-attention-functional-stage-001/execution.json",
             ROOT / "research/direct/latency58_temporal_attention_checkpoint.py"]
    bindings = {**read(PHASE / "temporal-attention-functional-001/plan.json")["source_bindings"],
                **{str(path): sha(path) for path in paths}}
    verify_inputs({"source_bindings": bindings})
    out = PHASE / "temporal-attention-checkpoint-memory-functional-001"
    require(not out.exists(), "Preserve earlier checkpoint checks")
    out.mkdir()
    write(out / "plan.json", {"source_bindings": bindings, "scope": "In-memory serializer and CPU optimizer restart",
          "checkpoint_files_written": False, "production_recipe_selected": False})
    began = time.monotonic()
    result = check(source, out)
    verify_inputs({"source_bindings": bindings})
    result.update(source_bindings_unchanged=True, elapsed_seconds=time.monotonic() - began,
                  plan_sha256=sha(out / "plan.json"), counted_bytes_after=require_space(source, 0))
    write(out / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
