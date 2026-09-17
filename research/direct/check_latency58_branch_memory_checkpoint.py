"""Check the branch-memory context, serializer and exact Adam restart in RAM."""
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
    from research.direct.latency58_branch_memory import Latency58BranchMemoryModel
    from research.direct.latency58_branch_memory_checkpoint import make_payloads, load_payload, audit_resume, audit_live
    from research.direct.latency58_temporal_attention_checkpoint import load_model as load_parent
    parent, _ = load_parent(source["parent_checkpoint"])
    parent_sha = state_sha256(parent.state_dict())
    model = Latency58BranchMemoryModel.from_parent(parent).train().requires_grad_(True)
    model.provenance = {**model.provenance,
                        "branch_memory_previous_provenance": dict(model.provenance),
                        "branch_memory_parent_model_state_sha256": parent_sha}
    model.training_precision = "fp32"
    with torch.no_grad():
        generator = torch.Generator().manual_seed(202610211)
        for projection in (model.spec_memory_output, model.waveform_memory_output):
            projection.weight.copy_(.002 * torch.randn(projection.weight.shape, generator=generator))
    from research.direct.check_latency58_branch_memory import compare_context
    context_generator = torch.Generator().manual_seed(202609151)
    context = compare_context(model, .03 * torch.randn(1, 2, 88064 + 8 * 128,
                              generator=context_generator), 88064)
    frozen = {name: value.clone() for name, value in model.named_buffers()}
    optimizer = torch.optim.Adam(model.parameters(), lr=6e-5, foreach=False)
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
    plan = {**source, "schema": "latency58-branch-memory-training-plan-v1",
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
               "spec_memory_output.weight": recovered["model"]["spec_memory_output.weight"].double()}}
    wrong_step = {**loaded["optimizer"], "step": 3}
    for name, operation, message in (
        ("wrong_tensor_dtype", lambda: load_payload(invalid), "Invalid branch_memory inference tensors"),
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
    restarted = torch.optim.Adam(restored.parameters(), lr=6e-5, foreach=False)
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
    return {"status": "pass", "trained_parent_context": context, "serialized_in_memory": serialized, "all_40_optimizer_states_checked": True,
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
    parent_root = PHASE / "attention-continuation-001"
    source_path = parent_root / "plan.json"
    source = read(source_path)
    review_path = parent_root / "selection-review.json"
    review = read(review_path)
    terminal = read(parent_root / "result.json")
    require(review["status"] == "selected_for_research" and review["actual_root_exit_code"] == 0
            and terminal["status"] == "training_audit_and_full14_complete"
            and terminal["checkpoint"] == review["best_research_checkpoint"]
            and terminal["full_sdr_db"] == 4.330200584072513 < 5.0,
            "Use the selected completed attention parent")
    verify_inputs(source)
    verify_inputs(review)
    paths = [Path(__file__).resolve(), source_path, review_path]
    paths.extend(parent_root / name for name in (
        "result.json", "production-stage/execution.json", "full14/execution.json", "full14/result.json",
        "production-run/checkpoint/model.pt", "production-run/checkpoint/optimizer.pt"))
    paths.extend(ROOT / "research/direct" / name for name in (
        "check_latency58_branch_memory.py", "latency58_branch_memory.py",
        "latency58_branch_memory_context.py", "latency58_branch_memory_checkpoint.py"))
    for name in ("production-stage/execution.json", "full14/execution.json"):
        execution = read(parent_root / name)
        require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"],
                "Parent training/scoring did not close")
    bindings = {**source["source_bindings"], **{str(path): sha(path) for path in paths}}
    verify_inputs({"source_bindings": bindings})
    source = {**source, "parent_checkpoint": terminal["checkpoint"],
              "parent_model_state_sha256": "94bf99892cfdabb134d76cf4238528129ab1427dbfc879337c9212ae6756e875",
              "parent_training_updates": 25250, "parent_kind": "saved_temporal_attention",
              "optimizer_initialization": "fresh_adam"}
    require_space(source, 385_000_000)
    out = PHASE / "branch-memory-checkpoint-functional-001"
    require(not out.exists(), "Preserve earlier checkpoint checks")
    out.mkdir()
    write(out / "plan.json", {"source_bindings": bindings, "parent_checkpoint": terminal["checkpoint"],
          "scope": "Branch-memory context parity on the selected attention parent, in-memory serialization and exact Adam restart; no production recipe selected",
          "checkpoint_files_written": False, "production_recipe_selected": False})
    began = time.monotonic()
    result = check(source, out)
    verify_inputs({"source_bindings": bindings})
    result.update(source_bindings_unchanged=True, elapsed_seconds=time.monotonic() - began,
                  plan_sha256=sha(out / "plan.json"), counted_bytes_after=require_space(source, 380_000_000))
    write(out / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
