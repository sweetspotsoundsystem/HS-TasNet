"""Exercise separate raw/Adam and EMA disk publication on a CPU fixture."""
import argparse
import copy
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs


def check(plan, out):
    import torch
    from research.direct.latency58_branch_ema import BranchParameterEMA
    from research.direct.latency58_branch_memory_checkpoint import load_model, audit_live, audit_resume
    from research.direct.latency58_branch_ema_checkpoint import save_generation, load_generation, audit_saved, audit_payloads
    fixture = plan["fixture_training_plan"]
    fixture_sha = sha(out / "fixture-plan.json")
    parent, _ = load_model(fixture["parent_checkpoint"])
    parent_sha = state_sha256(parent.state_dict())
    require(parent_sha == fixture["parent_model_state_sha256"], "Checkpoint fixture parent changed")
    model = copy.deepcopy(parent).train().requires_grad_(True)
    model.training_precision = "fp32"
    model.provenance = {**model.provenance, "branch_memory_previous_provenance": dict(model.provenance),
                        "branch_memory_parent_model_state_sha256": parent_sha}
    frozen = {name: b.clone() for name, b in model.named_buffers()}
    optimizer = torch.optim.Adam(model.parameters(), lr=6e-5, foreach=False)
    ema = BranchParameterEMA(model, decay=fixture["ema"]["decay"], base_state_sha256=parent_sha)
    generator = torch.Generator().manual_seed(202610261)
    audio = .03 * torch.randn(1, 2, 8 * 128, generator=generator)
    target = .02 * torch.randn(1, 4, 2, 8 * 128, generator=generator)

    def update(candidate, adam, average, step):
        candidate.train()
        adam.zero_grad(set_to_none=True)
        result = candidate.render(audio)
        loss = (result.deployed - target).square().mean() + .25 * (result.raw - target).square().mean()
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in candidate.parameters()),
                "Disk fixture lacks a finite gradient")
        norm = torch.nn.utils.clip_grad_norm_(candidate.parameters(), 5., error_if_nonfinite=True, foreach=False)
        adam.step()
        audit_live(candidate, adam, step, frozen)
        average.update(candidate, step=step)
        return {"step": step, "loss": float(loss.detach()), "grad_norm": float(norm),
                "raw_state_sha256": state_sha256(candidate.state_dict()),
                "ema_parameters_sha256": state_sha256(average.parameters)}

    journal = [update(model, optimizer, ema, step) for step in (1, 2)]
    run = out / "fixture-run"
    run.mkdir()
    with (run / "metrics.jsonl").open("x") as stream:
        for row in journal:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    binding = save_generation(model, optimizer, ema, 2, fixture, fixture_sha, run)
    audit = audit_saved(binding, fixture, fixture_sha)
    raw, resume, averaged, restored_ema, payloads = load_generation(binding, fixture, fixture_sha)
    require(audit["status"] == "pass" and audit["saved_optimizer_tensor_count"] == 40
            and audit["raw_model_state_sha256"] != audit["model_state_sha256"],
            "Fixture must exercise distinct raw and EMA weights")
    # Exercise the unchanged production inference loader directly on model.pt.
    deployed, deployed_payload = load_model(binding)

    def compare_outputs(first, second):
        with torch.inference_mode():
            a, b = first.eval().render(audio), second.eval().render(audio)
        require(all(torch.equal(getattr(a, k).view(torch.int32), getattr(b, k).view(torch.int32))
                    for k in ("raw", "deployed", "native_raw", "spectral", "waveform", "delayed_mixture"))
                and len(a.state) == len(b.state) == 8
                and all(torch.equal(x.view(torch.int32), y.view(torch.int32))
                        for x, y in zip(a.state, b.state, strict=True)), "Disk fixture inference replay differs")

    compare_outputs(model, raw)
    compare_outputs(ema.inference_copy(model), averaged)
    compare_outputs(averaged, deployed)
    require(deployed_payload["provenance"]["checkpoint_weight_role"] == "averaged_inference",
            "Existing inference loader lost the EMA provenance")
    rejected = []
    def reject(name, operation):
        try:
            operation()
        except RuntimeError as error:
            rejected.append({"case": name, "reason": str(error)})
        else:
            raise RuntimeError("Invalid EMA disk case accepted: " + name)

    reject("raw_optimizer_with_averaged_weights", lambda: audit_resume(
        averaged, payloads["averaged"], resume, fixture, fixture_sha))
    reject("wrong_inference_file_hash", lambda: load_generation(
        {**binding, "sha256": "0" * 64}, fixture, fixture_sha))
    reject("wrong_plan_hash", lambda: load_generation(binding, fixture, "0" * 64))
    reject("wrong_ema_decay", lambda: load_generation(binding,
        {**fixture, "ema": {**fixture["ema"], "decay": .99}}, fixture_sha))
    reject("changed_raw_ownership", lambda: audit_payloads(
        {**payloads["raw"], "provenance": {**payloads["raw"]["provenance"], "checkpoint_weight_role": "averaged_inference"}},
        resume, payloads["averaged"], payloads["metadata"], fixture, fixture_sha))
    reject("changed_averaged_lineage", lambda: audit_payloads(
        payloads["raw"], resume, {**payloads["averaged"], "provenance": {
            **payloads["averaged"]["provenance"], "weight_averaging": {}}}, payloads["metadata"], fixture, fixture_sha))
    reject("changed_ema_metadata", lambda: audit_payloads(payloads["raw"], resume, payloads["averaged"],
        {**payloads["metadata"], "raw_state_sha256": "0" * 64}, fixture, fixture_sha))
    file_hashes = {p.name: sha(p) for p in Path(binding["path"]).parent.iterdir()}
    reject("existing_generation_preserved", lambda: save_generation(model, optimizer, ema, 2, fixture, fixture_sha, run))
    pending_run = out / "pending-collision-fixture"
    pending_run.mkdir()
    (pending_run / "checkpoint.pending").mkdir()
    reject("existing_pending_preserved", lambda: save_generation(model, optimizer, ema, 2, fixture, fixture_sha, pending_run))
    require(list((pending_run / "checkpoint.pending").iterdir()) == []
            and {p.name: sha(p) for p in Path(binding["path"]).parent.iterdir()} == file_hashes,
            "Collision checks changed saved artifacts")
    raw.train().requires_grad_(True)
    raw.training_precision = "fp32"
    restarted = torch.optim.Adam(raw.parameters(), lr=6e-5, foreach=False)
    restarted.load_state_dict(resume["optimizer"])
    live_next = update(model, optimizer, ema, 3)
    restored_next = update(raw, restarted, restored_ema, 3)
    require(live_next == restored_next, "Disk-restored next update differs")
    for p, q in zip(model.parameters(), raw.parameters(), strict=True):
        require(all(torch.equal(optimizer.state[p][k], restarted.state[q][k])
                    for k in ("step", "exp_avg", "exp_avg_sq")), "Disk-restored Adam state differs")
    compare_outputs(model, raw)
    compare_outputs(ema.inference_copy(model), restored_ema.inference_copy(raw))
    require(state_sha256(parent.state_dict()) == parent_sha and not torch.cuda.is_initialized(),
            "Disk fixture parent or CPU scope changed")
    return {"status": "pass", "checkpoint": binding, "saved_audit": audit,
            "file_bytes": {name: value["bytes"] for name, value in payloads["receipt"]["files"].items()},
            "raw_optimizer_ownership_checked": True, "existing_inference_loader_bit_exact": True,
            "saved_raw_and_ema_outputs_and_eight_states_bit_exact": True,
            "resumed_third_update_and_all_40_adam_states_bit_exact": True,
            "resumed_third_ema_update_bit_exact": True, "rejected_cases": rejected,
            "parent_unchanged": True, "gpu_used": False, "checkpoint_files_written": True,
            "disk_publication_exercised": True, "quality_measured": False,
            "limitation": "Synthetic CPU MSE updates only. Exercises successful fsync/rename publication and collision rejection, not power-loss recovery, GPU parity or RNG replay, full14 quality, or native runtime cost."}


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
            "Require frozen input and CUDA-hidden CPU1 disk fixture")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    plan = read(args.plan)
    require(plan["schema"] == "latency58-branch-ema-disk-functional-plan-v1", "Wrong EMA disk fixture plan")
    out = Path(plan["output_directory"])
    require(out.resolve() == args.plan.parent.resolve() and out.resolve().is_relative_to(PHASE)
            and not (out / "result.json").exists(), "Preserve EMA disk results")
    verify_inputs(plan)
    counted = require_space(plan, 1_050_000_000)
    write(out / "fixture-plan.json", plan["fixture_training_plan"])
    started = time.monotonic()
    result = check(plan, out)
    verify_inputs(plan)
    require_space(plan, 450_000_000)
    result.update(plan_sha256=args.plan_sha256, source_bindings=plan["source_bindings"],
                  elapsed_seconds=time.monotonic() - started, counted_bytes_before=counted,
                  forecast_before_including_fixture_and_training=counted + 1_050_000_000 + plan["outside_roots_reservation_bytes"],
                  fixture_plan_sha256=sha(out / "fixture-plan.json"))
    write(out / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "source_bindings"}), flush=True)


if __name__ == "__main__":
    main()
