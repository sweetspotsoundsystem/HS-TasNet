"""Discarded GPU updates for both context arms on real production crops."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from research.direct.train_latency58 import PRODUCTION, continuity, load_source, read, require, sha, state_sha256, verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Context resource plan changed")
    plan = read(args.plan)
    verify_inputs(plan)
    warmup, scored = plan["warmup_samples"], plan["scored_samples"]
    require(plan["schema"] == "latency58-context-resource-v1" and (warmup, scored) in ((1024, 4096), (88064, 88064))
            and plan["teacher_weight"] == .5 and plan["teacher_kind"] in ("c91", "cropped11"), "Resource scope differs")
    require(all(os.environ.get(k) == v for k, v in plan["environment"].items()), "GPU environment differs")
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve context resource evidence")
    for binding in plan["functional_proofs"]:
        proof, execution = read(binding["result"]), read(binding["execution"])
        require(proof["status"] == "pass" and proof["source_bindings_unchanged"]
                and execution["actual_exit_code"] == 0 and not execution["timed_out"]
                and execution["source_bindings_unchanged"], "CPU context prerequisite failed")
    if scored == 88064:
        short, execution = read(plan["short_result"]), read(plan["short_execution"])
        monitor = read(execution["monitor_result"])
        require(short["status"] == "pass" and short["scored_samples"] == 4096 and short["warmup_samples"] == 1024
                and short["initial_model_state_sha256"] == plan["parent"]["model_state_sha256"]
                and short["teacher_kind"] == plan["teacher_kind"] and short["arms"] == [False, True]
                and execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                and monitor["status"] == monitor["supervisor_health"] == "pass"
                and monitor["child_exit_code"] == 0 and monitor["post_exit_quiet_completed"], "Short GPU prerequisite differs")
    watchdog = load_source("latency58_context_resource_watchdog", plan["watchdog_source"])
    newest, events = continuity(plan, watchdog)
    (out / "event-continuity.json").write_text(json.dumps(events, indent=2) + "\n")

    import torch
    from research.direct.latency58_context_checkpoint import load_parent
    from research.direct.latency58_sdr_context import render_scored_context, physical_context_teacher
    from research.direct.latency58_sdr_teacher import load_teacher
    from research.direct.latency58_teacher import deployed_teacher_l1
    from research.direct.latency_ola512_training import raw4_native_objective
    sys.path.insert(0, str(PRODUCTION))
    import train_production as production

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = load_parent(plan)
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    teacher, identity = load_teacher(plan["teacher_kind"], plan["teacher"])
    require(identity["model_state_sha256"] == plan["teacher_model_state_sha256"], "Resource teacher changed")
    production_config = read(PRODUCTION / "full_config.json")
    _, tracks, _, _ = production.load_corpus_manifest(
        PRODUCTION / "manifests/combined.manifest.json", expected_file_sha256=plan["manifest_sha256"], config=production_config)
    require(min(t.effective_frames for t in tracks) >= 176128, "Context corpus contains a short track")
    dataset = production.CounterAddressedCropDataset(
        tracks, root_weights=production_config["sampling"]["root_weights"], seed=production_config["seed"],
        crop_samples=176128, vocal_active_probability=production_config["sampling"]["vocal_active_probability"],
        final_sample_index=920004)
    examples = [dataset[i] for i in range(920000, 920004)]
    mixture, targets = (torch.stack([row[k][..., :warmup + scored] for row in examples]) for k in (0, 1))
    require(mixture.shape == (4, 2, warmup + scored) and targets.shape == (4, 4, 2, warmup + scored), "Resource batch differs")
    batch_hash = hashlib.sha256(mixture.numpy().tobytes() + targets.numpy().tobytes()).hexdigest()
    del dataset, examples, tracks
    require(torch.__version__ == plan["torch_version"] and torch.cuda.is_available()
            and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(), "Expected reviewed BF16 GPU")
    torch.cuda.set_per_process_memory_fraction(.75)
    production.configure_determinism(20260909)
    progress_file = (out / "progress.jsonl").open("x", buffering=1)
    progress_counter = 0

    def progress(phase):
        nonlocal progress_counter
        progress_counter += 1
        progress_file.write(json.dumps({"step": progress_counter, "phase": phase}) + "\n")
        print(phase, flush=True)

    began = time.monotonic()
    progress("context_cpu_gpu_fp32_parity")
    probe = torch.randn(1, 2, 2049) * .03
    with torch.no_grad():
        expected = [render_scored_context(model, probe, warmup_samples=1024, carry_state=c).deployed for c in (False, True)]
        teacher_expected = physical_context_teacher(teacher, probe, kind=plan["teacher_kind"], warmup_samples=1024)
        model.cuda()
        teacher.cuda()
        actual = [render_scored_context(model, probe.cuda(), warmup_samples=1024, carry_state=c).deployed for c in (False, True)]
        teacher_actual = physical_context_teacher(teacher, probe.cuda(), kind=plan["teacher_kind"], warmup_samples=1024)
        student_errors = [float((a.cpu() - e).abs().max()) for a, e in zip(actual, expected)]
        teacher_error = float((teacher_actual.cpu() - teacher_expected).abs().max())
        require(max(student_errors) <= 1e-5 and teacher_error <= 1e-5, "Context CPU/GPU waveform mismatch")
    del expected, actual, teacher_expected, teacher_actual
    mixture, targets = mixture.cuda().requires_grad_(), targets[..., warmup:].cuda()
    progress("full_physical_teacher_context")
    rng = torch.cuda.get_rng_state().clone()
    teacher_began = time.monotonic()
    teacher_targets = physical_context_teacher(teacher, mixture, kind=plan["teacher_kind"], warmup_samples=warmup)
    torch.cuda.synchronize()
    teacher_seconds = time.monotonic() - teacher_began
    teacher_peak_allocated = torch.cuda.max_memory_allocated()
    teacher_peak_reserved = torch.cuda.max_memory_reserved()
    require(torch.equal(rng, torch.cuda.get_rng_state()), "Context teacher changed augmentation RNG")
    rows = []
    for carry in (False, True):
        progress("reset_parent_" + str(carry))
        model.load_state_dict(initial, strict=True)
        require(state_sha256(model.state_dict()) == plan["parent"]["model_state_sha256"], "Resource arm did not reset its parent")
        model.train().requires_grad_(True)
        model.training_precision = "bf16"
        model.zero_grad(set_to_none=True)
        # A short analytic flush derivative checks the carried-state boundary
        # under the actual mixed-precision path before the full objective.
        x = probe.cuda().requires_grad_()
        tail = render_scored_context(model, x, warmup_samples=1024, carry_state=carry)
        tail.deployed.sum(dim=1)[..., -1].sum().backward()
        derivative = torch.zeros_like(x)
        derivative[..., -1] = 1
        flush_error = float((x.grad - derivative).abs().max())
        require(flush_error <= 1e-6, "BF16 analytic final-flush derivative differs")
        del tail, x, derivative
        model.zero_grad(set_to_none=True)
        mixture.grad = None
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-7, foreach=False)
        frozen = {name: value.clone() for name, value in model.named_buffers()}
        torch.cuda.reset_peak_memory_stats()
        update_began = time.monotonic()
        progress("student_forward_" + str(carry))
        output = render_scored_context(model, mixture, warmup_samples=warmup, carry_state=carry)
        require(output.initial_state_detached and torch.equal(output.physical_mixture, mixture[..., warmup:])
                and torch.equal(rng, torch.cuda.get_rng_state()), "Resource alignment, detached state or RNG differs")
        closure = float((output.deployed.sum(dim=1) - mixture[..., warmup:]).abs().max().detach())
        terms = raw4_native_objective(output.raw, targets, torch.zeros(4, dtype=torch.bool, device="cuda"), projection=True)
        auxiliary = deployed_teacher_l1(output.deployed, teacher_targets)
        loss = terms.total + plan["teacher_weight"] * auxiliary
        progress("backward_" + str(carry))
        loss.backward()
        require(closure <= 1e-6 and all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                and bool(torch.count_nonzero(p.grad)) for p in model.parameters())
                and mixture.grad is not None and bool(torch.isfinite(mixture.grad).all())
                and torch.count_nonzero(mixture.grad[..., :warmup]).item() == 0
                and bool((mixture.grad[..., -1].abs() > 0).all()), "Context gradients or mixture reconstruction differ")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True, foreach=False)
        progress("discarded_adam_update_" + str(carry))
        optimizer.step()
        torch.cuda.synchronize()
        for p in model.parameters():
            state = optimizer.state[p]
            require(p.dtype == torch.float32 and bool(torch.isfinite(p).all()) and float(state["step"]) == 1
                    and all(state[k].shape == p.shape and state[k].dtype == torch.float32 and state[k].device == p.device
                            and bool(torch.isfinite(state[k]).all()) for k in ("exp_avg", "exp_avg_sq")), "Invalid context Adam state")
        require(all(torch.equal(v, frozen[name]) for name, v in model.named_buffers()), "Fixed context buffers changed")
        rows.append({"carry_state": carry, "student_cpu_gpu_fp32_max_error": student_errors[int(carry)],
                     "initial_state_detached": True, "prefix_gradient_exactly_zero": True,
                     "final_sample_gradient_nonzero": True, "analytic_final_flush_gradient_max_abs": flush_error,
                     "closure_max_abs": closure, "finite_nonzero_parameter_gradients": 21, "fp32_adam_state_pairs": 21,
                     "loss": float(loss.detach()), "teacher_l1": float(auxiliary.detach()), "grad_norm": float(norm),
                     "update_seconds": time.monotonic() - update_began,
                     "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                     "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        del output, terms, auxiliary, loss, optimizer, frozen, state, p
        model.zero_grad(set_to_none=True)
        mixture.grad = None
        gc.collect()
        torch.cuda.empty_cache()
    require(state_sha256(teacher.state_dict()) == identity["model_state_sha256"]
            and all(not p.requires_grad and p.grad is None for p in teacher.parameters()), "Frozen context teacher changed")
    verify_inputs(plan)
    result = {"schema": "latency58-context-resource-result-v1", "status": "pass", "plan_sha256": args.plan_sha256,
              "teacher_kind": plan["teacher_kind"], "teacher_weight": plan["teacher_weight"], "teacher_identity": identity,
              "initial_model_state_sha256": plan["parent"]["model_state_sha256"], "warmup_samples": warmup,
              "scored_samples": scored, "arms": [False, True], "batch_size": 4, "unique_real_crops": 4,
              "sample_indices": list(range(920000, 920004)), "unaugmented_batch_sha256": batch_hash,
              "teacher_cpu_gpu_fp32_max_error": teacher_error, "teacher_forward_seconds": teacher_seconds,
              "teacher_phase_peak_allocated_bytes": teacher_peak_allocated,
              "teacher_phase_peak_reserved_bytes": teacher_peak_reserved,
              "teacher_unchanged_and_no_gradients": True, "arm_results": rows, "discarded_optimizer_updates": 2,
              "saved_checkpoints": 0, "source_bindings_unchanged": True, "pre_gpu_event_record_id": newest,
              "elapsed_seconds": time.monotonic() - began}
    from research.direct.run_latency58_quality import write
    write(out / "result.json", result)
    progress("completed")
    progress_file.close()
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
