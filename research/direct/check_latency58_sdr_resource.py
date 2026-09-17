"""One discarded GPU update for an authenticated SDR teacher/student pair."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from research.direct.train_latency58 import continuity, load_source, read, require, sha, state_sha256, verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Resource plan changed")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-sdr-resource-v1" and plan["samples"] in (4096, 88064)
            and plan["teacher_kind"] in ("c91", "cropped11") and plan["teacher_weight"] == 0.5,
            "Unsupported resource experiment")
    require(all(os.environ.get(k) == v for k, v in plan["environment"].items()), "GPU environment differs")
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve earlier resource result")
    functional = read(plan["functional_result"])
    require(functional["status"] == "pass" and read(plan["functional_execution"])["actual_exit_code"] == 0,
            "Functional prerequisite failed")
    if plan["samples"] == 88064:
        short = read(plan["short_result"])
        execution = read(plan["short_execution"])
        monitor = read(execution["monitor_result"])
        require(short["status"] == "pass" and short["samples"] == 4096
                and short["teacher_kind"] == plan["teacher_kind"] and execution["actual_exit_code"] == 0
                and monitor["status"] == "pass" and monitor["post_exit_quiet_completed"],
                "Full fixture requires the matching successful short fixture")
    watchdog = load_source("latency58_sdr_resource_watchdog", plan["watchdog_source"])
    newest, events = continuity(plan, watchdog)
    (out / "event-continuity.json").write_text(json.dumps(events, indent=2) + "\n")

    import torch
    import torch.nn.functional as F
    from research.direct.latency58_sdr_teacher import (
        STUDENT_STATE_SHA256, load_initial_student, load_teacher, physical_targets)
    from research.direct.latency58_teacher import deployed_teacher_l1
    from research.direct.latency_ola512_training import raw4_native_objective

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = load_initial_student()
    kind = plan["teacher_kind"]
    teacher, identity = load_teacher(kind, plan["teacher"])
    require(identity == functional["teachers"][kind]["identity"]
            and functional["student_model_state_sha256"] == STUDENT_STATE_SHA256,
            "Resource pair differs from CPU functional evidence")
    require(torch.__version__ == plan["torch_version"] and torch.cuda.is_available()
            and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(), "Expected one BF16 CUDA GPU")
    torch.cuda.set_per_process_memory_fraction(0.75)
    torch.manual_seed(20260908)
    torch.cuda.manual_seed_all(20260908)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    progress_file = (out / "progress.jsonl").open("x", buffering=1)
    progress_counter = 0

    def progress(phase):
        nonlocal progress_counter
        progress_counter += 1
        progress_file.write(json.dumps({"step": progress_counter, "phase": phase}) + "\n")
        print(phase, flush=True)

    progress("cpu_gpu_fp32_parity")
    with torch.no_grad():
        probe = torch.randn(1, 2, 1025) * .03
        expected = model.render(F.pad(probe, (0, (-1025) % 128 + 128)))
        teacher_expected = physical_targets(teacher, probe, kind=kind)
        model.cuda()
        teacher.cuda()
        actual = model.render(F.pad(probe.cuda(), (0, (-1025) % 128 + 128)))
        teacher_actual = physical_targets(teacher, probe.cuda(), kind=kind)
        student_error = float((actual.deployed.cpu() - expected.deployed).abs().max())
        teacher_error = float((teacher_actual.cpu() - teacher_expected).abs().max())
        require(student_error <= 1e-5 and teacher_error <= 1e-5, "CPU/GPU FP32 waveform mismatch")
        for index, (a, b) in enumerate(zip(actual.state, expected.state, strict=True)):
            scale = 2.0**-18 if index == 1 else 1.0
            require(torch.allclose(a.cpu() / scale, b / scale, atol=5e-4, rtol=5e-5),
                    "CPU/GPU physical state mismatch")
    del expected, actual, teacher_expected, teacher_actual
    cached = torch.load(plan["cached_batch"], map_location="cpu", weights_only=True)
    require(cached["mixture"].shape == (2, 2, 88064) and cached["targets"].shape == (2, 4, 2, 88064),
            "Frozen production-pair geometry differs")
    samples = plan["samples"]
    mixture = cached["mixture"][..., :samples].repeat(2, 1, 1).cuda().requires_grad_()
    targets = cached["targets"][..., :samples].repeat(2, 1, 1, 1).cuda()
    del cached
    model.train().requires_grad_(True)
    model.training_precision = "bf16"
    frozen = {name: value.clone() for name, value in model.named_buffers()}
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-7, foreach=False)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    began = time.monotonic()
    progress("frozen_teacher_forward")
    rng = torch.cuda.get_rng_state().clone()
    teacher_began = time.monotonic()
    teacher_targets = physical_targets(teacher, mixture, kind=kind)
    torch.cuda.synchronize()
    teacher_seconds = time.monotonic() - teacher_began
    require(torch.equal(rng, torch.cuda.get_rng_state()), "Teacher changed augmentation RNG")
    progress("student_forward")
    output = model.render(F.pad(mixture, (0, 128)))
    raw, deployed = (v[..., 128:128 + samples] for v in (output.raw, output.deployed))
    output.raw.retain_grad()
    require(torch.equal(output.delayed_mixture[..., 128:128 + samples], mixture), "Student alignment mismatch")
    closure = float((deployed.sum(dim=1) - mixture).abs().max().detach())
    require(closure <= 1e-6, "Student reconstruction mismatch")
    terms = raw4_native_objective(raw, targets, torch.zeros(4, dtype=torch.bool, device="cuda"), projection=True)
    auxiliary = deployed_teacher_l1(deployed, teacher_targets)
    total = terms.total + plan["teacher_weight"] * auxiliary
    with torch.no_grad():
        expected_gradient = torch.sign(raw - targets) / targets.numel()
        teacher_sign = torch.sign(deployed - teacher_targets)
        expected_gradient[:, :3] += plan["teacher_weight"] * (
            teacher_sign[:, :3] - teacher_sign[:, 3:4]) / teacher_targets.numel()
    progress("backward")
    total.backward()
    torch.cuda.synchronize()
    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool(torch.count_nonzero(p.grad))
                for p in model.parameters()), "Missing or nonfinite learned gradient")
    flush_error = float((output.raw.grad[..., -128:] - expected_gradient[..., -128:]).abs().max())
    require(torch.count_nonzero(output.raw.grad[..., :128]).item() == 0 and flush_error <= 1e-12
            and mixture.grad is not None and bool(torch.isfinite(mixture.grad).all())
            and bool((mixture.grad[..., -1].abs() > 0).all()), "Preroll or final-real-sample gradient mismatch")
    progress("discarded_adam_update")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True, foreach=False)
    optimizer.step()
    torch.cuda.synchronize()
    for parameter in model.parameters():
        state = optimizer.state[parameter]
        require(parameter.dtype == torch.float32 and bool(torch.isfinite(parameter).all())
                and float(state["step"]) == 1
                and all(state[k].shape == parameter.shape and state[k].dtype == torch.float32
                        and state[k].device == parameter.device and bool(torch.isfinite(state[k]).all())
                        for k in ("exp_avg", "exp_avg_sq")), "Invalid FP32 Adam state")
    require(all(torch.equal(value, frozen[name]) for name, value in model.named_buffers()), "Fixed buffers changed")
    require(state_sha256(teacher.cpu().state_dict()) == identity["model_state_sha256"]
            and all(not p.requires_grad and p.grad is None for p in teacher.parameters()), "Teacher changed")
    verify_inputs(plan)
    result = {"schema": "latency58-sdr-resource-result-v1", "status": "pass", "teacher_kind": kind,
              "teacher_identity": identity, "teacher_weight": plan["teacher_weight"],
              "initial_model_state_sha256": STUDENT_STATE_SHA256, "samples": samples, "batch_size": 4,
              "teacher_unchanged_and_no_gradients": True, "student_cpu_gpu_fp32_max_error": student_error,
              "teacher_cpu_gpu_fp32_max_error": teacher_error, "teacher_forward_seconds": teacher_seconds,
              "mixture_closure_max_error": closure, "final_flush_gradient_max_error": flush_error,
              "finite_nonzero_parameter_gradients": 21, "fp32_adam_state_pairs": 21,
              "discarded_optimizer_updates": 1, "loss": float(total.detach()),
              "teacher_l1": float(auxiliary.detach()), "unclipped_gradient_norm": float(norm),
              "update_seconds_with_audits": time.monotonic() - began,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "pre_gpu_event_record_id": newest,
              "saved_checkpoints": 0, "source_bindings_unchanged": True, "plan_sha256": args.plan_sha256}
    with (out / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    progress("completed")
    progress_file.close()
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
