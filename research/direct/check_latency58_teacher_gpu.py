"""One discarded frozen-teacher GPU resource update; no saved checkpoint.

The short fixture precedes the full B4/88064-sample cost fixture. Reuse the
authenticated production pair from earlier latency experiments, repeated twice.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Resource plan changed")
    plan = json.loads(args.plan.read_text())
    require(plan["schema"] == "latency58-teacher-gpu-resource-v1" and plan["samples"] in (4096, 88064),
            "Unexpected bounded resource fixture")
    require(all(sha(path) == digest for path, digest in plan["source_bindings"].items()),
            "A resource input changed")
    require(all(os.environ.get(key) == value for key, value in plan["environment"].items())
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "Explicit GPU0 environment required")
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists()
            and not (out / "progress.jsonl").exists(), "Preserve previous resource outputs")
    prerequisite = json.loads(Path(plan["prerequisite_execution"]).read_text())
    require(prerequisite["actual_exit_code"] == 0 and prerequisite["source_bindings_unchanged"],
            "CPU functional prerequisite did not finish successfully")
    if plan["samples"] == 88064:
        short = json.loads(Path(plan["short_resource_terminal"]).read_text())
        require(short["status"] == "pass" and short["child_exit_code"] == 0
                and short["supervisor_health"] == "pass" and short["post_exit_quiet_completed"],
                "Short GPU fixture and monitor must finish first")

    # Check the gap since the previous monitored GPU stage, before importing Torch.
    spec = importlib.util.spec_from_file_location("latency58_resource_monitor", plan["watchdog_source"])
    monitor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monitor)
    script = monitor.event_query(plan["previous_event_record_id"], verify_sentinel=True)
    response = subprocess.run([monitor.POWERSHELL, "-NoProfile", "-NonInteractive", "-EncodedCommand",
                               base64.b64encode(script.encode("utf-16le")).decode()],
                              capture_output=True, text=True, timeout=10)
    require(response.returncode == 0, "Pre-GPU Windows event query failed")
    payload = json.loads(response.stdout)
    newest, rows = monitor.validate_events(payload, plan["previous_event_record_id"])
    require(payload["SentinelVerified"] and not any(monitor.reset_event(row) for row in rows),
            "New host fault since the previous monitored GPU stage")
    (out / "continuity.json").write_text(json.dumps(payload, indent=2) + "\n")

    import torch
    from research.direct.latency58 import HOP
    from research.direct.latency58_asymmetric import Latency58AsymmetricModel, AsymmetricState
    from research.direct.latency58_asymmetric_checkpoint import load_model_state, make_model
    from research.direct.latency58_teacher import load_frozen_teacher, physical_teacher_targets, deployed_teacher_l1
    from research.direct.latency58_evaluate import model_state_sha256
    from research.direct.latency_ola512_training import raw4_native_objective

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = make_model(plan["parent_checkpoint"])
    require(load_model_state(model, plan["checkpoint"]) == 500
            and model_state_sha256(model) == plan["model_state_sha256"], "Student checkpoint differs")
    proof = json.loads(Path(plan["functional_result"]).read_text())
    require(proof["status"] == "pass" and proof["student_checkpoint"] == plan["checkpoint"],
            "Resource model differs from functional evidence")
    teacher, identity = load_frozen_teacher(plan["teacher_export_plan"])
    teacher_sha = model_state_sha256(teacher)
    require(teacher_sha == proof["teacher_identity"]["model_state_sha256"]
            == identity["model_state_sha256"] and plan["teacher_weight"] == 0.5, "Frozen teacher/objective differs")
    require(torch.__version__ == plan["torch_version"], "Unexpected Torch runtime")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1
            and torch.cuda.is_bf16_supported(), "One BF16 CUDA GPU is required; no CPU fallback")
    torch.cuda.set_per_process_memory_fraction(0.75)
    torch.manual_seed(20260907)
    torch.cuda.manual_seed_all(20260907)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    progress_file = (out / "progress.jsonl").open("x", buffering=1)
    progress_counter = 0

    def progress(phase):
        nonlocal progress_counter
        progress_counter += 1
        progress_file.write(json.dumps({"step": progress_counter, "phase": phase}) + "\n")
        progress_file.flush()
        print(phase, flush=True)

    progress("construct_authenticated_model")
    cpu_probe = torch.randn(1, 2, 3 * HOP) * 0.1
    with torch.no_grad():
        expected = model.render(cpu_probe)
        model.cuda()
        actual = model.render(cpu_probe.cuda())
        parity = float((actual.deployed.cpu() - expected.deployed).abs().max())
        require(parity <= 1e-5, "GPU FP32 output differs from CPU reference")
        for index, (left, right) in enumerate(zip(actual.state, expected.state, strict=True)):
            scale = 2.0**-18 if index == 1 else 1.0
            require(torch.allclose(left.cpu() / scale, right / scale, atol=5e-4, rtol=5e-5),
                    "GPU FP32 state differs from CPU reference")
        teacher_probe = torch.randn(1, 2, 1025) * 0.1
        teacher_expected = physical_teacher_targets(teacher, teacher_probe)
        teacher.cuda()
        teacher_actual = physical_teacher_targets(teacher, teacher_probe.cuda())
        teacher_parity = float((teacher_actual.cpu() - teacher_expected).abs().max())
        require(teacher_parity <= 1e-5, "Frozen FP32 GPU teacher differs from CPU")
    del expected, actual, teacher_expected, teacher_actual
    progress("cpu_gpu_fp32_parity_passed")
    batch = torch.load(plan["cached_batch"], map_location="cpu", weights_only=True)
    require(batch["mixture"].shape == (2, 2, 88064) and batch["targets"].shape == (2, 4, 2, 88064),
            "Frozen production pair shape differs")
    samples = plan["samples"]
    mixture = batch["mixture"][..., :samples].repeat(2, 1, 1).cuda().requires_grad_()
    targets = batch["targets"][..., :samples].repeat(2, 1, 1, 1).cuda()
    require(torch.isfinite(mixture).all().item() and torch.isfinite(targets).all().item(),
            "Resource pair contains nonfinite samples")
    del batch
    flags = torch.zeros(4, dtype=torch.bool, device="cuda")
    model.train().requires_grad_(True)
    model.training_precision = "bf16"
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-7, foreach=False)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    progress("frozen_teacher_forward")
    teacher_started = time.perf_counter()
    teacher_targets = physical_teacher_targets(teacher, mixture)
    torch.cuda.synchronize()
    teacher_seconds = time.perf_counter() - teacher_started
    progress("student_forward")
    joined = torch.cat((mixture, mixture.new_zeros((4, 2, HOP))), dim=-1)
    output = model.render(joined)
    require(isinstance(output.state, AsymmetricState)
            and all(value.dtype == torch.float32 and torch.isfinite(value).all().item()
                    for value in (*output.state, output.raw)), "BF16 path broke the FP32 state/output ABI")
    output.raw.retain_grad()
    raw = output.raw[..., HOP:HOP + samples]
    deployed = output.deployed[..., HOP:HOP + samples]
    require(torch.equal(output.delayed_mixture[..., HOP:HOP + samples], mixture),
            "Resource training alignment lost real samples")
    closure = float((deployed.sum(dim=1) - mixture).abs().max().detach())
    require(closure <= 1e-6, "Resource four-stem closure failed")
    terms = raw4_native_objective(raw, targets, flags, projection=True)
    require(float(terms.projection.detach()) == 0, "No-derangement cost fixture produced a projection term")
    auxiliary = deployed_teacher_l1(deployed, teacher_targets)
    total = terms.total + plan["teacher_weight"] * auxiliary
    with torch.no_grad():
        expected_gradient = torch.sign(raw - targets) / targets.numel()
        teacher_sign = torch.sign(deployed - teacher_targets)
        expected_gradient[:, :3] += plan["teacher_weight"] * (
            teacher_sign[:, :3] - teacher_sign[:, 3:4]) / teacher_targets.numel()
    require(all(not p.requires_grad and p.grad is None for p in teacher.parameters()),
            "Teacher gained gradients")
    progress("backward")
    total.backward()
    torch.cuda.synchronize()
    require(all(p.grad is not None and torch.isfinite(p.grad).all().item()
                and torch.count_nonzero(p.grad).item() > 0 for p in model.parameters()),
            "A learned tensor has missing, nonfinite or zero gradients")
    flush_error = float((output.raw.grad[..., -HOP:] - expected_gradient[..., -HOP:]).abs().max())
    require(torch.count_nonzero(output.raw.grad[..., :HOP]).item() == 0 and flush_error <= 1e-12,
            "Global pre-roll cut or final flush L1 gradient differs")
    require(mixture.grad is not None and torch.isfinite(mixture.grad).all().item()
            and torch.count_nonzero(mixture.grad[..., -HOP:]).item() > 0,
            "Final real input has no finite gradient through flush")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True, foreach=False)
    progress("discarded_adam_update")
    optimizer.step()
    torch.cuda.synchronize()
    for p in model.parameters():
        saved = optimizer.state[p]
        require(p.dtype == torch.float32 and torch.isfinite(p).all().item()
                and float(saved["step"]) == 1
                and all(saved[key].shape == p.shape and saved[key].dtype == torch.float32
                        and saved[key].device == p.device and torch.isfinite(saved[key]).all().item()
                        for key in ("exp_avg", "exp_avg_sq")), "GPU FP32 Adam state audit failed")
    require(model_state_sha256(teacher) == teacher_sha
            and all(not p.requires_grad and p.grad is None for p in teacher.parameters()),
            "Teacher parameters changed or acquired gradients")
    require(all(torch.equal(value, buffers[name]) for name, value in model.named_buffers()),
            "Resource update changed fixed buffers or gains")
    result = {"status": "pass", "family": model.architecture_metadata["version"],
              "parent_checkpoint": plan["parent_checkpoint"], "initial_model_state_sha256": plan["model_state_sha256"],
              "student_checkpoint": plan["checkpoint"], "teacher_model_state_sha256": teacher_sha,
              "teacher_unchanged_and_no_gradients": True, "teacher_weight": plan["teacher_weight"],
              "teacher_cpu_gpu_fp32_max_error": teacher_parity, "teacher_forward_seconds": teacher_seconds,
              "objective": "raw4_l1_plus_half_deployed_teacher_l1", "samples": samples, "batch_size": 4, "data_hops": samples // HOP,
              "flush_hops": 1, "graph_delay_samples": HOP, "group_hops": None,
              "training_precision": "bf16_dense_fp32_synthesis_state_parameters_adam",
              "cpu_gpu_fp32_max_error": parity, "mixture_closure_max_error": closure,
              "final_flush_gradient_max_error": flush_error, "finite_nonzero_parameter_gradients": 21,
              "fp32_adam_state_pairs": 21, "discarded_optimizer_updates": 1,
              "loss": float(total.detach()), "supervised_loss": float(terms.total.detach()),
              "teacher_l1": float(auxiliary.detach()), "unclipped_gradient_norm": float(norm),
              "update_seconds_with_audits": time.perf_counter() - started,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "pre_gpu_event_record_id": newest, "saved_checkpoints": 0,
              "source_bindings_unchanged": all(sha(path) == digest for path, digest in plan["source_bindings"].items()),
              "plan_sha256": args.plan_sha256, "host_latency_qualified": False}
    require(result["source_bindings_unchanged"], "Resource inputs changed during execution")
    with (out / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    progress("completed")
    progress_file.close()
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
