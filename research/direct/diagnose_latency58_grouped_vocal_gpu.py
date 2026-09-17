"""Measure the failed BF16 grouped-gradient comparison without changing weights.

This is a diagnostic, not a relaxed resource qualification or training run.
The original failing plan, checker and tolerances remain immutable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, read, require, sha, write
from research.direct.train_latency58 import PRODUCTION, verify_inputs, state_sha256, load_source, continuity
from research.direct.run_latency58_deployed_vocal_views import budget_snapshot

FAILED = PHASE / "branch-grouped-vocal-007"
OUT = PHASE / "grouped-vocal-gradient-diagnostic-001"


def tensor_error(actual, expected):
    import torch
    difference = (actual - expected).double()
    require(bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()), "Nonfinite diagnostic gradient")
    norm = expected.double().norm()
    return {"maximum_absolute_error": float(difference.abs().max()),
            "relative_l2_error": float(difference.norm() / norm.clamp_min(1e-30)),
            "reference_norm": float(norm), "bitwise_equal": torch.equal(actual, expected),
            "original_elementwise_tolerance_pass": torch.allclose(actual, expected, atol=1e-7, rtol=1e-4),
            "original_relative_l2_tolerance_pass": bool(difference.norm() / norm.clamp_min(1e-30) < 5e-5),
            "different_elements": int(torch.count_nonzero(actual != expected)),
            "different_after_bf16_cast": int(torch.count_nonzero(actual.bfloat16() != expected.bfloat16()))}


def compare(model, progress):
    import torch
    from research.direct.latency58_branch_memory_context import render_scored_context
    from research.direct.latency58_grouped_vocal_auxiliary import source_views, prepare_groups, contribution, AUXILIARY_WEIGHT
    from research.direct.latency58_branch_sdr_blend import objective as whole_objective
    from research.direct.latency58_long_context_data import CROP_SAMPLES, WARMUP_SAMPLES
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(202610310)
    truth = .02 * torch.randn(16, 4, 2, CROP_SAMPLES, generator=generator)
    truth[:3, 2] = 0; truth[4:6, 1] = 0; truth[8:9, 3] = 0
    ordinary = truth.sum(1), truth
    auxiliary = source_views(*ordinary)
    inputs = {"ordinary": tuple(v.to(device) for v in ordinary),
              "auxiliary": tuple(v.to(device) for v in auxiliary)}
    groups = prepare_groups(inputs["ordinary"][1][..., WARMUP_SAMPLES:], inputs["auxiliary"][1][..., WARMUP_SAMPLES:])
    outputs, reference_vjps, grouped_vjps, boundaries = {}, {}, {}, {}
    for group, microbatch in (("ordinary", 4), ("auxiliary", 2)):
        audio, targets = inputs[group]
        captured = [[], []]
        for offset in range(0, len(audio), microbatch):
            result = render_scored_context(model, audio[offset:offset + microbatch], warmup_samples=WARMUP_SAMPLES, carry_state=True)
            captured[0].append(result.raw.detach().clone()); captured[1].append(result.deployed.detach().clone())
            del result
            progress("capture", group, offset)
        raw, deployed = (torch.cat(values).requires_grad_() for values in captured)
        del captured
        weight = 1. if group == "ordinary" else AUXILIARY_WEIGHT
        value = weight * whole_objective(raw, deployed, targets[..., WARMUP_SAMPLES:], audio[..., WARMUP_SAMPLES:]).total
        reference_loss = float(value.detach())
        reference_vjps[group] = tuple(v.detach() for v in torch.autograd.grad(value, (raw, deployed)))
        outputs[group] = raw.detach(), deployed.detach()
        del raw, deployed, value
        pieces, loss = [[], []], 0.
        for offset in range(0, len(audio), microbatch):
            end = offset + microbatch
            raw, deployed = (v[offset:end].detach().clone().requires_grad_() for v in outputs[group])
            value, terms = contribution(group, raw, deployed, targets[offset:end, ..., WARMUP_SAMPLES:],
                                        audio[offset:end, ..., WARMUP_SAMPLES:], groups)
            loss += float(value.detach())
            for values, derivative in zip(pieces, torch.autograd.grad(value, (raw, deployed)), strict=True):
                values.append(derivative.detach())
            del raw, deployed, value, terms
        grouped_vjps[group] = tuple(torch.cat(values) for values in pieces)
        boundaries[group] = {"whole_loss": reference_loss, "summed_contribution_loss": loss,
            "scalar_absolute_error": abs(loss - reference_loss),
            "output_derivatives": {name: tensor_error(a, b) for name, a, b in
                zip(("raw", "deployed"), grouped_vjps[group], reference_vjps[group], strict=True)}}
        print(json.dumps({"event": "loss_boundary", "group": group, "result": boundaries[group]}), flush=True)
    gradients, group_cumulative = {}, {}
    for mode in ("independent_whole_group_vjp", "grouped_contributions", "detached_grouped_vjp_replay"):
        model.zero_grad(set_to_none=True)
        group_cumulative[mode] = {}
        for group, microbatch in (("ordinary", 4), ("auxiliary", 2)):
            audio, targets = inputs[group]
            for offset in range(0, len(audio), microbatch):
                end = offset + microbatch
                physical = audio[offset:end].detach().clone().requires_grad_()
                result = render_scored_context(model, physical, warmup_samples=WARMUP_SAMPLES, carry_state=True)
                require(torch.equal(result.raw, outputs[group][0][offset:end]) and
                        torch.equal(result.deployed, outputs[group][1][offset:end]), "Replay outputs differ")
                if mode == "grouped_contributions":
                    value, terms = contribution(group, result.raw, result.deployed, targets[offset:end, ..., WARMUP_SAMPLES:],
                                                audio[offset:end, ..., WARMUP_SAMPLES:], groups)
                    value.backward()
                    del value, terms
                else:
                    vjps = reference_vjps if mode == "independent_whole_group_vjp" else grouped_vjps
                    torch.autograd.backward((result.raw, result.deployed), tuple(v[offset:end] for v in vjps[group]))
                require(physical.grad is not None and bool(torch.isfinite(physical.grad).all())
                        and torch.count_nonzero(physical.grad[..., :WARMUP_SAMPLES]) == 0
                        and torch.count_nonzero(physical.grad[..., WARMUP_SAMPLES:]) > 0, "Input gradient contract differs")
                del result, physical
                progress(mode, group, offset)
            group_cumulative[mode][group] = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()}
        gradients[mode] = group_cumulative[mode]["auxiliary"]
    reference, direct, replay = (gradients[name] for name in
        ("independent_whole_group_vjp", "grouped_contributions", "detached_grouped_vjp_replay"))
    comparisons = {}
    for label, actual, expected in (("direct_vs_whole", direct, reference), ("replay_vs_direct", replay, direct),
            ("ordinary_direct_vs_whole", group_cumulative["grouped_contributions"]["ordinary"],
                                       group_cumulative["independent_whole_group_vjp"]["ordinary"])):
        comparisons[label] = {name: tensor_error(actual[name], gradient) for name, gradient in expected.items()}
        require(len(comparisons[label]) == 40, "Parameter inventory differs")
    model.zero_grad(set_to_none=True)
    return {"boundaries": boundaries, "all_parameter_comparisons": comparisons,
            "same_outputs_on_all_replays": True, "warmup_input_gradients_zero": True,
            "training_optimizer_updates": 0, "checkpoint_files_written": False, "quality_measured": False}


def prepare():
    require(Path.cwd() == ROOT and not OUT.exists(), "Preserve diagnostic output")
    plan = read(FAILED / "plan.json")
    root = read(FAILED / "resource-root-execution.json")
    execution = read(FAILED / "resource-stage/execution.json")
    monitor_path = Path(execution["monitor_result"])
    monitor = read(monitor_path)
    require(root["actual_exit_code"] == execution["actual_exit_code"] == monitor["child_exit_code"] == 1
            and root["actual_tool_chunk_id"] == "9c3860" and root["recorded_training_updates"] == 0
            and monitor["supervisor_health"] == "pass" and monitor["post_exit_quiet_completed"]
            and monitor["identities_unchanged"] and monitor["latest_completed_step_seen"] is None
            and monitor["event_worker_close"]["actual_exit_code"] == 0
            and not monitor["event_worker_close"]["forced"] and not Path("/proc", str(monitor["child_pid"])).exists(),
            "Previous failed child or monitor is not settled")
    verify_inputs(plan)
    paths = [Path(__file__).resolve(), FAILED / "plan.json", FAILED / "resource-root-execution.json",
             FAILED / "resource-stage/execution.json", monitor_path]
    inputs = {"schema": "latency58-grouped-gradient-diagnostic-v1", "source_bindings":
        {**plan["source_bindings"], **{str(p): sha(p) for p in paths}},
        "parent_plan": str(FAILED / "plan.json"), "storage_budget": plan["storage_budget"],
        "previous_event_record_id": monitor["last_event_record_id"],
        "previous_monitor": {"path": str(monitor_path), "sha256": sha(monitor_path)},
        "watchdog_source": plan["watchdog_source"], "expected_final_step": 21,
        "original_resource_pass": False, "original_tolerances_changed": False,
        "budget_before": budget_snapshot(plan["storage_budget"])}
    OUT.mkdir(); write(OUT / "inputs.json", inputs)
    spec = {"schema": "gpu-watchdog-launch-finalization-v1", "cwd": str(ROOT), "environment": plan["environment"],
        "progress_path": str(OUT / "metrics.jsonl"), "expected_final_step": 21,
        "argv": [PYTHON, "-u", "-m", "research.direct.diagnose_latency58_grouped_vocal_gpu", "--child-inputs-sha256", sha(OUT / "inputs.json")]}
    write(OUT / "watchdog-spec.json", spec)
    monitor_out = Path(plan["watchdog_source"]).parent / OUT.name
    require(not monitor_out.exists(), "Preserve monitor output")
    argv = [PYTHON, plan["watchdog_source"], "--launch-spec", str(OUT / "watchdog-spec.json"),
        "--launch-spec-sha256", sha(OUT / "watchdog-spec.json"), "--output-dir", str(monitor_out),
        "--max-runtime-seconds", "900", "--poll-seconds", "2", "--query-timeout-seconds", "10",
        "--startup-grace-seconds", "180", "--progress-timeout-seconds", "120", "--finalization-timeout-seconds", "180",
        "--stop-grace-seconds", "15", "--post-exit-quiet-seconds", "10", "--max-temperature-c", "80", "--memory-headroom-mib", "4096"]
    write(OUT / "command.json", {"argv": argv, "monitor_result": str(monitor_out / "result.json"), "inputs_sha256": sha(OUT / "inputs.json")})
    print(json.dumps({"status": "prepared", "command_sha256": sha(OUT / "command.json")}), flush=True)


def child(expected_sha):
    require(Path.cwd() == ROOT and sha(OUT / "inputs.json") == expected_sha, "Diagnostic inputs changed")
    inputs = read(OUT / "inputs.json"); verify_inputs(inputs)
    plan = read(inputs["parent_plan"])
    require(all(os.environ.get(k) == v for k, v in plan["environment"].items()), "Child environment differs")
    monitor = load_source("grouped_diagnostic_monitor", inputs["watchdog_source"])
    _, events = continuity(inputs, monitor)
    write(OUT / "event-continuity.json", events)
    import torch
    from research.direct.latency58_branch_memory_checkpoint import load_model
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    model, _ = load_model(plan["parent_checkpoint"])
    require(state_sha256(model.state_dict()) == plan["initialized_model_state_sha256"]
            and not torch.cuda.is_initialized(), "CPU parent differs")
    require(torch.__version__ == plan["torch_version"] and torch.cuda.device_count() == 1
            and torch.cuda.is_bf16_supported(), "GPU arithmetic environment differs")
    torch.cuda.set_per_process_memory_fraction(.75)
    sys.path.insert(0, str(PRODUCTION))
    import train_production
    train_production.configure_determinism(plan["config"]["seed"])
    model.cuda().train().requires_grad_(True); model.training_precision = "bf16"
    began, count = time.monotonic(), 0
    with (OUT / "metrics.jsonl").open("x", buffering=1) as journal:
        def progress(phase, group, offset):
            nonlocal count
            count += 1
            row = {"step": count, "diagnostic_phase": phase, "group": group, "offset": offset, "optimizer_updates": 0}
            journal.write(json.dumps(row) + "\n"); journal.flush(); os.fsync(journal.fileno())
            print(json.dumps(row), flush=True)
        result = compare(model, progress)
        require(count == 20 and state_sha256(model.state_dict()) == plan["initialized_model_state_sha256"], "Parent or diagnostic count changed")
        verify_inputs(inputs)
        result.update(status="diagnostic_complete", source_bindings_unchanged=True, inputs_sha256=expected_sha,
            parent_weights_unchanged=True, original_resource_pass=False, original_tolerances_changed=False,
            elapsed_seconds=time.monotonic() - began, peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30,
            observed_utc=datetime.now(timezone.utc).isoformat())
        write(OUT / "child-result.json", result)
        progress("complete", "both", 0)


def run(expected_sha):
    require(sha(OUT / "command.json") == expected_sha, "Diagnostic command changed")
    command, inputs = read(OUT / "command.json"), read(OUT / "inputs.json")
    require(command["inputs_sha256"] == sha(OUT / "inputs.json"), "Inputs binding changed")
    verify_inputs(inputs)
    began = time.monotonic()
    with (OUT / "console.log").open("x") as log:
        done = subprocess.run(command["argv"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    unchanged = all(sha(p) == h for p, h in inputs["source_bindings"].items())
    write(OUT / "execution.json", {"actual_exit_code": done.returncode, "source_bindings_unchanged": unchanged,
        "command_sha256": expected_sha, "inputs_sha256": command["inputs_sha256"],
        "monitor_result": command["monitor_result"], "elapsed_seconds": time.monotonic() - began})
    require(done.returncode == 0 and unchanged, "Monitored diagnostic failed")
    terminal, result = read(command["monitor_result"]), read(OUT / "child-result.json")
    require(terminal["status"] == terminal["supervisor_health"] == "pass" and terminal["child_exit_code"] == 0
            and terminal["post_exit_quiet_completed"] and terminal["identities_unchanged"]
            and terminal["latest_completed_step_seen"] == 21 and terminal["finalization_started"]
            and terminal["event_worker_close"]["actual_exit_code"] == 0 and not terminal["event_worker_close"]["forced"]
            and result["status"] == "diagnostic_complete", "Diagnostic terminal records differ")
    write(OUT / "result.json", {"status": "diagnostic_complete", "source_bindings_unchanged": True,
        "child_result_sha256": sha(OUT / "child-result.json"), "original_resource_pass": False,
        "budget_after": budget_snapshot(inputs["storage_budget"]), "overall_goal_complete": False})
    print(json.dumps({"status": "diagnostic_complete", "original_resource_pass": False}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-only", action="store_true")
    group.add_argument("--child-inputs-sha256")
    group.add_argument("--run-command-sha256")
    args = parser.parse_args()
    if args.prepare_only:
        prepare()
    elif args.child_inputs_sha256:
        child(args.child_inputs_sha256)
    else:
        run(args.run_command_sha256)
