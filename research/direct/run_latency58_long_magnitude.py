"""Train and score causal trailing4096 normalized magnitude features."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, read, require, sha, write, execute
from research.direct.train_latency58 import PRODUCTION, state_sha256, verify_inputs


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def launch(plan_path, previous_execution_path, *, resource):
    plan = read(plan_path)
    verify_inputs(plan)
    from research.direct.latency58_sdr_checkpoint import require_space
    require_space(plan, 370_000_000)
    previous_execution = read(previous_execution_path)
    previous_path = Path(previous_execution["monitor_result"])
    previous = read(previous_path)
    require(previous_execution["actual_exit_code"] == 0 and previous_execution["source_bindings_unchanged"]
            and previous["status"] == previous["supervisor_health"] == "pass"
            and previous["child_exit_code"] == 0 and previous["post_exit_quiet_completed"], "Previous GPU monitor failed")
    root = Path(plan["output_directory"])
    name = "resource" if resource else "production"
    out, run = root / (name + "-stage"), root / (name + "-run")
    require(not out.exists() and not run.exists(), "Preserve existing monitored stages")
    out.mkdir()
    stop = 2 if resource else plan["config"]["steps"]
    stage = {"schema": "latency58-direct-sdr-stage-v1", "plan_sha256": sha(plan_path), "resource_only": resource,
             "stop_step": stop, "run_directory": str(run), "output_directory": str(out),
             "previous_event_record_id": previous["last_event_record_id"], "previous_monitor": binding(previous_path),
             "source_bindings": {str(p): sha(p) for p in (plan_path, previous_path, previous_execution_path)}}
    if not resource:
        path = root / "resource-run/result.json"
        result = read(path)
        require(result["status"] == "pass" and result["updates"] == 2 and not result["checkpoint_written"],
                "Resource rehearsal did not pass")
        stage["resource_result"] = binding(path)
        stage["source_bindings"][str(path)] = sha(path)
    write(out / "stage.json", stage)
    spec = {"schema": "gpu-watchdog-launch-v1", "cwd": str(ROOT), "environment": plan["environment"],
            "progress_path": str(run / "metrics.jsonl"),
            "argv": [PYTHON, "-u", "-m", "research.direct.train_latency58_long_magnitude", "--plan", str(plan_path),
                     "--plan-sha256", sha(plan_path), "--stage", str(out / "stage.json"),
                     "--stage-sha256", sha(out / "stage.json")]}
    write(out / "watchdog-spec.json", spec)
    monitor_out = Path(plan["watchdog_source"]).parent / (root.name + "-" + name)
    require(not monitor_out.exists(), "Preserve monitor output")
    argv = [PYTHON, plan["watchdog_source"], "--launch-spec", str(out / "watchdog-spec.json"),
            "--launch-spec-sha256", sha(out / "watchdog-spec.json"), "--output-dir", str(monitor_out),
            "--max-runtime-seconds", str(max(300, stop * 32 + 180)), "--poll-seconds", "2",
            "--query-timeout-seconds", "10", "--startup-grace-seconds", "120", "--progress-timeout-seconds", "60",
            "--stop-grace-seconds", "15", "--post-exit-quiet-seconds", "10", "--max-temperature-c", "80",
            "--memory-headroom-mib", "4096"]
    write(out / "command.json", {"argv": argv})
    print(json.dumps({"event": "launch", "stage": name, "updates": stop}), flush=True)
    began = time.monotonic()
    with (out / "console.log").open("x") as log:
        child = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    unchanged = all(sha(p) == h for p, h in {**plan["source_bindings"], **stage["source_bindings"]}.items())
    write(out / "execution.json", {"actual_exit_code": child.returncode, "elapsed_seconds": time.monotonic() - began,
          "source_bindings_unchanged": unchanged, "plan_sha256": sha(plan_path), "monitor_result": str(monitor_out / "result.json")})
    require(child.returncode == 0 and unchanged, "Monitored direct-SDR stage failed")
    terminal = read(monitor_out / "result.json")
    require(terminal["status"] == terminal["supervisor_health"] == "pass" and terminal["child_exit_code"] == 0
            and terminal["post_exit_quiet_completed"], "GPU supervisor did not close cleanly")
    print(json.dumps({"event": "stage_pass", "stage": name, "updates": stop}), flush=True)
    return out / "execution.json"


def prepare(args):
    import torch
    from research.direct.latency58_wave_spectral import VERSION, AUGMENTATION
    from research.direct.latency58_long_magnitude_checkpoint import load_model
    from research.direct.latency58_sdr_checkpoint import require_space
    from research.direct.latency58_recorded301_data import selection_contract, ROOT_WEIGHTS
    require(Path.cwd() == ROOT and all(c.isalnum() or c in "-_" for c in args.name)
            and args.steps == 2000 and args.lr == 3e-5 and args.data_start == 2_500_000
            and args.seed == 20261011, "Use the fixed long-feature experiment recipe")
    out = PHASE / args.name
    require(not out.exists(), "Preserve completed or running experiments")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    previous_path = PHASE / "full-magnitude-sdr-001/plan.json"
    previous = read(previous_path)
    verify_inputs(previous)
    cpu = PHASE / "long-magnitude-cpu-001"
    cpu_plan, cpu_result = read(cpu / "plan.json"), read(cpu / "result.json")
    verify_inputs(cpu_plan)
    require(cpu_plan["actual_exit_codes"] == [0, 0]
            and cpu_result["full_training"]["status"] == cpu_result["fast_context"]["status"] == "pass"
            and cpu_result["full_training"]["all_neural_tensor_count_updated"] == 23
            and cpu_result["full_training"]["zero_projection_parent_bit_exact"], "CPU functional checks incomplete")
    objective_path = PHASE / "full-magnitude-001/functional.json"
    objective_check = read(objective_path)["objective"]
    require(objective_check["status"] == "pass" and objective_check["objective_version"] == VERSION,
            "Existing unchanged waveform/spectral objective check differs")
    parent = binding(args.parent)
    model, _ = load_model(parent)
    reference_path = args.reference.resolve()
    reference = read(reference_path)
    require(reference["status"] == "pass" and reference["results"][0]["checkpoint"] == parent
            and reference["results"][0]["aggregate"]["full_sdr_db"] == 4.114394535058717,
            "Parent must be the saved, scored 4.114 magnitude checkpoint")
    selection = selection_contract()
    paths = [Path(__file__).resolve(), previous_path, objective_path, cpu / "plan.json", cpu / "result.json",
             Path(parent["path"]), reference_path,
             PHASE / "failed-full-optimizers-retirement-001/receipt.json"]
    paths.extend(ROOT / "research/direct" / name for name in (
        "latency58_long_magnitude.py", "latency58_long_magnitude_checkpoint.py", "latency58_long_magnitude_context.py",
        "check_latency58_long_magnitude.py", "check_latency58_long_magnitude_context.py", "check_latency58_long_magnitude_gpu.py",
        "train_latency58_long_magnitude.py", "evaluate_latency58_long_magnitude.py", "latency58_full_magnitude_checkpoint.py",
        "latency58_wave_spectral.py", "latency58_evaluate.py", "compare.py", "report_latency58_vocal_focus.py",
        "run_latency58_quality.py", "train_latency58.py", "latency58_sdr_checkpoint.py"))
    sources = {**previous["source_bindings"], **cpu_plan["source_bindings"], **{str(p): sha(p) for p in paths}}
    config = {**previous["config"], "steps": args.steps, "seed": args.seed, "data_start": args.data_start,
              "lr": args.lr, "min_lr": args.lr / 10, "checkpoint_every": args.steps, "augmentation": AUGMENTATION}
    require(config["root_weights"] == ROOT_WEIGHTS and config["microbatch_size"] == config["batch_size"] == 16,
            "Recorded corpus weights or batch geometry changed")
    plan = {**previous, "name": args.name, "output_directory": str(out), "config": config,
            "parent_checkpoint": parent, "parent_model_state_sha256": state_sha256(model.state_dict()),
            "parent_training_updates": model.provenance["training_updates"], "reference_result": str(reference_path),
            "objective_version": VERSION, "training_selection": selection,
            "spectral_fft_sizes": [512, 1024, 2048], "supervision": "relative waveform L1 plus 0.25 complex STFT and 0.25 raw L1",
            "training_context_implementation": "long-feature state-only warmup plus unchanged scored native render",
            "architecture": model.architecture_metadata,
            "cpu_fast_context_bit_exact": True, "gpu_fast_context_bit_exact_required_before_updates": True,
            "additional_long_feature_parameters": 369000, "training_neural_tensor_count": 23,
            "quality_endpoints": [args.steps], "source_bindings": sources, "torch_version": torch.__version__}
    counted = require_space(plan, 380_000_000)
    verify_inputs(plan)
    require(not torch.cuda.is_initialized(), "Preparation unexpectedly initialized CUDA")
    out.mkdir()
    write(out / "functional.json", {"objective": objective_check, **cpu_result})
    write(out / "preparation.json", {"status": "pass", "parent_model_state_sha256": plan["parent_model_state_sha256"],
          "parent_full_sdr_db": 4.114394535058717, "target_full_sdr_db": 5.0,
          "counted_bytes_before": counted, "forecast_including_outside_and_run": counted + 800_000_000 + 380_000_000,
          "rationale": "Test longer phase-invariant input features from the preserved best checkpoint: trailing Hann4096 power, individual low-frequency bins, groups of eight upper-frequency bins, frame-local scale normalization and log1p, with a zero-initialized 738-to-500 projection. Train all 23 neural tensors for 2000 updates at peak Adam LR 3e-5 using the original waveform/complex-STFT recipe and fresh recorded301 crops. The earlier C191 raw-magnitude2048 experiment was negative; this branch uses a different parent, normalized pooled4096 features and unchanged hop128 synthesis. The latest direct-SDR-loss trial regressed to 4.058454 and is not the parent. The additional state holds received past input, with no new future callbacks. Native export and M4 qualification are required after any quality success.",
          "saved_generation_bytes_measured_in_cpu_check": 344562136,
          "cuda_initialized": False, "training_selection": selection})
    plan["source_bindings"].update({str(out / name): sha(out / name) for name in ("functional.json", "preparation.json")})
    write(out / "plan.json", plan)
    return out / "plan.json"

def audit_and_score(plan_path):
    import torch
    from research.direct.latency58_long_magnitude_checkpoint import load_model
    plan = read(plan_path)
    out = Path(plan["output_directory"])
    training = read(out / "production-run/result.json")
    checkpoint = training["checkpoint"]
    model, payload = load_model(checkpoint)
    require(training["status"] == "pass" and training["updates"] == plan["config"]["steps"] == payload["step"]
            and training["final_model_state_sha256"] == state_sha256(model.state_dict())
            and payload["plan_sha256"] == sha(plan_path), "Saved inference endpoint differs")
    optimizer_path = Path(checkpoint["path"]).with_name("optimizer.pt")
    resume = torch.load(optimizer_path, map_location="cpu", weights_only=True)
    require(resume["step"] == payload["step"] and resume["model_state_sha256"] == payload["model_state_sha256"]
            and resume["plan_sha256"] == sha(plan_path), "Saved optimizer belongs to another endpoint")
    states = resume["optimizer"]["state"]
    parameters = list(model.parameters())
    require(len(states) == len(parameters) and set(states) == set(range(len(parameters))), "Saved Adam inventory differs")
    for index, state in states.items():
        require(state["step"].item() == payload["step"] and state["exp_avg"].shape == parameters[index].shape
                and state["exp_avg_sq"].shape == parameters[index].shape
                and bool(torch.isfinite(state["exp_avg"]).all()) and bool(torch.isfinite(state["exp_avg_sq"]).all())
                and bool((state["exp_avg_sq"] >= 0).all()), "Saved Adam moments are invalid")
    with torch.inference_mode():
        signal = torch.linspace(-.1, .1, 2 * 8 * 128).reshape(1, 2, 8 * 128)
        first = model.render(signal)
        second = model.render(signal)
        require(torch.equal(first.deployed, second.deployed), "Saved reset replay differs")
        closure = float((first.deployed.sum(dim=1) - first.delayed_mixture).abs().max())
        require(closure < 1e-6 and model.algorithmic_latency_samples == 256, "Saved model closure or geometry differs")
    del model, resume, states, parameters
    verify_inputs(plan)
    write(out / "checkpoint-audit.json", {"status": "pass", "step": payload["step"], "checkpoint": checkpoint,
          "model_state_sha256": payload["model_state_sha256"], "optimizer_sha256": sha(optimizer_path),
          "fixed_buffers_sha256": payload["fixed_buffers_sha256"], "exact_reset_replay": True,
          "closure_max_abs": closure, "algorithmic_latency_samples": 256, "source_bindings_unchanged": True})
    quality = out / "full14"
    quality.mkdir()
    bindings = {**plan["source_bindings"], str(plan_path): sha(plan_path), checkpoint["path"]: checkpoint["sha256"],
                **{str(out / name): sha(out / name) for name in ("checkpoint-audit.json", "production-run/result.json", "production-stage/execution.json")}}
    quality_plan = {"schema": "latency58-direct-sdr-full14-plan-v1", "label": plan["name"], "checkpoint": checkpoint,
                    "reference_result": plan["reference_result"], "workers": 2, "track_indices": list(range(14)),
                    "source_bindings": bindings, "output_directory": str(quality)}
    write(quality / "plan.json", quality_plan)
    execute([PYTHON, "-u", "-m", "research.direct.evaluate_latency58_long_magnitude", "--plan", str(quality / "plan.json"),
             "--plan-sha256", sha(quality / "plan.json")], quality, "evaluation", 1800, bindings,
            {"plan_sha256": sha(quality / "plan.json")})
    report = read(quality / "result.json")
    write(out / "result.json", {"status": "training_audit_and_full14_complete", "checkpoint": checkpoint,
          "full_sdr_db": report["results"][0]["aggregate"]["full_sdr_db"], "target_full_sdr_db": 5.0,
          "target_reached": report["target_reached"], "quality_result": binding(quality / "result.json"),
          "plugin_replaced": False})
    print(json.dumps(read(out / "result.json")), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="long-magnitude-001")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--data-start", type=int, default=2_500_000)
    parser.add_argument("--parent", type=Path, default=PHASE / "full-magnitude-001/production-run/checkpoint/model.pt")
    parser.add_argument("--reference", type=Path, default=PHASE / "full-magnitude-001/full14/result.json")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepared-plan", type=Path)
    parser.add_argument("--resource-only", action="store_true")
    parser.add_argument("--after-resource", action="store_true")
    args = parser.parse_args()
    require(not (args.prepare_only and args.prepared_plan), "Choose preparation or an existing frozen plan")
    require(not (args.after_resource and (args.prepare_only or args.resource_only))
            and (not args.after_resource or args.prepared_plan is not None), "Invalid resource-stage request")
    plan_path = args.prepared_plan.resolve() if args.prepared_plan else prepare(args)
    if args.prepare_only:
        print(json.dumps({"event": "prepared", "plan": str(plan_path), "sha256": sha(plan_path)}), flush=True)
        return
    resource = (plan_path.parent / "resource-stage/execution.json" if args.after_resource else
                launch(plan_path, PHASE / "full-magnitude-sdr-001/production-stage/execution.json", resource=True))
    if args.resource_only:
        print(json.dumps({"event": "resource_complete", "plan": str(plan_path), "execution": str(resource)}), flush=True)
        return
    launch(plan_path, resource, resource=False)
    audit_and_score(plan_path)


if __name__ == "__main__":
    main()
