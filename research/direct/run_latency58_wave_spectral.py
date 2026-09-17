"""Prepare and score full-model waveform/spectral reconstruction with recorded-source gain views."""
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
    require_space(plan, 350_000_000)
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
            "argv": [PYTHON, "-u", "-m", "research.direct.train_latency58_wave_spectral", "--plan", str(plan_path),
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
    from research.direct.latency58_wave_spectral import check, VERSION, AUGMENTATION
    from research.direct.latency58_direct_sdr_checkpoint import load_model
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and all(c.isalnum() or c in "-_" for c in args.name)
            and args.steps > 2 and args.lr > 0 and args.data_start >= 1_000_000, "Invalid pilot parameters")
    out = PHASE / args.name
    require(not out.exists(), "Preserve completed or running pilots")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    functional = check()
    template = read(PHASE / "latency58-reduced-teacher-001/training-plan.json")
    from research.direct.latency58_musdb_sdr_data import selection_contract
    selection = selection_contract()
    parent = binding(args.parent)
    model, _ = load_model(parent)
    reference_path = args.reference.resolve()
    reference = read(reference_path)
    require(reference["status"] == "pass" and reference["results"][0]["checkpoint"]["sha256"] == parent["sha256"],
            "Parent must have a completed full14 score")
    parent_qualification = read(PHASE / "c204-residual-model-001/qualification.json")
    paths = [Path(__file__).resolve(), ROOT / "research/direct/latency58_direct_sdr.py",
             ROOT / "research/direct/latency58_direct_sdr_checkpoint.py", ROOT / "research/direct/train_latency58_wave_spectral.py",
             ROOT / "research/direct/latency58_wave_spectral.py", ROOT / "research/direct/reserve_latency58_wave_spectral.py",
             PHASE / "wave-spectral-storage-001/receipt.json",
             ROOT / "research/direct/evaluate_latency58_direct_sdr.py", ROOT / "research/direct/reserve_latency58_direct_sdr.py",
             Path(parent["path"]), reference_path, PHASE / "direct-sdr-storage-001/receipt.json", PHASE / "direct-sdr-storage-002/receipt.json",
             ROOT / "research/direct/latency58_musdb_sdr_data.py", ROOT / "research/direct/reserve_latency58_direct_sdr_followup.py",
             PRODUCTION / "train_production.py", PRODUCTION / "full_config.json", PRODUCTION / "manifests/combined.manifest.json",
             Path(template["watchdog_source"]), ROOT / "research/direct/latency58_sdr_context.py",
             ROOT / "research/experiment.py", ROOT / "research/direct/latency58_asymmetric.py",
             ROOT / "research/direct/latency58_gpu.py", ROOT / "research/direct/latency58.py"]
    sources = {**parent_qualification["source_bindings"], **{str(p): sha(p) for p in paths}}
    config = {**template["config"], "steps": args.steps, "seed": args.seed, "data_start": args.data_start,
              "crop_samples": 176384, "lr": args.lr, "min_lr": args.lr / 10, "warmup": 25,
              "microbatch_size": 4, "checkpoint_every": args.steps,
              "source_corpus_root_weights": template["config"]["root_weights"],
              "root_weights": {"musdb18hq_train": 1.0}, "augmentation": AUGMENTATION}
    plan = {"schema": "latency58-direct-sdr-plan-v1", "name": args.name, "output_directory": str(out), "config": config,
            "parent_checkpoint": parent, "parent_model_state_sha256": state_sha256(model.state_dict()),
            "parent_training_updates": model.provenance["training_updates"], "reference_result": str(reference_path),
            "objective_version": VERSION, "teacher_used": False, "fixed_residual_share": 1 / 16,
            "online_teacher_used": False, "teacher_generated_targets_in_current_stage": False,
            "training_selection": selection, "rendered_training_examples_always_ordinary_mixture": False,
            "augmentation_probability": .5, "source_gain_bounds_db": [-3., 3.], "spectral_fft_sizes": [512, 1024, 2048],
            "warmup_samples": 88064, "scored_samples": 88320, "accumulation_steps": 4,
            "target_full_sdr_db": 5.0, "quality_endpoints": [args.steps], "automatic_plugin_replacement": False,
            "source_bindings": sources, "environment": template["environment"], "torch_version": torch.__version__,
            "precision_policy": template["precision_policy"], "watchdog_source": template["watchdog_source"],
            "manifest_sha256": template["manifest_sha256"], "counted_roots": template["counted_roots"],
            "stop_counted_bytes": 79_200_000_000, "outside_roots_reservation_bytes": 800_000_000}
    counted = require_space(plan, 350_000_000)
    verify_inputs(plan)
    require(not torch.cuda.is_initialized(), "Preparation unexpectedly initialized CUDA")
    out.mkdir()
    write(out / "functional.json", functional)
    write(out / "preparation.json", {"status": "pass", "parent_model_state_sha256": plan["parent_model_state_sha256"],
          "parent_full_sdr_db": reference["results"][0]["aggregate"]["full_sdr_db"], "target_full_sdr_db": 5.0,
          "counted_bytes_before": counted, "forecast_including_outside_and_pilot": counted + 800_000_000 + 350_000_000,
          "rationale": "Full-model waveform and complex spectral reconstruction from the preserved best checkpoint: half recorded mixtures, half same-song source gains within +/-3 dB and shared polarity; no teacher or validation/test fitting. Loss changes add no inference work or audio buffering.",
          "cuda_initialized": False, "training_selection": selection})
    plan["source_bindings"].update({str(out / name): sha(out / name) for name in ("functional.json", "preparation.json")})
    write(out / "plan.json", plan)
    return out / "plan.json"


def audit_and_score(plan_path):
    import torch
    from research.direct.latency58_direct_sdr_checkpoint import load_model
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
    execute([PYTHON, "-u", "-m", "research.direct.evaluate_latency58_direct_sdr", "--plan", str(quality / "plan.json"),
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
    parser.add_argument("--name", default="wave-spectral-001")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--data-start", type=int, default=1_700_000)
    parser.add_argument("--parent", type=Path, default=PHASE / "c204-residual-model-001/model.pt")
    parser.add_argument("--reference", type=Path, default=PHASE / "c204-residual-model-full14-001/result.json")
    args = parser.parse_args()
    plan_path = prepare(args)
    resource = launch(plan_path, PHASE / "magnitude-sdr-001/production-stage/execution.json", resource=True)
    launch(plan_path, resource, resource=False)
    audit_and_score(plan_path)


if __name__ == "__main__":
    main()
