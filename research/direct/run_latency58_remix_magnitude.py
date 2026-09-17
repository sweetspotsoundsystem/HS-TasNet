"""Train addressed source remixing from the preserved best magnitude checkpoint."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs
from research.direct.run_latency58_fast_magnitude16 import audit_and_score


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
            "argv": [PYTHON, "-u", "-m", "research.direct.train_latency58_remix_magnitude", "--plan", str(plan_path),
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
    from research.direct.latency58_wave_spectral import VERSION
    from research.direct.latency58_remix_augmentation import check, AUGMENTATION
    from research.direct.latency58_full_magnitude_checkpoint import load_model
    from research.direct.latency58_sdr_checkpoint import require_space
    from research.direct.latency58_recorded301_data import ROOT_WEIGHTS
    require(Path.cwd() == ROOT and all(c.isalnum() or c in "-_" for c in args.name)
            and args.steps == 4000 and args.lr == 6e-5 and args.data_start == 2_700_000
            and args.seed == 20261012, "Use the fixed source-remixing recipe")
    out = PHASE / args.name
    require(not out.exists(), "Preserve completed or running trials")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    previous_path = PHASE / "full-magnitude-fast16-001/plan.json"
    previous = read(previous_path)
    verify_inputs(previous)
    functional_path = PHASE / "full-magnitude-fast16-001/functional.json"
    functional = read(functional_path)
    require(functional["objective"]["status"] == functional["full_training"]["status"]
            == functional["fast_context"]["status"] == "pass"
            and functional["objective"]["objective_version"] == VERSION
            and len(functional["full_training"]["all_neural_tensors_updated"]) == 22
            and functional["full_training"]["saved_optimizer_tensor_count"] == 22,
            "Existing objective, all-layer update or fast-context checks differ")
    long_root = PHASE / "long-magnitude-001"
    rejected = read(long_root / "result.json")
    review = read(long_root / "selection-review.json")
    require(rejected["status"] == "training_audit_and_full14_complete"
            and rejected["full_sdr_db"] == 4.082512807757332 and not rejected["target_reached"]
            and review["status"] == "not_selected"
            and read(long_root / "full14/execution.json")["actual_exit_code"] == 0,
            "Previous architecture trial is not closed and reviewed")
    parent = binding(args.parent)
    model, _ = load_model(parent)
    reference_path = args.reference.resolve()
    reference = read(reference_path)
    require(reference["status"] == "pass" and reference["results"][0]["checkpoint"] == parent
            and reference["results"][0]["aggregate"]["full_sdr_db"] == 4.114394535058717
            and state_sha256(model.state_dict()) == functional["fast_context"]["model_state_sha256"],
            "Use the preserved best short-magnitude checkpoint")
    augmentation_check = check()
    paths = [Path(__file__).resolve(), previous_path, functional_path, reference_path,
             Path(parent["path"]), long_root / "result.json", long_root / "selection-review.json",
             long_root / "full14/execution.json", PHASE / "historical-ola-resume-retirement-001/receipt.json"]
    paths.extend(ROOT / "research/direct" / name for name in (
        "latency58_remix_augmentation.py", "train_latency58_remix_magnitude.py",
        "run_latency58_fast_magnitude16.py", "evaluate_latency58_full_magnitude.py",
        "latency58_full_magnitude_checkpoint.py", "latency58_fast_context.py",
        "latency58_state_warmup.py", "check_latency58_fast_context16.py"))
    sources = {**previous["source_bindings"], **{str(p): sha(p) for p in paths}}
    config = {**previous["config"], "steps": args.steps, "seed": args.seed, "data_start": args.data_start,
              "lr": args.lr, "min_lr": args.lr / 10, "checkpoint_every": args.steps, "augmentation": AUGMENTATION}
    require(config["root_weights"] == ROOT_WEIGHTS and config["microbatch_size"] == config["batch_size"] == 16
            and previous["accumulation_steps"] == 1, "Batch geometry or recorded corpus weights changed")
    plan = {**previous, "name": args.name, "output_directory": str(out), "config": config,
            "parent_checkpoint": parent, "parent_model_state_sha256": state_sha256(model.state_dict()),
            "parent_training_updates": model.provenance["training_updates"], "reference_result": str(reference_path),
            "objective_version": VERSION, "quality_endpoints": [args.steps], "source_bindings": sources,
            "augmentation_probability": .75, "augmentation_group_size": 16,
            "augmentation_fractions": {"untouched": .25, "same_crop_transformed": .25, "cross_crop_transformed": .5},
            "cross_crop_source_selection": "four distinct nonzero cyclic shifts within each addressed batch of 16",
            "channel_swap": "independent per source in transformed examples only",
            "polarity": "shared per transformed output example", "source_gain_bounds_db": [-3., 3.],
            "augmentation_applied_before_microbatching": True, "augmentation_mapping_logged_every_update": True,
            "inference_architecture_changed": False, "torch_version": torch.__version__}
    counted = require_space(plan, 380_000_000)
    verify_inputs(plan)
    require(not torch.cuda.is_initialized(), "Preparation initialized CUDA")
    out.mkdir()
    write(out / "functional.json", {"unchanged_wave_spectral_objective": functional["objective"],
          "unchanged_full_training": functional["full_training"], "unchanged_fast_context": functional["fast_context"],
          "new_augmentation": augmentation_check})
    write(out / "preparation.json", {"status": "pass", "parent_full_sdr_db": 4.114394535058717,
          "target_full_sdr_db": 5.0, "counted_bytes_before": counted,
          "forecast_including_outside_and_run": counted + 800_000_000 + 380_000_000,
          "rationale": "The normalized trailing4096 architecture regressed on full14 to 4.082513 dB and added only 0.005161 dB in the separate 64-crop training intervention. Return to the preserved 4.114395 short-magnitude parent. Test stronger recorded-stem augmentation: quarter untouched, quarter same-crop gain/polarity/channel transformations, half recombined from four distinct other crop addresses within the batch. Keep all 22 learned tensors, the waveform/complex-STFT objective, 2 s detached warmup and 1 s scored crops. Use the earlier fast16 continuation's 4000-update cosine schedule and 6e-5 peak LR, but new seed/crop addresses and augmentation; this is an exploratory single-seed trial, not a matched causal ablation. No inference geometry changes are introduced.",
          "method_reference": "https://www.l-acoustics.com/wp-content/uploads/2024/04/real_time_demixer_2024_04_19.pdf",
          "reference_scope": "The paper describes source shuffling, channel swapping and gain augmentation; this exact grouped recipe is our own implementation.",
          "cuda_initialized": False, "source_audio_modified": False})
    plan["source_bindings"].update({str(out / name): sha(out / name) for name in ("functional.json", "preparation.json")})
    write(out / "plan.json", plan)
    return out / "plan.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="remix-magnitude-001")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--data-start", type=int, default=2_700_000)
    parser.add_argument("--parent", type=Path, default=PHASE / "full-magnitude-001/production-run/checkpoint/model.pt")
    parser.add_argument("--reference", type=Path, default=PHASE / "full-magnitude-001/full14/result.json")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepared-plan", type=Path)
    parser.add_argument("--resource-only", action="store_true")
    parser.add_argument("--after-resource", action="store_true")
    args = parser.parse_args()
    require(not (args.prepare_only and args.prepared_plan), "Choose preparation or existing plan")
    require(not (args.after_resource and (args.prepare_only or args.resource_only))
            and (not args.after_resource or args.prepared_plan is not None), "Invalid resource-stage request")
    plan_path = args.prepared_plan.resolve() if args.prepared_plan else prepare(args)
    if args.prepare_only:
        print(json.dumps({"event": "prepared", "plan": str(plan_path), "sha256": sha(plan_path)}), flush=True)
        return
    resource = (plan_path.parent / "resource-stage/execution.json" if args.after_resource else
                launch(plan_path, PHASE / "long-magnitude-001/production-stage/execution.json", resource=True))
    if args.resource_only:
        print(json.dumps({"event": "resource_complete", "plan": str(plan_path), "execution": str(resource)}), flush=True)
        return
    launch(plan_path, resource, resource=False)
    audit_and_score(plan_path)


if __name__ == "__main__":
    main()
