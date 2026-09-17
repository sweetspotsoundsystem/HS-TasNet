"""Run a bounded successor independently of plugin qualification or release work."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import time

from research.direct.run_latency58_quality import PHASE, PYTHON, execute, read, require, sha, write
from research.direct.train_latency58 import ROOT, state_sha256, verify_inputs
from research.direct.latency58_cleanup_successor_checkpoint import load_parent, require_space, validate_recipe


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def stage(plan_path, previous_execution, out):
    plan = read(plan_path)
    verify_inputs(plan)
    validate_recipe(plan)
    require_space(plan, 400_000_000)
    previous_execution = Path(previous_execution)
    prior = read(previous_execution)
    previous_path = Path(prior["monitor_result"])
    previous = read(previous_path)
    require(prior["actual_exit_code"] == 0 and prior["source_bindings_unchanged"]
            and previous["status"] == previous["supervisor_health"] == "pass"
            and previous["child_exit_code"] == 0 and previous["post_exit_quiet_completed"],
            "Previous GPU stage did not close successfully")
    require(not out.exists() and not Path(plan["run_dir"]).exists(), "Preserve existing stages")
    out.mkdir()
    stop = 2 if plan["resource_only"] else 250
    stage_data = {"schema": "latency58-cleanup-successor-stage-v1", "plan_sha256": sha(plan_path),
                  "start_step": 0, "stop_step": stop, "output_directory": str(out),
                  "previous_event_record_id": previous["last_event_record_id"],
                  "previous_monitor": binding(previous_path),
                  "source_bindings": {str(p.resolve()): sha(p) for p in (plan_path, previous_path, previous_execution)}}
    write(out / "stage.json", stage_data)
    spec = {"schema": "gpu-watchdog-launch-v1", "cwd": str(ROOT), "environment": plan["environment"],
            "progress_path": str(Path(plan["run_dir"]) / "metrics.jsonl"),
            "argv": [PYTHON, "-u", "-m", "research.direct.train_latency58_cleanup_successor",
                     "--plan", str(plan_path), "--plan-sha256", sha(plan_path),
                     "--stage", str(out / "stage.json"), "--stage-sha256", sha(out / "stage.json")]}
    write(out / "watchdog-spec.json", spec)
    monitor_out = Path(plan["watchdog_source"]).parent / ("latency58-" + out.name)
    require(not monitor_out.exists(), "Preserve earlier watchdog evidence")
    argv = [PYTHON, plan["watchdog_source"], "--launch-spec", str(out / "watchdog-spec.json"),
            "--launch-spec-sha256", sha(out / "watchdog-spec.json"), "--output-dir", str(monitor_out),
            "--max-runtime-seconds", str(max(240, stop * 32 + 180)), "--poll-seconds", "2",
            "--query-timeout-seconds", "10", "--startup-grace-seconds", "120",
            "--progress-timeout-seconds", "60", "--stop-grace-seconds", "15",
            "--post-exit-quiet-seconds", "10", "--max-temperature-c", "80", "--memory-headroom-mib", "4096"]
    write(out / "command.json", {"argv": argv, "plan_sha256": sha(plan_path)})
    print(json.dumps({"event": "launch", "stop_step": stop, "output": str(out)}), flush=True)
    began = time.monotonic()
    with (out / "console.log").open("x") as stream:
        child = subprocess.run(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    unchanged = all(sha(p) == h for p, h in {**plan["source_bindings"], **stage_data["source_bindings"]}.items())
    receipt = {"actual_exit_code": child.returncode, "elapsed_seconds": time.monotonic() - began,
               "source_bindings_unchanged": unchanged, "plan_sha256": sha(plan_path),
               "stage_sha256": sha(out / "stage.json"), "monitor_result": str(monitor_out / "result.json")}
    write(out / "execution.json", receipt)
    print(json.dumps(receipt), flush=True)
    require(child.returncode == 0 and unchanged, "Training failed; retained watchdog evidence governs next action")
    monitor = read(monitor_out / "result.json")
    require(monitor["status"] == monitor["supervisor_health"] == "pass" and monitor["child_exit_code"] == 0
            and monitor["post_exit_quiet_completed"], "GPU stage health failed")
    require(read(Path(plan["run_dir"]) / "status.json")["step"] == stop, "Training endpoint differs")
    return out / "execution.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--previous-execution", type=Path, required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.resource_plan) == args.plan_sha256, "Frozen plan differs")
    plan = read(args.resource_plan)
    verify_inputs(plan)
    validate_recipe(plan)
    require_space(plan, 400_000_000)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = load_parent(plan)
    require(state_sha256(model.state_dict()) == plan["initialized_model_state_sha256"], "Parent tensor identity differs")
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        audio = torch.linspace(-.1, .1, 1024).reshape(1, 2, 512)
        a, b = model.render(audio), model.render(audio)
        require(torch.equal(a.deployed, b.deployed) and bool(torch.isfinite(a.deployed).all())
                and float((a.deployed.sum(dim=1) - a.delayed_mixture).abs().max()) < 1e-6,
                "Selected parent reset or mixture reconstruction differs")
    del a, b, model
    prep = args.resource_plan.parent
    write(prep / "parent-check.json", {"status": "pass", "model_state_sha256": plan["initialized_model_state_sha256"],
          "source_bindings_unchanged": True, "cuda_initialized": torch.cuda.is_initialized(),
          "exact_reset_replay": True, "training_updates_executed": 0})
    resource_execution = stage(args.resource_plan, args.previous_execution, PHASE / "cleanup-successor-resource-001")
    resource = Path(plan["run_dir"]) / "resource.json"
    result = read(resource)
    require(result["status"] == "pass" and result["training_updates_executed"] == 2
            and result["source_bindings_unchanged"] and not result["checkpoint_written"], "Resource rehearsal failed")
    production = copy.deepcopy(plan)
    production.update(resource_only=False, run_dir=str(PHASE / "cleanup-successor-b16-micro4-lr1e5-250"),
                      resource_plan=binding(args.resource_plan), full_resource=binding(resource),
                      full_resource_execution=binding(resource_execution))
    production["source_bindings"].update({str(p): sha(p) for p in (args.resource_plan, resource, resource_execution)})
    path = prep / "training-plan.json"
    write(path, production)
    execution = stage(path, resource_execution, PHASE / "cleanup-successor-to-000250-001")
    run = Path(production["run_dir"])
    generation = Path(read(run / "latest.json")["generation"])
    bindings = {**production["source_bindings"], str(path): sha(path),
                **{str(p): sha(p) for p in generation.iterdir() if p.is_file()}}
    out = execution.parent
    execute([PYTHON, "-u", "-m", "research.direct.audit_latency58_cleanup_successor", "--plan", str(path),
             "--plan-sha256", sha(path), "--generation", str(generation), "--output", str(out / "audit.json")],
            out, "audit", 240, bindings, {"plan_sha256": sha(path)})
    write(run / "audit-latest.json", {"step": 250, "generation_receipt_sha256": sha(generation / "receipt.json"),
          "audit": binding(out / "audit.json"), "execution": binding(out / "audit-execution.json")})
    print(json.dumps({"event": "successor_training_and_audit_pass", "generation": str(generation)}), flush=True)


if __name__ == "__main__":
    main()
