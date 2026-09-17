"""Execute two-group attention training through monitored rehearsal, saved audit and full14 scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from research.direct.run_latency58_quality import ROOT, PYTHON, read, require, sha, write, execute
from research.direct.train_latency58 import verify_inputs


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def launch(plan_path, previous_execution_path, *, resource):
    plan = read(plan_path)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-attention-grouped-training-plan-v1", "Wrong temporal-attention training plan")
    from research.direct.latency58_sdr_checkpoint import require_space
    require_space(plan, 380_000_000)
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
        parity_path = root / "resource-run/temporal_attention-context-gpu-parity.json"
        parity = read(parity_path)
        require(result["status"] == "pass" and result["updates"] == 2 and not result["checkpoint_written"]
                and result["source_bindings_unchanged"] and result["plan_sha256"] == sha(plan_path)
                and parity["status"] == "pass" and parity["temporary_nonzero_expansion_restored"]
                and parity["original_model_state_sha256"] == plan["initialized_model_state_sha256"]
                and len(parity["all_30_gradients"]) == 30
                and result["optimizer_group_count"] == 2
                and result["optimizer_configuration"]["schema"] == plan["optimizer_schema"]
                and result["optimizer_configuration"]["attention_lr_multiplier"] == plan["attention_lr_multiplier"],
                "Resource rehearsal did not pass")
        stage["resource_result"] = binding(path)
        stage["source_bindings"][str(path)] = sha(path)
        stage["source_bindings"][str(parity_path)] = sha(parity_path)
    write(out / "stage.json", stage)
    spec = {"schema": "gpu-watchdog-launch-v1", "cwd": str(ROOT), "environment": plan["environment"],
            "progress_path": str(run / "metrics.jsonl"),
            "argv": [PYTHON, "-u", "-m", "research.direct.train_latency58_attention_grouped", "--plan", str(plan_path),
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


def audit_and_score(plan_path):
    import torch
    from research.direct.latency58_attention_grouped_checkpoint import audit_saved
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    plan = read(plan_path)
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    training = read(out / "production-run/result.json")
    execution = read(out / "production-stage/execution.json")
    require(training["status"] == "pass" and training["checkpoint_written"]
            and execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"],
            "Require completed temporal-attention training before scoring")
    checkpoint = training["checkpoint"]
    audit = audit_saved(checkpoint, plan, sha(plan_path))
    require(audit["step"] == training["updates"]
            and audit["model_state_sha256"] == training["final_model_state_sha256"],
            "Saved temporal_attention model differs from the training endpoint")
    verify_inputs(plan)
    write(out / "checkpoint-audit.json", {**audit, "source_bindings_unchanged": True})
    quality = out / "full14"
    quality.mkdir()
    bindings = {**plan["source_bindings"], str(plan_path): sha(plan_path), checkpoint["path"]: checkpoint["sha256"],
                **{str(out / name): sha(out / name) for name in (
                    "checkpoint-audit.json", "production-run/result.json", "production-stage/execution.json")}}
    quality_plan = {"schema": "latency58-temporal-attention-full14-plan-v1", "label": plan["name"], "checkpoint": checkpoint,
                    "reference_result": plan["reference_result"], "workers": 2, "track_indices": list(range(14)),
                    "source_bindings": bindings, "output_directory": str(quality)}
    write(quality / "plan.json", quality_plan)
    execute([PYTHON, "-u", "-m", "research.direct.evaluate_latency58_temporal_attention", "--plan", str(quality / "plan.json"),
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
    parser.add_argument("--prepared-plan", type=Path, required=True)
    parser.add_argument("--previous-execution", type=Path)
    parser.add_argument("--resource-only", action="store_true")
    parser.add_argument("--after-resource", action="store_true")
    args = parser.parse_args()
    require(not (args.resource_only and args.after_resource)
            and ((args.previous_execution is None) == args.after_resource),
            "Provide the previous execution for a new resource stage only")
    plan_path = args.prepared_plan.resolve(strict=True)
    resource = (plan_path.parent / "resource-stage/execution.json" if args.after_resource else
                launch(plan_path, args.previous_execution.resolve(strict=True), resource=True))
    if args.resource_only:
        print(json.dumps({"event": "resource_complete", "plan": str(plan_path), "execution": str(resource)}), flush=True)
        return
    launch(plan_path, resource, resource=False)
    audit_and_score(plan_path)


if __name__ == "__main__":
    main()
