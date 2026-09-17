"""Run a monitored resource rehearsal or an audited soft-capped SDR stage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from research.direct.run_latency58_quality import PYTHON, execute, write
from research.direct.train_latency58 import ROOT, load_source, read, require, sha, verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--stop-step", type=int, choices=(2, 250, 500), required=True)
    parser.add_argument("--previous-monitor", type=Path, required=True)
    parser.add_argument("--previous-execution", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Plan or cwd differs")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-sdr-softcap-training-v1", "Unknown training plan")
    from research.direct.latency58_sdr_softcap_checkpoint import read_generation, require_space
    require_space(plan, 10_000_000 if plan["resource_only"] else 350_000_000)
    out, run = args.output_directory.absolute(), Path(plan["run_dir"])
    require(out.is_relative_to(ROOT / "research/direct/runs/latency58") and not out.exists(), "Use a fresh stage directory")
    previous, execution = read(args.previous_monitor), read(args.previous_execution)
    require(previous["status"] == previous["supervisor_health"] == "pass" and previous["child_exit_code"] == 0
            and previous["post_exit_quiet_completed"] and execution["actual_exit_code"] == 0
            and execution["source_bindings_unchanged"], "Previous GPU stage did not finish successfully")
    proof = read(plan["functional_proof"]["path"])
    proof_execution = read(plan["functional_execution"]["path"])
    require(proof["status"] == "pass" and proof["source_bindings_unchanged"]
            and proof["version"] == plan["sdr_softcap_version"] and proof["auxiliary_weight"] == plan["sdr_softcap_weight"]
            and proof["error_ratio_floor"] == plan["sdr_softcap_error_ratio_floor"] == 0.01
            and proof_execution["actual_exit_code"] == 0 and not proof_execution["timed_out"]
            and proof_execution["source_bindings_unchanged"]
            and all(sha(p) == v for p, v in proof["source_bindings"].items()),
            "Functional auxiliary qualification differs")
    if plan["resource_only"]:
        require(args.stop_step == 2 and not run.exists(), "Resource rehearsal must be fresh and exactly two updates")
    else:
        resource = read(plan["full_resource"]["path"])
        resource_execution = read(plan["full_resource_execution"]["path"])
        resource_monitor = read(resource_execution["monitor_result"])
        resource_plan = read(plan["resource_plan"]["path"])
        shared = ("config", "parent", "warmup_samples", "scored_samples", "carry_state",
                  "teacher_kind", "teacher_weight", "teacher_model_state_sha256", "teacher",
                  "precision_policy", "torch_version", "environment", "sdr_softcap_weight", "sdr_softcap_version", "sdr_softcap_error_ratio_floor",
                  "helper_source", "watchdog_source", "manifest_sha256", "functional_proof", "functional_execution")
        require(resource["schema"] == "latency58-sdr-softcap-resource-result-v1" and resource["status"] == "pass"
                and resource["training_updates_executed"] == 2 and not resource["checkpoint_written"]
                and resource["source_bindings_unchanged"] and resource["all_parameter_gradients_present"]
                and resource["fixed_buffers_unchanged"] and resource["teacher_unchanged"]
                and resource["initial_model_state_sha256"] == plan["parent"]["model_state_sha256"]
                and resource["teacher_model_state_sha256"] == plan["teacher_model_state_sha256"]
                and resource["config"] == plan["config"] and resource["carry_state"]
                and resource["sdr_softcap_error_ratio_floor"] == plan["sdr_softcap_error_ratio_floor"]
                and resource_plan["resource_only"] and all(resource_plan[k] == plan[k] for k in shared)
                and resource_execution["actual_exit_code"] == 0 and resource_execution["source_bindings_unchanged"]
                and resource_execution["plan_sha256"] == resource["plan_sha256"] == sha(plan["resource_plan"]["path"])
                and resource_monitor["status"] == resource_monitor["supervisor_health"] == "pass"
                and resource_monitor["child_exit_code"] == 0 and resource_monitor["post_exit_quiet_completed"],
                "Full GPU resource proof differs from the training recipe")
    start = 0
    bind = [args.plan, args.previous_monitor, args.previous_execution]
    resume_fields = {}
    previous_generation = None
    if run.exists():
        status, pointer, audit_pointer = read(run / "status.json"), read(run / "latest.json"), read(run / "audit-latest.json")
        require(status["status"] == "paused" and not Path("/proc", str(status["pid"])).exists(),
                "Existing training arm is not terminal and paused")
        previous_generation = Path(pointer["generation"])
        receipt = read_generation(previous_generation, expected_plan_sha=args.plan_sha256)
        start = receipt["step"]
        require(pointer["step"] == status["step"] == audit_pointer["step"] == start
                and pointer["receipt_sha256"] == sha(previous_generation / "receipt.json")
                and audit_pointer["generation_receipt_sha256"] == pointer["receipt_sha256"],
                "Resume pointers disagree")
        resume_fields = {"resume_generation": str(previous_generation),
                         "resume_audit": audit_pointer["audit"], "resume_audit_execution": audit_pointer["execution"]}
        bind += [previous_generation / name for name in ("model.pt", "optimizer.pt", "rng.pt", "metrics.jsonl", "receipt.json")]
        bind += [Path(audit_pointer[k]["path"]) for k in ("audit", "execution")]
    require(start < args.stop_step <= plan["config"]["steps"], "Stage must advance the existing training arm")
    stage = {"schema": "latency58-sdr-softcap-stage-v1", "plan_sha256": args.plan_sha256,
             "start_step": start, "stop_step": args.stop_step, "output_directory": str(out),
             "previous_event_record_id": previous["last_event_record_id"],
             "previous_monitor": {"path": str(args.previous_monitor.resolve()), "sha256": sha(args.previous_monitor)},
             "source_bindings": {str(p.resolve()): sha(p) for p in bind}, **resume_fields}
    monitor_out = Path(plan["watchdog_source"]).parent / ("latency58-" + out.name)
    require(not monitor_out.exists(), "Preserve previous GPU monitor output")
    out.mkdir()
    write(out / "stage.json", stage)
    spec = {"schema": "gpu-watchdog-launch-v1", "cwd": str(ROOT), "environment": plan["environment"],
            "progress_path": str(run / "metrics.jsonl"),
            "argv": [PYTHON, "-u", "-m", "research.direct.train_latency58_sdr_softcap", "--plan", str(args.plan.resolve()),
                     "--plan-sha256", args.plan_sha256, "--stage", str(out / "stage.json"),
                     "--stage-sha256", sha(out / "stage.json")]}
    write(out / "watchdog-spec.json", spec)
    timeout = max(240, (args.stop_step - start) * 8 + 180)
    argv = [PYTHON, plan["watchdog_source"], "--launch-spec", str(out / "watchdog-spec.json"),
            "--launch-spec-sha256", sha(out / "watchdog-spec.json"), "--output-dir", str(monitor_out),
            "--max-runtime-seconds", str(timeout), "--poll-seconds", "2", "--query-timeout-seconds", "10",
            "--startup-grace-seconds", "120", "--progress-timeout-seconds", "60", "--stop-grace-seconds", "15",
            "--post-exit-quiet-seconds", "10", "--max-temperature-c", "80", "--memory-headroom-mib", "4096"]
    write(out / "command.json", {"argv": argv, "cwd": str(ROOT), "plan_sha256": args.plan_sha256,
                                 "stage_sha256": sha(out / "stage.json")})
    print(json.dumps({"event": "training_launch", "teacher": plan["teacher_kind"],
                      "start_step": start, "stop_step": args.stop_step, "carry_state": plan["carry_state"]}), flush=True)
    began = time.monotonic()
    with (out / "console.log").open("x") as stream:
        completed = subprocess.run(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    execution = {"actual_exit_code": completed.returncode, "elapsed_seconds": time.monotonic() - began,
                 "source_bindings_unchanged": all(sha(p) == s for p, s in
                                                  {**plan["source_bindings"], **stage["source_bindings"]}.items()),
                 "plan_sha256": args.plan_sha256, "stage_sha256": sha(out / "stage.json"),
                 "monitor_result": str(monitor_out / "result.json")}
    write(out / "execution.json", execution)
    print(json.dumps(execution), flush=True)
    require(completed.returncode == 0 and execution["source_bindings_unchanged"], "Training stage failed")
    terminal = read(monitor_out / "result.json")
    require(terminal["status"] == terminal["supervisor_health"] == "pass" and terminal["child_exit_code"] == 0
            and terminal["post_exit_quiet_completed"] and read(run / "status.json")["step"] == args.stop_step,
            "Training endpoint or monitor did not pass")
    if plan["resource_only"]:
        result_path = run / "resource.json"
        resource = read(result_path)
        require(resource["status"] == "pass" and resource["training_updates_executed"] == 2
                and resource["plan_sha256"] == args.plan_sha256
                and not resource["checkpoint_written"] and read(run / "status.json")["status"] == "resource_complete"
                and all(sha(p) == v for p, v in resource["source_bindings"].items()),
                "Resource rehearsal did not finish cleanly")
        write(out / "resource-qualification.json", {
            "schema": "latency58-sdr-softcap-resource-qualification-v1", "status": "pass",
            "resource": {"path": str(result_path), "sha256": sha(result_path)},
            "execution": {"path": str(out / "execution.json"), "sha256": sha(out / "execution.json")},
            "monitor": {"path": str(monitor_out / "result.json"), "sha256": sha(monitor_out / "result.json")},
            "actual_exit_code": 0, "post_exit_quiet_completed": True, "checkpoint_written": False,
            "source_bindings_unchanged": True, "quality_selected": False})
        print(json.dumps({"event": "resource_rehearsal_pass", "step": 2, "peak_vram_gib": resource["peak_vram_gib"]}), flush=True)
        return
    generation = Path(read(run / "latest.json")["generation"])
    bindings = {**plan["source_bindings"], str(args.plan.resolve()): args.plan_sha256}
    bindings.update({str(p): sha(p) for p in generation.iterdir() if p.is_file()})
    argv = [PYTHON, "-u", "-m", "research.direct.audit_latency58_sdr_softcap", "--plan", str(args.plan.resolve()),
            "--plan-sha256", args.plan_sha256, "--generation", str(generation), "--output", str(out / "audit.json")]
    execute(argv, out, "audit", 180, bindings, {"plan_sha256": args.plan_sha256})
    audit = read(out / "audit.json")
    require(audit["status"] == "pass" and audit["step"] == args.stop_step, "Saved-state audit failed")
    helpers = load_source("latency58_sdr_stage_helpers", plan["helper_source"])
    helpers.atomic_json(run / "audit-latest.json", {
        "step": args.stop_step, "generation_receipt_sha256": sha(generation / "receipt.json"),
        "audit": {"path": str(out / "audit.json"), "sha256": sha(out / "audit.json")},
        "execution": {"path": str(out / "audit-execution.json"), "sha256": sha(out / "audit-execution.json")}})
    if previous_generation is not None:
        # The new complete generation and independent audit are durable before
        # removing only the superseded optimizer. Every model and RNG remains.
        old_receipt = read_generation(previous_generation, expected_plan_sha=args.plan_sha256)
        old_optimizer = previous_generation / "optimizer.pt"
        retirement = {"schema": "latency58-sdr-softcap-replaced-optimizer-v1", "path": str(old_optimizer),
                      "sha256": sha(old_optimizer), "bytes": old_optimizer.stat().st_size,
                      "retained_model": str(previous_generation / "model.pt"),
                      "replacement_generation": str(generation), "replacement_audit_sha256": sha(out / "audit.json")}
        write(out / "retire-previous-optimizer-intent.json", retirement)
        require(retirement["sha256"] == old_receipt["files"]["optimizer.pt"]["sha256"], "Old optimizer changed")
        old_optimizer.unlink()
        require(sha(previous_generation / "model.pt") == old_receipt["files"]["model.pt"]["sha256"], "Old model changed")
        write(out / "retire-previous-optimizer-receipt.json", {"status": "complete", "freed_bytes": retirement["bytes"],
                                                              "intent_sha256": sha(out / "retire-previous-optimizer-intent.json"),
                                                              "retained_model_unchanged": True})
    print(json.dumps({"event": "training_and_audit_pass", "step": args.stop_step, "teacher": plan["teacher_kind"],
                      "carry_state": plan["carry_state"]}), flush=True)


if __name__ == "__main__":
    main()
