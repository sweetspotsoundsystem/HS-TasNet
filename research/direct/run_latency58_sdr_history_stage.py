"""Run a monitored, independently audited stage of the frozen matched history study."""
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
    parser.add_argument("--continuation-decision", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Plan or cwd differs")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-sdr-history-training-v1", "Unknown training plan")
    protocol_binding = plan["matched_protocol"]
    require(sha(protocol_binding["path"]) == protocol_binding["sha256"], "Matched protocol changed")
    protocol = read(protocol_binding["path"])
    verify_inputs(protocol)
    shared_protocol = ("config", "teacher_history_samples", "scored_samples", "teacher", "teacher_weight",
                       "teacher_model_state_sha256", "drum_weight", "objective_version", "accumulation_version",
                       "history_version", "microbatch_size", "accumulation_steps", "carry_state",
                       "optimizer_initialization", "quality_endpoints", "continuation_rules",
                       "counted_roots", "stop_counted_bytes")
    require(protocol["schema"] == "latency58-matched-history-protocol-v1"
            and all(plan[k] == protocol[k] for k in shared_protocol)
            and plan["warmup_samples"] in protocol["student_history_samples"] == [88064, 352256]
            and plan["parent"]["model_state_sha256"] == protocol["parent_model_state_sha256"],
            "Training arm differs from the frozen matched protocol")
    from research.direct.latency58_sdr_history_checkpoint import read_generation, require_space
    require_space(plan, 10_000_000 if plan["resource_only"] else 350_000_000)
    out, run = args.output_directory.absolute(), Path(plan["run_dir"])
    require(out.is_relative_to(ROOT / "research/direct/runs/latency58") and not out.exists(), "Use a fresh stage directory")
    previous, execution = read(args.previous_monitor), read(args.previous_execution)
    require(previous["status"] == previous["supervisor_health"] == "pass" and previous["child_exit_code"] == 0
            and previous["post_exit_quiet_completed"] and execution["actual_exit_code"] == 0
            and execution["source_bindings_unchanged"], "Previous GPU stage did not finish successfully")
    for binding in plan["functional_proofs"]:
        proof, proof_execution = read(binding["result"]), read(binding["execution"])
        require(proof["status"] == "pass" and proof["source_bindings_unchanged"]
                and proof_execution["actual_exit_code"] == 0 and not proof_execution["timed_out"]
                and proof_execution["source_bindings_unchanged"], "Preserved context qualification failed")
    functional = read(plan["accumulation_functional"]["path"])
    functional_execution = read(plan["accumulation_functional_execution"]["path"])
    require(functional["schema"] == "latency58-matched-history-functional-result-v1"
            and functional["status"] == "pass" and functional["source_bindings_unchanged"]
            and functional["accumulation_version"] == plan["accumulation_version"]
            and functional["objective_version"] == plan["objective_version"]
            and functional["stem_weights"] == [plan["drum_weight"], 1, 1, 1] == [2, 1, 1, 1]
            and functional["teacher_weight"] == plan["teacher_weight"] == .5
            and functional["model_state_sha256"] == plan["parent"]["model_state_sha256"]
            and functional["teacher_model_state_sha256"] == plan["teacher_model_state_sha256"]
            and functional["microbatch_size"] == plan["microbatch_size"] == 4
            and functional["accumulation_steps"] == plan["accumulation_steps"] == 4
            and functional["all_21_parameter_gradients_match"]
            and functional["history_version"] == plan["history_version"]
            and functional["teacher_history_samples"] == plan["teacher_history_samples"] == 352256
            and functional["student_history_samples"] == [88064, 352256]
            and functional["scored_samples"] == plan["scored_samples"] == 88064
            and len(functional["geometry"]) == 4 and len(functional["gradient_arms"]) == 2
            and functional_execution["actual_exit_code"] == 0 and not functional_execution["timed_out"]
            and functional_execution["source_bindings_unchanged"]
            and functional_execution["plan_sha256"] == functional["plan_sha256"]
            and sha(plan["accumulation_functional"]["path"]) == plan["accumulation_functional"]["sha256"]
            and sha(plan["accumulation_functional_execution"]["path"]) == plan["accumulation_functional_execution"]["sha256"]
            and all(sha(p) == v for p, v in functional["source_bindings"].items()),
            "Accumulation functional qualification differs")
    if plan["resource_only"]:
        require(args.stop_step == 2 and not run.exists(), "Resource rehearsal must be fresh and exactly two updates")
    else:
        resource = read(plan["full_resource"]["path"])
        resource_execution = read(plan["full_resource_execution"]["path"])
        resource_monitor = read(resource_execution["monitor_result"])
        resource_plan = read(plan["resource_plan"]["path"])
        shared = ("config", "parent", "warmup_samples", "scored_samples", "carry_state",
                  "teacher_kind", "teacher_weight", "teacher_model_state_sha256", "teacher",
                  "precision_policy", "torch_version", "environment",
                  "helper_source", "watchdog_source", "manifest_sha256", "functional_proofs",
                  "microbatch_size", "accumulation_steps", "accumulation_version",
                  "drum_weight", "objective_version", "teacher_history_samples", "history_version",
                  "optimizer_initialization", "matched_protocol",
                  "accumulation_functional", "accumulation_functional_execution")
        require(resource["schema"] == "latency58-sdr-history-resource-result-v1" and resource["status"] == "pass"
                and resource["training_updates_executed"] == 2 and not resource["checkpoint_written"]
                and resource["source_bindings_unchanged"] and resource["all_parameter_gradients_present"]
                and resource["fixed_buffers_unchanged"] and resource["teacher_unchanged"]
                and resource["initial_model_state_sha256"] == plan["parent"]["model_state_sha256"]
                and resource["teacher_model_state_sha256"] == plan["teacher_model_state_sha256"]
                and resource["config"] == plan["config"] and resource["carry_state"]
                and resource["microbatch_size"] == plan["microbatch_size"]
                and resource["accumulation_steps"] == plan["accumulation_steps"]
                and resource["accumulation_version"] == plan["accumulation_version"]
                and resource["drum_weight"] == plan["drum_weight"]
                and resource["objective_version"] == plan["objective_version"]
                and resource["augmented_examples_executed"] == 32
                and resource["warmup_samples"] == plan["warmup_samples"]
                and resource["teacher_history_samples"] == plan["teacher_history_samples"]
                and resource["history_version"] == plan["history_version"]
                and resource["optimizer_initialization"] == plan["optimizer_initialization"]
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
    if start:
        require(args.continuation_decision is not None, "Continuation requires the planned quality review")
        decision = read(args.continuation_decision)
        verify_inputs(decision)
        require(decision["schema"] == "latency58-sdr-history-continuation-v1"
                and decision["status"] == "continue" and decision["training_plan_sha256"] == args.plan_sha256
                and decision["start_step"] == start and decision["stop_step"] == args.stop_step,
                "Continuation decision does not authorize this stage")
        bind.append(args.continuation_decision)
    else:
        require(args.continuation_decision is None and args.stop_step == (2 if plan["resource_only"] else 250),
                "Require the resource rehearsal or first quality endpoint before further updates")
    stage = {"schema": "latency58-sdr-history-stage-v1", "plan_sha256": args.plan_sha256,
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
            "argv": [PYTHON, "-u", "-m", "research.direct.train_latency58_sdr_history", "--plan", str(args.plan.resolve()),
                     "--plan-sha256", args.plan_sha256, "--stage", str(out / "stage.json"),
                     "--stage-sha256", sha(out / "stage.json")]}
    write(out / "watchdog-spec.json", spec)
    timeout = max(360, (args.stop_step - start) * 60 + 180)
    argv = [PYTHON, plan["watchdog_source"], "--launch-spec", str(out / "watchdog-spec.json"),
            "--launch-spec-sha256", sha(out / "watchdog-spec.json"), "--output-dir", str(monitor_out),
            "--max-runtime-seconds", str(timeout), "--poll-seconds", "2", "--query-timeout-seconds", "10",
            "--startup-grace-seconds", "120", "--progress-timeout-seconds", "120", "--stop-grace-seconds", "15",
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
            "schema": "latency58-sdr-history-resource-qualification-v1", "status": "pass",
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
    argv = [PYTHON, "-u", "-m", "research.direct.audit_latency58_sdr_history", "--plan", str(args.plan.resolve()),
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
        retirement = {"schema": "latency58-sdr-history-replaced-optimizer-v1", "path": str(old_optimizer),
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
