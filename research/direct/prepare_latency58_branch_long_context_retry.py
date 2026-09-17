"""Preserve the allocation failure and retry with isolated comparison tensor lifetimes."""
import json
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs


def main():
    from research.direct.latency58_sdr_checkpoint import require_space
    old = PHASE / "branch-long-context-001"
    original_path = old / "plan.json"
    original = read(original_path)
    verify_inputs(original)
    stopped = read(old / "resource-stage/root-execution.json")
    execution = read(old / "resource-stage/execution.json")
    failed_path = Path(execution["monitor_result"])
    failed = read(failed_path)
    require(stopped["actual_exit_code"] == execution["actual_exit_code"] == failed["child_exit_code"] == 1
            and stopped["source_bindings_unchanged"] and execution["source_bindings_unchanged"]
            and stopped["training_updates"] == 0 and not stopped["checkpoint_written"]
            and not (old / "resource-run/metrics.jsonl").exists() and not (old / "production-run").exists()
            and failed["supervisor_health"] == "pass" and failed["post_exit_quiet_completed"]
            and not Path("/proc", str(failed["child_pid"])).exists(),
            "Require the closed allocation failure before any training updates")
    log = failed_path.parent / "child.log"
    require("torch.OutOfMemoryError" in log.read_text()
            and "latency58_branch_memory_context.py" in log.read_text(),
            "Failed comparison traceback changed")
    idle = PHASE / "branch-long-context-idle-002"
    idle_execution = read(idle / "execution.json")
    idle_root = read(idle / "root-execution.json")
    idle_monitor_path = Path(idle_execution["monitor_result"])
    idle_monitor = read(idle_monitor_path)
    idle_child = read(idle / "child-result.json")
    idle_inputs = read(idle / "inputs.json")
    verify_inputs(idle_inputs)
    require(idle_execution["actual_exit_code"] == idle_root["actual_exit_code"] == 0
            and idle_execution["source_bindings_unchanged"] and idle_root["source_bindings_unchanged"]
            and idle_monitor["status"] == idle_monitor["supervisor_health"] == "pass"
            and idle_monitor["child_exit_code"] == 0 and idle_monitor["post_exit_quiet_completed"]
            and idle_child["status"] == "pass" and idle_child["event_continuity_passed"]
            and idle_child["source_bindings_unchanged"] and not idle_child["gpu_workload_started"]
            and idle_inputs["previous_event_record_id"] == failed["last_event_record_id"],
            "Require completed monitored idle continuity")
    source = ROOT / "research/direct"
    before = (source / "train_latency58_branch_long_context.py").read_text()
    after = (source / "train_latency58_branch_long_context_retry.py").read_text()
    require(after == before.replace("check_latency58_branch_long_context_gpu import",
                                    "check_latency58_branch_long_context_gpu_released import"),
            "Retry changed production training instead of only the resource checker import")
    paths = [Path(__file__).resolve(), original_path, old / "resource-stage/root-execution.json",
             old / "resource-stage/execution.json", failed_path, log, idle_monitor_path]
    paths.extend(idle / name for name in ("execution.json", "root-execution.json", "child-result.json",
                 "event-continuity.json", "inputs.json", "watchdog-spec.json", "command.json"))
    paths.extend(source / name for name in ("check_latency58_branch_long_context_gpu_released.py",
                 "train_latency58_branch_long_context_retry.py", "run_latency58_branch_long_context_retry.py",
                 "observe_latency58_branch_long_context_retry_prefix.py", "report_latency58_branch_long_context_retry.py"))
    for name in ("execution.json", "root-execution.json"):
        path = PHASE / "branch-long-context-preparation-stage-001" / name
        previous = read(path)
        require(previous["actual_exit_code"] == 0 and previous["source_bindings_unchanged"]
                and not previous.get("timed_out", False), "Original CPU qualification did not close")
        paths.append(path)
    out = PHASE / "branch-long-context-002"
    require(Path.cwd() == ROOT and not out.exists(), "Preserve all earlier attempts")
    counted = require_space(original, 650_000_000)
    plan = {**original, "name": out.name, "output_directory": str(out),
            "retry_of": str(original_path),
            "retry_reason": "Resource comparison allocation failed before training; release completed comparison pass tensors before the next render",
            "previous_execution_for_resource": str(idle / "execution.json"),
            "resource_context_comparison": "Full B8 two-second reference and optimized passes in isolated scopes with explicit lifetime checks",
            "source_bindings": {**original["source_bindings"], **idle_inputs["source_bindings"],
                                **{str(path): sha(path) for path in paths}}}
    require(plan["config"] == original["config"] and plan["ema"] == original["ema"]
            and plan["parent_checkpoint"] == original["parent_checkpoint"]
            and plan["qualified_data_prefix"] == original["qualified_data_prefix"]
            and plan["logical_batch_loss"] == original["logical_batch_loss"],
            "Retry changed the qualified learning or data recipe")
    verify_inputs(plan)
    out.mkdir()
    write(out / "preparation.json", {"status": "pass", "original_plan": str(original_path),
          "original_plan_sha256": sha(original_path), "cpu_qualification_reused_with_sources_unchanged": True,
          "production_training_changed_only_resource_checker_import": True,
          "cpu_qualification_repeated": False, "full_two_second_b8_gpu_rehearsal_required": True,
          "parent_full_sdr_db": plan["parent_full_sdr_db"], "starting_from_saved_parent": True,
          "failed_attempt_training_updates": 0, "source_bindings_unchanged": True,
          "forecast_including_outside_and_run": counted + 650_000_000 + plan["outside_roots_reservation_bytes"],
          "artifact_cap_bytes": 90_000_000_000, "monitored_idle_continuity": str(idle / "execution.json"),
          "limitation": "Reference lifetime retention is a hypothesis for the allocation failure. This preparation does not establish GPU capacity or an SDR improvement."})
    plan["source_bindings"][str(out / "preparation.json")] = sha(out / "preparation.json")
    write(out / "plan.json", plan)
    print(json.dumps({"status": "pass", "plan": str(out / "plan.json"), "plan_sha256": sha(out / "plan.json"),
                      "learning_recipe_unchanged": True, "original_attempt_preserved": True}), flush=True)


if __name__ == "__main__":
    main()
