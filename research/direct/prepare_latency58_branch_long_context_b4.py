"""Retain two scored seconds and the full B16 objective using four B4 microbatches."""
import json
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs


def main():
    from research.direct.latency58_sdr_checkpoint import require_space
    old = PHASE / "branch-long-context-002"
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
            and (old / "resource-run/metrics.jsonl").stat().st_size == 0 and not (old / "production-run").exists()
            and failed["supervisor_health"] == "pass" and failed["post_exit_quiet_completed"]
            and not Path("/proc", str(failed["child_pid"])).exists(),
            "Require the closed allocation failure before any training updates")
    log = failed_path.parent / "child.log"
    require("torch.OutOfMemoryError" in log.read_text()
            and "terms.total.backward()" in log.read_text(),
            "Failed comparison traceback changed")
    idle = PHASE / "branch-long-context-idle-003"
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
    before = (source / "train_latency58_branch_long_context_retry.py").read_text()
    after = (source / "train_latency58_branch_long_context_b4.py").read_text()
    replacements = [("two B8 microbatches", "four B4 microbatches"),
                    ('"microbatch_size": 8', '"microbatch_size": 4'),
                    ('config["microbatch_size"] == 8 and plan["accumulation_steps"] == 2',
                     'config["microbatch_size"] == 4 and plan["accumulation_steps"] == 4'),
                    ("check_latency58_branch_long_context_gpu_released import",
                     "check_latency58_branch_long_context_gpu_b4 import")]
    expected = before
    for initial, replacement in replacements:
        require(initial in expected, "Missing intended B4 trainer substitution")
        expected = expected.replace(initial, replacement)
    require(after == expected, "B4 trainer changed more than microbatch geometry and resource checker")
    b8_parity_path = old / "resource-run/branch-long-context-gpu-parity.json"
    b8_parity = read(b8_parity_path)
    require(b8_parity["status"] == b8_parity["logical_batch_loss"]["status"] == "pass"
            and b8_parity["microbatch_size"] == 8 and b8_parity["scored_samples"] == 88320
            and b8_parity["trained_parent_weights_unchanged"] and len(b8_parity["all_40_gradients"]) == 40
            and all(v["maximum_error"] == 0 and v["reference_norm"] > 0
                    for v in b8_parity["all_40_gradients"].values()),
            "Completed B8 functional evidence changed")
    loss = PHASE / "branch-long-context-loss-003"
    loss_plan, loss_result = read(loss / "plan.json"), read(loss / "result.json")
    verify_inputs(loss_plan)
    require(loss_result["status"] == "pass" and loss_result["source_bindings_unchanged"]
            and loss_result["plan_sha256"] == sha(loss / "plan.json")
            and loss_result["unchanged_full_batch_objective_and_audio_gradients_match"]
            and loss_result["policy"] == original["logical_batch_loss"]
            and loss_result["cases"][0]["batch_size"] == 16
            and loss_result["cases"][0]["microbatch_size"] == loss_plan["microbatch_size"] == 4,
            "Full B16 loss and gradient accumulation across four pieces is unqualified")
    for name in ("execution.json", "root-execution.json"):
        result = read(loss / name)
        require(result["actual_exit_code"] == 0 and result["source_bindings_unchanged"]
                and not result.get("timed_out", False), "B4 loss qualification did not close")
    paths = [Path(__file__).resolve(), original_path, old / "resource-stage/root-execution.json",
             old / "resource-stage/execution.json", failed_path, log, idle_monitor_path, b8_parity_path]
    paths.extend(loss / name for name in ("plan.json", "result.json", "execution.json", "root-execution.json"))
    paths.extend(idle / name for name in ("execution.json", "root-execution.json", "child-result.json",
                 "event-continuity.json", "inputs.json", "watchdog-spec.json", "command.json"))
    paths.extend(source / name for name in ("check_latency58_branch_long_context_gpu_b4.py",
                 "train_latency58_branch_long_context_b4.py", "run_latency58_branch_long_context_b4.py",
                 "observe_latency58_branch_long_context_b4_prefix.py", "report_latency58_branch_long_context_b4.py"))
    for name in ("execution.json", "root-execution.json"):
        path = PHASE / "branch-long-context-preparation-stage-001" / name
        previous = read(path)
        require(previous["actual_exit_code"] == 0 and previous["source_bindings_unchanged"]
                and not previous.get("timed_out", False), "Original CPU qualification did not close")
        paths.append(path)
    out = PHASE / "branch-long-context-003"
    require(Path.cwd() == ROOT and not out.exists(), "Preserve all earlier attempts")
    counted = require_space(original, 650_000_000)
    plan = {**original, "name": out.name, "output_directory": str(out),
            "retry_of": str(original_path),
            "retry_reason": "B8 full-context comparison passed; real objective backward exceeded the unchanged allocator limit before training. Retain B16 and two scored seconds with four B4 microbatches.",
            "config": {**original["config"], "microbatch_size": 4}, "accumulation_steps": 4,
            "training_context_implementation": "two-second scored suffix after detached two-second final-frame warmup; globally normalized B4 accumulation",
            "previous_execution_for_resource": str(idle / "execution.json"),
            "resource_context_comparison": "Full B4 two-second reference and optimized passes in isolated scopes with explicit lifetime checks",
            "source_bindings": {**original["source_bindings"], **idle_inputs["source_bindings"], **loss_plan["source_bindings"],
                                **{str(path): sha(path) for path in paths}}}
    require(plan["config"] == {**original["config"], "microbatch_size": 4} and plan["ema"] == original["ema"]
            and plan["parent_checkpoint"] == original["parent_checkpoint"]
            and plan["qualified_data_prefix"] == original["qualified_data_prefix"]
            and plan["logical_batch_loss"] == original["logical_batch_loss"],
            "Retry changed the qualified learning or data recipe")
    verify_inputs(plan)
    out.mkdir()
    write(out / "preparation.json", {"status": "pass", "original_plan": str(original_path),
          "original_plan_sha256": sha(original_path), "parent_and_data_cpu_qualification_reused_with_sources_unchanged": True,
          "production_training_changes": replacements, "logical_batch_size": 16, "microbatch_size": 4,
          "accumulation_steps": 4, "b4_loss_qualification": str(loss / "result.json"),
          "full_two_second_b4_gpu_rehearsal_required": True,
          "parent_full_sdr_db": plan["parent_full_sdr_db"], "starting_from_saved_parent": True,
          "failed_attempt_training_updates": 0, "source_bindings_unchanged": True,
          "forecast_including_outside_and_run": counted + 650_000_000 + plan["outside_roots_reservation_bytes"],
          "artifact_cap_bytes": 90_000_000_000, "monitored_idle_continuity": str(idle / "execution.json"),
          "limitation": "B8 context tensor release passed, but real loss backward did not fit the allocator. B4 preserves the mathematical logical objective and two-second context. GPU capacity and an SDR improvement remain unproven."})
    plan["source_bindings"][str(out / "preparation.json")] = sha(out / "preparation.json")
    write(out / "plan.json", plan)
    print(json.dumps({"status": "pass", "plan": str(out / "plan.json"), "plan_sha256": sha(out / "plan.json"),
                      "logical_objective_and_two_second_context_preserved": True, "microbatch_size": 4, "original_attempt_preserved": True}), flush=True)


if __name__ == "__main__":
    main()
