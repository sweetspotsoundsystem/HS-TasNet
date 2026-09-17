"""Qualify endpoint-role and completion guards without model inference."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.run_latency58_paired_vocal_serial import cpu_environment, load_endpoint, validate_records, checker_budget_from_snapshot
from research.direct.run_latency58_deployed_vocal_views import budget_snapshot


def main():
    cpu_environment()
    out = PHASE / "paired-vocal-serial-launcher-check-001"
    require(not out.exists(), "Preserve earlier launcher qualification")
    parent = PHASE / "branch-pitch-ema-002"
    live = PHASE / "branch-grouped-vocal-012"
    records, models, bindings = load_endpoint(parent)
    require(models["raw"]["checkpoint"]["sha256"] == "4ca7e5ea6dfc4298f6f77377337c96e860ad54a1c35e8ff2af917346afc20d01"
            and models["ema"]["checkpoint"]["sha256"] == "2e0600a3619106682d789c1e8c9648c5c6adfe67dce51e08d5e17556064f3fb0",
            "Completed retained fixture changed")
    launcher = ROOT / "research/direct/run_latency58_paired_vocal_serial.py"
    for path in (Path(__file__).resolve(), launcher, live / "plan.json"):
        bindings[str(path)] = sha(path)
    verify_inputs({"source_bindings": bindings})
    budget_path = live / "plan.json"
    budget = read(budget_path)["storage_budget"]
    before = budget_snapshot(budget)
    previous_launcher = ROOT / "research/direct/run_latency58_paired_vocal_views.py"
    trees = [ast.parse(path.read_text()) for path in (previous_launcher, launcher)]
    preserved = ("binding", "merge_bindings", "cpu_environment", "validate_records", "load_endpoint", "run", "main")
    for name in preserved:
        functions = [next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name) for tree in trees]
        require(ast.dump(functions[0], include_attributes=False) == ast.dump(functions[1], include_attributes=False),
                "Endpoint or inference orchestration changed: " + name)
    original_qualified_plan = PHASE / "paired-vocal-long-context-006/plan.json"
    qualified = read(original_qualified_plan)
    unchanged_dependencies = {}
    for filename in ("check_latency58_branch_vocal_views.py", "evaluate_latency58_branch_vocal_views.py",
                     "compare_latency58_vocal_views.py", "report_latency58_branch_gru_int8.py", "latency58_vocal_views.py"):
        path = ROOT / "research/direct" / filename
        require(sha(path) == qualified["source_bindings"][str(path)], "Qualified inference/metric source changed")
        unchanged_dependencies[str(path)] = sha(path)
    bindings.update(unchanged_dependencies)
    bindings.update({str(path): sha(path) for path in (budget_path, previous_launcher, original_qualified_plan)})
    verify_inputs({"source_bindings": bindings})
    cases = [
        ("root_nonzero_exit", ("root_execution", "actual_exit_code"), 1, "Root execution"),
        ("root_timeout", ("root_execution", "timed_out"), True, "Root execution"),
        ("monitor_nonzero_exit", ("monitor", "child_exit_code"), 1, "Training monitor"),
        ("incomplete_training_step", ("training", "updates"), 3999, "Training, save or audit"),
        ("optimizer_owned_by_ema", ("audit", "optimizer_owner"), "model.pt", "Raw/EMA ownership"),
        ("latency_changed", ("audit", "algorithmic_latency_samples"), 512, "Raw/EMA ownership"),
        ("raw_role_replaced_by_ema", ("terminal", "checkpoints", "raw"), models["ema"]["checkpoint"], "Raw/EMA role"),
        ("wrong_raw_model_state", ("quality", "raw", "result", "results", 0, "model", "model_state_sha256_after"),
            models["ema"]["model_state_sha256"], "Raw/EMA role"),
        ("incomplete_full14", ("quality", "ema", "result", "track_count"), 13, "Original full14 score"),
        ("failed_full14_execution", ("quality", "raw", "execution", "actual_exit_code"), 1, "Full14 execution"),
    ]
    out.mkdir()
    plan = {"schema": "latency58-paired-vocal-launcher-control-v1", "source_bindings": bindings,
        "completed_fixture": str(parent), "live_fixture": str(live), "negative_cases": [c[0] for c in cases],
        "model_inference": False, "source_audio_decoded": False, "budget_before": before}
    write(out / "plan.json", plan)
    plan_sha, began = sha(out / "plan.json"), time.monotonic()
    unchanged_records = copy.deepcopy(records)
    require(validate_records(records) == models and records == unchanged_records, "Valid role validation mutated its inputs")
    results = []
    for name, location, replacement, expected in cases:
        changed = copy.deepcopy(records)
        node = changed
        for key in location[:-1]:
            node = node[key]
        node[location[-1]] = replacement
        try:
            validate_records(changed)
        except RuntimeError as error:
            require(expected in str(error), "Negative case failed for an unrelated reason: " + name)
            results.append({"case": name, "rejected": True, "reason": str(error)})
        else:
            raise RuntimeError("Invalid endpoint was accepted: " + name)
    mapped, mapping = checker_budget_from_snapshot(budget, before)
    counted = sum(before["counted_roots"].values())
    # Exercise the exact arithmetic used by the unchanged functional checker
    # and evaluator, including their respective 10 MB and 5 MB increments.
    require(counted + mapped["outside_roots_reservation_bytes"] + 450_000_000
                == before["conservative_total_with_reservations"] + 10_000_000
            and counted + mapped["outside_roots_reservation_bytes"] + 445_000_000
                == before["conservative_total_with_reservations"] + 5_000_000
            and mapped["stop_counted_bytes"] + mapped["outside_roots_reservation_bytes"] == 90_000_000_000,
            "Child guards lost standing or diagnostic reservations")
    budget_results = []
    for name in ("wrong_cap", "small_save_reserve", "small_outside_allowance", "small_diagnostic_allowance",
                 "inconsistent_observation", "insufficient_diagnostic_space"):
        changed_budget, changed_observed = copy.deepcopy(budget), copy.deepcopy(before)
        if name == "wrong_cap":
            changed_budget["authorized_cap_bytes"] = 80_000_000_000
        elif name == "small_save_reserve":
            changed_budget["live_training_save_reservation_bytes"] = 599_999_999
        elif name == "small_outside_allowance":
            changed_budget["other_outside_allowance_bytes"] = 799_999_999
        elif name == "small_diagnostic_allowance":
            changed_budget["diagnostic_artifact_allowance_bytes"] = 49_999_999
        elif name == "inconsistent_observation":
            changed_observed["external_git_common_bytes"] += 1
        else:
            increment = changed_observed["headroom_bytes"] - 49_999_999
            first_root = next(iter(changed_observed["counted_roots"]))
            changed_observed["counted_roots"][first_root] += increment
            changed_observed["conservative_total_with_reservations"] += increment
            changed_observed["headroom_bytes"] -= increment
        try:
            checker_budget_from_snapshot(changed_budget, changed_observed)
        except RuntimeError as error:
            budget_results.append({"case": name, "rejected": True, "reason": str(error)})
        else:
            raise RuntimeError("Invalid budget was accepted: " + name)
    before_files = {str(p.relative_to(live)) for p in live.rglob("*") if p.is_file()}
    try:
        load_endpoint(live)
    except RuntimeError as error:
        require("Finish the saved endpoint" in str(error), "Live endpoint failed for an unrelated reason")
        live_reason = str(error)
    else:
        raise RuntimeError("Unfinished live training was accepted")
    after_files = {str(p.relative_to(live)) for p in live.rglob("*") if p.is_file()}
    require(before_files == after_files, "Read-only refusal changed the live endpoint inventory")
    verify_inputs(plan)
    require(sha(out / "plan.json") == plan_sha, "Control plan changed")
    result = {"status": "pass", "plan_sha256": plan_sha, "launcher_sha256": sha(launcher),
        "source_bindings_unchanged": True, "completed_pair_roles_authenticated": models,
        "reservation_mapping_qualified": True, "reservation_mapping": mapping,
        "budget_negative_cases": budget_results, "unchanged_function_asts": list(preserved),
        "inference_and_metric_function_asts_unchanged": True,
        "unchanged_inference_and_metric_dependency_sha256": unchanged_dependencies,
        "requalification_scope": "Endpoint guards, reservation arithmetic and unchanged qualified inference/metric sources; no new inference or quality result.",
        "negative_cases": results, "live_endpoint_refused": True, "live_refusal_reason": live_reason,
        "live_directory_inventory_unchanged": True, "model_inference": False,
        "source_audio_decoded": False, "training_updates": 0, "gpu_used": False,
        "quality_measured": False, "elapsed_seconds": time.monotonic() - began, "budget_after": budget_snapshot(budget)}
    write(out / "result.json", result)
    print(json.dumps({"status": "pass", "negative_cases": len(results), "live_endpoint_refused": True}), flush=True)


if __name__ == "__main__":
    main()
