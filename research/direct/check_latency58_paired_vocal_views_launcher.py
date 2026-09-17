"""Qualify endpoint-role and completion guards without model inference."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.run_latency58_paired_vocal_views import cpu_environment, load_endpoint, validate_records
from research.direct.run_latency58_deployed_vocal_views import budget_snapshot


def main():
    cpu_environment()
    out = PHASE / "paired-vocal-launcher-check-001"
    require(not out.exists(), "Preserve earlier launcher qualification")
    parent = PHASE / "branch-pitch-ema-002"
    live = PHASE / "branch-long-context-006"
    records, models, bindings = load_endpoint(parent)
    require(models["raw"]["checkpoint"]["sha256"] == "4ca7e5ea6dfc4298f6f77377337c96e860ad54a1c35e8ff2af917346afc20d01"
            and models["ema"]["checkpoint"]["sha256"] == "2e0600a3619106682d789c1e8c9648c5c6adfe67dce51e08d5e17556064f3fb0",
            "Completed retained fixture changed")
    launcher = ROOT / "research/direct/run_latency58_paired_vocal_views.py"
    for path in (Path(__file__).resolve(), launcher, live / "plan.json"):
        bindings[str(path)] = sha(path)
    verify_inputs({"source_bindings": bindings})
    budget = read(PHASE / "branch-gru-int8-post-ci-storage-001.json")
    before = budget_snapshot(budget)
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
        "negative_cases": results, "live_endpoint_refused": True, "live_refusal_reason": live_reason,
        "live_directory_inventory_unchanged": True, "model_inference": False,
        "source_audio_decoded": False, "training_updates": 0, "gpu_used": False,
        "quality_measured": False, "elapsed_seconds": time.monotonic() - began, "budget_after": budget_snapshot(budget)}
    write(out / "result.json", result)
    print(json.dumps({"status": "pass", "negative_cases": len(results), "live_endpoint_refused": True}), flush=True)


if __name__ == "__main__":
    main()
