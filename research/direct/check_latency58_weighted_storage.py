"""Check reservation occupancy against complete before/during/after file inventories."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.run_latency58_deployed_vocal_views import budget_snapshot
from research.direct.latency58_weighted_storage import policy, snapshot, peak_accounting


def main():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Use the CPU workspace")
    out = PHASE / "weighted-vocal-quarter-storage-001"
    require(out.is_dir() and not (out / "allocation-qualification.json").exists(), "Preserve storage qualification")
    source_path = PHASE / "branch-grouped-vocal-013/plan.json"
    source = read(source_path)
    plan = {"storage_budget": source["storage_budget"], "weighted_storage": policy()}
    require(all(not Path(path).exists() for path in policy()["diagnostic_roots"]), "Allocation must start with fresh artifact roots")
    bindings = {str(path): sha(path) for path in (Path(__file__).resolve(), source_path,
        ROOT / "research/direct/latency58_weighted_storage.py", out / "inventory.json", out / "historical-references.json")}
    verify_inputs({"source_bindings": bindings})
    old = budget_snapshot(source["storage_budget"])
    fresh = snapshot(plan, require_room=False)
    require(fresh["projected_peak_bytes"] == old["conservative_total_with_reservations"] + 380_000_000 + 30_000_000,
            "Fresh allocation did not retain all legacy reserves and the full new generation")
    baseline = fresh["actual_counted_and_external_git_bytes"]
    expected = baseline + 800_000_000 + 600_000_000 + 280_000_000 + 380_000_000
    cases = []
    # Explicit inventories from an empty trial through a durable final file.
    # Both the current and pending files are included simultaneously at the
    # replacement boundary; completed reports occupy diagnostic capacity.
    for name, files in (
        ("before_first_update", {}),
        ("resource_monitor_and_reports", {"diagnostics": 4_000_000}),
        ("first_checkpoint", {"current": 359_000_000, "diagnostics": 10_000_000}),
        ("atomic_replacement", {"current": 359_000_000, "pending": 380_000_000, "diagnostics": 137_000_000}),
        ("interrupted_replacement", {"current": 380_000_000, "pending": 380_000_000, "diagnostics": 200_000_000}),
        ("completed_final_and_evaluations", {"final": 380_000_000, "diagnostics": 250_000_000}),
        ("all_diagnostic_capacity_occupied", {"final": 380_000_000, "diagnostics": 280_000_000}),
    ):
        value = peak_accounting(baseline + sum(files.values()),
            permanent_occupied=files.get("current", 0) + files.get("final", 0),
            transient_occupied=files.get("pending", 0), diagnostic_occupied=files.get("diagnostics", 0))
        require(value["projected_peak_bytes"] == expected, "Reservation occupancy lost or double-counted bytes: " + name)
        require(sum(value["remaining_reserved_bytes"].values()) + sum(files.values())
                == 800_000_000 + 600_000_000 + 280_000_000 + 380_000_000,
                "Occupied files reduced the declared total coverage")
        cases.append({"case": name, "explicit_file_inventory": files, "accounting": value})
    rejected = []
    invalid = (
        ("negative_occupancy", {"permanent_occupied": -1}),
        ("permanent_file_over_ceiling", {"permanent_occupied": 380_000_001}),
        ("pending_file_over_ceiling", {"transient_occupied": 380_000_001}),
        ("diagnostics_over_allocation", {"diagnostic_occupied": 280_000_001}),
        ("two_permanent_generations", {"permanent_occupied": 760_000_000}),
        ("noninteger_occupancy", {"diagnostic_occupied": 1.5}),
    )
    for name, change in invalid:
        try:
            peak_accounting(baseline, **({"permanent_occupied": 0, "transient_occupied": 0,
                                         "diagnostic_occupied": 0} | change))
        except RuntimeError as error:
            rejected.append({"case": name, "reason": str(error)})
        else:
            raise RuntimeError("Invalid storage accounting accepted: " + name)
    for name, modified in (
        ("historical_directory_as_new_diagnostics", {**plan, "weighted_storage": {**policy(),
            "diagnostic_roots": [str(PHASE / "branch-grouped-vocal-013")]}}),
        ("changed_standing_save_reserve", {**plan, "storage_budget": {**source["storage_budget"],
            "live_training_save_reservation_bytes": 380_000_000}}),
    ):
        try:
            snapshot(modified, require_room=False)
        except RuntimeError as error:
            rejected.append({"case": name, "reason": str(error)})
        else:
            raise RuntimeError("Invalid allocation policy accepted: " + name)
    verify_inputs({"source_bindings": bindings})
    result = {"schema": "latency58-quarter-vocal-storage-qualification-v1", "status": "pass",
              "source_bindings": bindings, "source_bindings_unchanged": True, "policy": policy(),
              "legacy_budget": old, "fresh_allocation": fresh, "inventory_cases": cases,
              "rejected_cases": rejected, "all_standing_reservation_capacity_retained": True,
              "production_space_currently_sufficient": fresh["headroom_after_complete_peak_bytes"] > 0,
              "cache_files_removed": False, "gpu_used": False, "completed_utc": datetime.now(timezone.utc).isoformat()}
    write(out / "allocation-qualification.json", result)
    print(json.dumps({"status": "pass", "inventory_cases": len(cases), "rejected_cases": len(rejected),
                      "projected_peak_before_cache_reclamation": fresh["projected_peak_bytes"],
                      "result_sha256": sha(out / "allocation-qualification.json")}), flush=True)


if __name__ == "__main__":
    main()
