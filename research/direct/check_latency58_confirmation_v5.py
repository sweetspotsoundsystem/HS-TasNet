"""Check reserved geometry and reject a real below-target endpoint without confirmation inference."""
from __future__ import annotations

import argparse
import ast
import copy
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256 and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "Use frozen CUDA-hidden confirmation preflight")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-confirmation-v5-preflight-plan-v1", "Different confirmation preflight")
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "result.json").exists(), "Preserve preflight")
    reservation = read(plan["reservation"]["path"])
    manifest = read(reservation["manifest"])
    primary = read(plan["primary_result"]["path"])
    require(reservation["excerpt_starts"] == [105., 135.] and reservation["duration_seconds"] == 15
            and primary["track_names"] == reservation["track_names"] == [t["name"] for t in manifest["tracks"]]
            and [e["start_seconds"] for e in primary["excerpts"]] == [30., 75.], "Original interval definitions differ")
    windows = []
    for track in manifest["tracks"]:
        for start in reservation["excerpt_starts"]:
            a, b = int(start * 44100), int((start + 15) * 44100)
            require(0 <= a < b <= track["frames"]
                    and all(b <= int(e["start_seconds"] * 44100)
                            or a >= int((e["start_seconds"] + e["duration_seconds"]) * 44100)
                            for e in primary["excerpts"]), "Confirmation overlaps primary selection windows or exceeds source")
            windows.append({"track": track["name"], "reference_start": a, "reference_end": b})
    def definitions(path):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef)}
    previous = definitions(ROOT / "research/direct/evaluate_latency58_sdr_confirmation_v4.py")
    current = definitions(ROOT / "research/direct/evaluate_latency58_sdr_confirmation_v5.py")
    unchanged = ["bound_read", "initialize_worker", "score_track", "main"]
    require(all(previous[k] == current[k] for k in unchanged), "Confirmation scoring or execution changed")
    quality = read(plan["primary_plan"]["path"])
    score = primary["results"][0]["aggregate"]["full_sdr_db"]
    require(score == 3.9306213592562225 and score < 4.057715948706591, "Negative fixture must remain the real below-target endpoint")
    # This deliberately invalid request is an executable negative fixture, not
    # an authored selection. It also explicitly denies a passed quality review.
    fixture = {"schema": "latency58-sdr-primary-selection-v1", "status": "selected_for_confirmation",
               "fixture_not_a_selection": True, "expected_outcome": "reject_real_below_target_primary_score",
               "confirmation_material_used_for_selection": False, "quality_and_probe_review_passed": False,
               "reservation": plan["reservation"], "primary_plan": plan["primary_plan"],
               "primary_result": plan["primary_result"], "primary_execution": plan["primary_execution"],
               "primary_summary": plan["primary_summary"], "summary_execution": plan["summary_execution"],
               "checkpoint": quality["checkpoint"], "model_state_sha256": primary["results"][0]["model"]["model_state_sha256"],
               "source_bindings": plan["source_bindings"]}
    path = out / "rejected-below-target-fixture.json"
    write(path, fixture)
    attempt = {"reservation": plan["reservation"], "selection": {"path": str(path), "sha256": sha(path)},
               "model_kind": "working_baseline", "source_bindings": {**plan["source_bindings"], str(path): sha(path)}}
    from research.direct.evaluate_latency58_sdr_confirmation_v5 import validate_selection
    try:
        validate_selection(attempt)
    except RuntimeError as error:
        require(str(error) == "Selected primary evidence is incomplete", "Negative fixture failed before the target check: " + str(error))
        rejection = str(error)
    else:
        raise RuntimeError("Below-target endpoint was admitted to confirmation")
    verify_inputs(plan)
    write(out / "result.json", {"schema": "latency58-confirmation-v5-preflight-v1", "status": "pass",
          "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
          "complete_reserved_windows": windows, "reserved_windows": len(windows),
          "all_windows_within_sources_and_disjoint_from_primary": True, "unchanged_functions": unchanged,
          "real_below_target_primary_score": score, "required_primary_score": 4.057715948706591,
          "rejected_fixture": {"path": str(path), "sha256": sha(path)}, "rejection": rejection,
          "current_summary_and_full_training_match_authenticated_before_rejection": True,
          "leader_cleanup_summary_dispatch_exercised": False,
          "model_instances": 0, "inference_executed": False, "training_updates_executed": 0,
          "confirmation_material_scored": False, "quality_selected": False,
          "limitations": ["This preflight is not a primary selection or a confirmation evaluation.",
                          "Leader-cleanup summary dispatch awaits its completed actual quality endpoint.",
                          "Additional passages use previously developed tracks; they are not unseen-track evidence."]})
    print({"status": "pass", "reserved_windows": len(windows), "below_target_endpoint_rejected": True}, flush=True)


if __name__ == "__main__":
    main()
