"""Exercise additional-confirmation contracts without selecting or scoring audio."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from research.direct.latency58_additional_confirmation_gate import (
    ACCEPTED_STATE, PHASE, ROOT, RESERVATION_PATH, RESERVATION_SHA256, REVIEW_AXES,
    TRIALS, SELECTION_PATH, quality_paths, validate_reservation, validate_quality_contract,
    validate_choice, load_quality_bundle, validate_selection,
)
from research.direct.run_latency58_quality import read, require, sha, write


def main():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "Use metadata-only CPU preflight")
    out = PHASE / "additional-confirmation-gate-preflight-001"
    require(not out.exists() and not SELECTION_PATH.exists()
            and not (RESERVATION_PATH.parent / "first-use.json").exists(),
            "Preserve preflight and the unconsumed reservation")
    reservation_binding = {"path": str(RESERVATION_PATH), "sha256": RESERVATION_SHA256}
    reservation = validate_reservation(reservation_binding, {})
    metadata = {str(p): sha(p) for p in (RESERVATION_PATH, Path(reservation["manifest"]["path"]),
                Path(reservation["config"]["path"]), Path(__file__).resolve(),
                ROOT / "research/direct/latency58_additional_confirmation_gate.py",
                ROOT / "research/direct/evaluate_latency58_sdr_confirmation_v5.py",
                ROOT / "research/direct/latency58_evaluate.py", ROOT / "research/direct/evaluate.py",
                ROOT / "research/evaluate.py", ROOT / "research/metrics.py")}
    completed, documents = {}, {}
    for prefix in ("leader-cleanup-250", "cleanup-successor-250", "cleanup-followup-250", "cleanup-rebound-250"):
        paths = quality_paths(prefix)
        documents[prefix] = {key: read(path) for key, path in paths.items()}
        metadata.update({str(path): sha(path) for path in paths.values()})
        completed[prefix] = validate_quality_contract(prefix, documents[prefix], reservation)
        completed[prefix]["primary_plan"] = {"path": str(paths["primary_plan"]), "sha256": sha(paths["primary_plan"])}

    rejected = []
    def reject(label, action):
        try:
            action()
        except (RuntimeError, KeyError) as error:
            rejected.append({"case": label, "error": str(error)})
        else:
            raise RuntimeError("Corrupted contract was admitted: " + label)

    quality_changes = {
        "missing_declared_axis": lambda d: d["summary_result"]["comparisons"]["working"].pop("probes"),
        "changed_model_identity": lambda d: d["summary_result"].update(model_state_sha256="0" * 64),
        "changed_primary_interval": lambda d: d["primary_result"]["excerpts"][0].update(start_seconds=45.),
        "failed_primary_execution": lambda d: d["primary_execution"].update(actual_exit_code=1),
        "failed_summary_execution": lambda d: d["summary_execution"].update(timed_out=True),
        "unreviewed_metric_coverage": lambda d: d["summary_result"].update(all_metrics_compared=False),
        "changed_aggregate": lambda d: d["summary_result"]["full_mixture_aggregate"].update(full_sdr_db=9.),
    }
    for label, change in quality_changes.items():
        corrupt = copy.deepcopy(documents["leader-cleanup-250"])
        change(corrupt)
        reject(label, lambda: validate_quality_contract("leader-cleanup-250", corrupt, reservation))

    # Two incomplete future endpoints are represented ONLY inside this pure
    # contract fixture. No corresponding quality reports or selection are
    # written, and the actual file-authentication entry point rejects them.
    synthetic = copy.deepcopy(completed)
    placeholders = {"cleanup-lr3e6-250": "e" * 64, "quarter-controlled-250": "f" * 64}
    for prefix, digest in placeholders.items():
        synthetic[prefix] = {"prefix": prefix, "model_state_sha256": digest, "full_sdr_db": 0.,
                             "checkpoint": {"path": "fixture-not-a-checkpoint", "sha256": digest},
                             "primary_plan": {"path": "fixture-not-a-primary-plan", "sha256": digest}}
    candidate = synthetic["leader-cleanup-250"]
    choice = {"schema": "latency58-additional-primary-selection-v1", "status": "selected_for_confirmation",
              "candidate_prefix": candidate["prefix"], "model_state_sha256": ACCEPTED_STATE,
              "checkpoint": candidate["checkpoint"], "primary_plan": candidate["primary_plan"],
              "reviewed_model_states": {name: row["model_state_sha256"] for name, row in synthetic.items()},
              "confirmation_material_used_for_selection": False, "quality_and_probe_review_passed": True,
              "quality_review": {axis: "Synthetic contract fixture only; this is not an actual review." for axis in REVIEW_AXES}}
    require(validate_choice(choice, synthetic) == candidate, "Positive choice contract fixture failed")
    choice_changes = {
        "confirmation_used_for_selection": lambda c: c.update(confirmation_material_used_for_selection=True),
        "review_not_passed": lambda c: c.update(quality_and_probe_review_passed=False),
        "marked_fixture_as_real_choice": lambda c: c.update(fixture_not_a_selection=True),
        "missing_listening_limitations": lambda c: c["quality_review"].pop("listening_limitations"),
        "changed_selected_checkpoint": lambda c: c["checkpoint"].update(sha256="0" * 64),
        "stale_reviewed_state": lambda c: c["reviewed_model_states"].update({"cleanup-lr3e6-250": "0" * 64}),
    }
    for label, change in choice_changes.items():
        corrupt = copy.deepcopy(choice)
        change(corrupt)
        reject(label, lambda: validate_choice(corrupt, synthetic))
    reject("incomplete_trial_pool", lambda: validate_choice(choice, completed))
    below = copy.deepcopy(choice)
    actual_high = synthetic["cleanup-rebound-250"]
    below.update(candidate_prefix=actual_high["prefix"], model_state_sha256=actual_high["model_state_sha256"],
                 checkpoint=actual_high["checkpoint"], primary_plan=actual_high["primary_plan"])
    reject("real_higher_rate_below_target_and_accepted", lambda: validate_choice(below, synthetic))
    reject("actual_quarter_quality_incomplete", lambda: load_quality_bundle("quarter-controlled-250", {}, reservation))
    reject("noncanonical_selection_path", lambda: validate_selection(
        {"path": str(out / "not-a-selection.json"), "sha256": "0" * 64}, reservation_binding))

    require(all(sha(path) == digest for path, digest in metadata.items()), "Preflight metadata changed")
    import torch
    require(not torch.cuda.is_initialized() and not SELECTION_PATH.exists()
            and not (RESERVATION_PATH.parent / "first-use.json").exists(),
            "Preflight initialized CUDA, selected a model or consumed the reserved windows")
    out.mkdir()
    plan = {"schema": "latency58-additional-confirmation-gate-preflight-plan-v1",
            "reservation": reservation_binding, "source_bindings": metadata,
            "scope": "Metadata contract fixtures only; no model loading, audio decode, selection or confirmation scoring."}
    write(out / "plan.json", plan)
    result = {"schema": "latency58-additional-confirmation-gate-preflight-result-v1", "status": "pass",
              "plan_sha256": sha(out / "plan.json"), "source_bindings": metadata, "source_bindings_unchanged": True,
              "actual_completed_primary_contracts": completed, "reserved_windows_checked": 28,
              "rejected_cases": rejected, "synthetic_positive_choice_contract_passed": True,
              "synthetic_missing_endpoint_placeholders": list(placeholders),
              "successful_scoring_or_model_loading_executed": False, "model_instances": 0,
              "new_audio_decoded": False, "source_audio_rehashed_by_this_preflight": False,
              "cuda_initialized": False, "quality_selected": False, "reservation_consumed": False,
              "confirmation_scoring_executor_implemented": False,
              "limitations": ["Pure choice fixtures cannot authenticate uncompleted actual trial results.",
                              "Actual selection and inference still require all completed, bound quality evidence.",
                              "Additional within-song windows are not an independent new-song test."]}
    write(out / "result.json", result)
    print(json.dumps({"status": "pass", "actual_primary_contracts": len(completed),
                      "rejected_cases": len(rejected), "model_instances": 0, "reservation_consumed": False}), flush=True)


if __name__ == "__main__":
    main()
