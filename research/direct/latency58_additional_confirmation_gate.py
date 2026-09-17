"""Authenticate one primary selection before consuming the additional windows.

No model is constructed here. The caller must bind every returned evidence file
into its execution plan and verify those bytes before and after actual scoring.
The pure contract checks also support metadata-only preflight fixtures.
"""
from __future__ import annotations

from pathlib import Path

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha

RESERVATION_PATH = PHASE / "additional-confirmation-reservation-001/plan.json"
RESERVATION_SHA256 = "d5f2b54ca3db512eb01dbaacb342f4536246dbd8ce1298018692704edf63e035"
SELECTION_PATH = PHASE / "additional-primary-selection-001/selection.json"
ACCEPTED_STATE = "c204b0fcb9627ca7fecd287db42fb869a1ae6783a1bc24cf2d8864c3b4a565fb"
TARGET = 4.057715948706591
TRIALS = {
    "leader-cleanup-250": "leader_cleanup",
    "cleanup-successor-250": "cleanup_successor",
    "cleanup-followup-250": "cleanup_followup",
    "cleanup-rebound-250": "cleanup_rebound",
    "cleanup-lr3e6-250": "cleanup_lr3e6",
    "quarter-controlled-250": "quarter_controlled",
}
COMPARISON_AXES = {"full14", "full14_all_track_stem_band_absence", "actions60",
                   "actions60_all_track_stem_band_absence", "probes", "vocal_views"}
REVIEW_AXES = {"full_band_sdr", "low_band_fidelity", "per_stem", "interference",
               "absence", "controlled_views", "actions60", "listening_limitations"}


def add_bindings(evidence, incoming):
    for path, digest in incoming.items():
        require(path not in evidence or evidence[path] == digest, "Conflicting confirmation evidence: " + path)
        evidence[path] = digest


def bound_read(binding, evidence):
    path = Path(binding["path"])
    require(path.is_file() and sha(path) == binding["sha256"], "Confirmation prerequisite changed")
    add_bindings(evidence, {str(path): binding["sha256"]})
    return read(path)


def validate_reservation(binding, evidence):
    require(binding == {"path": str(RESERVATION_PATH), "sha256": RESERVATION_SHA256},
            "Require the prospectively reserved additional windows")
    reservation = bound_read(binding, evidence)
    require(reservation["schema"] == "latency58-additional-within-track-confirmation-reservation-v1"
            and reservation["status"] == "reserved_not_evaluated"
            and reservation["excerpt_starts"] == [45., 120.]
            and reservation["duration_seconds"] == 15.
            and reservation["maximum_selected_candidates"] == 1
            and reservation["comparators"] == ["accepted_8250_model", "original_working_5p8ms_model"]
            and reservation["new_inference"] is False and reservation["new_scores_computed"] is False
            and reservation["new_audio_export_authorized_by_this_plan"] is False,
            "Additional reservation extent or purpose differs")
    manifest = bound_read(reservation["manifest"], evidence)
    config = bound_read(reservation["config"], evidence)
    require(reservation["manifest"]["path"] == str(ROOT / "research/manifests/valid.json")
            and manifest["track_count"] == 14
            and reservation["track_names"] == [track["name"] for track in manifest["tracks"]],
            "Additional reservation track identity differs")
    from research.direct.evaluate import select_panel
    from research.evaluate import _reference_intervals
    from research.direct.latency58_evaluate import plan_latency58_stream
    tracks, excerpts = select_panel(manifest, config, panel="full", excerpt_starts=[45., 120.],
                                   duration=15., alignment_samples=128)
    for track in tracks:
        intervals = _reference_intervals(track, excerpts)
        stream = plan_latency58_stream(intervals, track["frames"])
        require(reservation["track_intervals"][track["name"]]
                == {"reference_intervals": intervals, "frames": track["frames"], "receive_end": stream.receive_end}
                and stream.receive_end <= track["frames"], "Reserved physical sample support changed")
    add_bindings(evidence, reservation["source_bindings"])
    return reservation


def quality_paths(prefix):
    require(prefix in TRIALS, "Unknown reviewed trial")
    full = PHASE / (prefix + "-full14-001")
    summary = PHASE / (prefix + "-summary-001")
    return {"primary_plan": full / "plan.json", "primary_result": full / "result.json",
            "primary_execution": full / "execution.json", "summary_plan": summary / "plan.json",
            "summary_result": summary / "result.json", "summary_execution": summary / "summary-execution.json"}


def validate_quality_contract(prefix, documents, reservation):
    """Check complete, same-endpoint primary and multidimensional evidence."""
    require(prefix in TRIALS, "Unknown reviewed trial")
    primary_plan, primary, execution = (documents[k] for k in
                                       ("primary_plan", "primary_result", "primary_execution"))
    summary_plan, summary, summary_execution = (documents[k] for k in
                                               ("summary_plan", "summary_result", "summary_execution"))
    family = TRIALS[prefix].replace("_", "-")
    require(summary["schema"] == f"latency58-{family}-quality-summary-v1"
            and summary_plan["candidate_prefix"] == prefix and primary_plan["label"] == prefix
            and summary["status"] == "pass" and summary["source_bindings_unchanged"] is True
            and summary["all_metrics_compared"] is True
            and COMPARISON_AXES <= set(summary["comparisons"]["working"])
            and all(summary["comparisons"]["working"][axis] for axis in COMPARISON_AXES),
            "Trial lacks its complete declared quality axes")
    require(primary_plan["mode"] == "full14" and primary_plan["track_indices"] == list(range(14))
            and primary["track_names"] == reservation["track_names"] and primary["excerpt_count"] == 28
            and [excerpt["start_seconds"] for excerpt in primary["excerpts"]] == [30., 75.]
            and all(excerpt["duration_seconds"] == 15. for excerpt in primary["excerpts"])
            and primary["inputs_unchanged"] is True, "Primary selection protocol differs")
    require(all(item["actual_exit_code"] == 0 and item["timed_out"] is False
                and item["source_bindings_unchanged"] is True for item in (execution, summary_execution))
            and execution["plan_sha256"] == primary["plan_sha256"]
            and summary_execution["plan_sha256"] == summary["plan_sha256"],
            "Trial evaluation or summary did not complete successfully")
    fingerprint = primary["results"][0]["model"]["model_state_sha256"]
    aggregate = primary["results"][0]["aggregate"]
    require(summary["training_plan"] == primary_plan["training_plan"]
            and summary["step"] == primary_plan["step"] == 250
            and summary["model_state_sha256"] == fingerprint
            and summary["full_mixture_aggregate"] == aggregate
            and summary["comparisons"]["working"]["full14"]["metrics"]["full_sdr_db"]["candidate"]
            == aggregate["full_sdr_db"], "Primary and full quality bundle refer to different evidence")
    return {"prefix": prefix, "model_state_sha256": fingerprint, "full_sdr_db": aggregate["full_sdr_db"],
            "checkpoint": primary_plan["checkpoint"],
            "evaluation_loader_module": "research.direct.evaluate_latency58_" + TRIALS[prefix]}


def load_quality_bundle(prefix, evidence, reservation):
    paths = quality_paths(prefix)
    require(all(path.is_file() for path in paths.values()), "Trial quality is still incomplete: " + prefix)
    documents = {key: bound_read({"path": str(path), "sha256": sha(path)}, evidence) for key, path in paths.items()}
    result = validate_quality_contract(prefix, documents, reservation)
    require(documents["primary_result"]["plan_sha256"] == evidence[str(paths["primary_plan"])]
            and documents["summary_result"]["plan_sha256"] == evidence[str(paths["summary_plan"])],
            "Quality plan fingerprints differ")
    for key in ("primary_plan", "summary_plan", "summary_result"):
        add_bindings(evidence, documents[key]["source_bindings"])
    result["primary_plan"] = {"path": str(paths["primary_plan"]), "sha256": evidence[str(paths["primary_plan"])]}
    return result


def validate_choice(selection, reviewed):
    """Admit only one recorded primary choice, after all current trials finish."""
    require(set(reviewed) == set(TRIALS) and selection["reviewed_model_states"]
            == {prefix: row["model_state_sha256"] for prefix, row in reviewed.items()},
            "Selection omits a reviewed trial or changes its endpoint")
    require(selection["schema"] == "latency58-additional-primary-selection-v1"
            and selection["status"] == "selected_for_confirmation"
            and selection.get("fixture_not_a_selection", False) is False
            and selection["confirmation_material_used_for_selection"] is False
            and selection["quality_and_probe_review_passed"] is True,
            "Require a real recorded primary selection")
    require(set(selection["quality_review"]) == REVIEW_AXES
            and all(isinstance(text, str) and len(text.strip()) >= 12 for text in selection["quality_review"].values()),
            "Selection must record every declared review axis and listening limitations")
    chosen = reviewed.get(selection["candidate_prefix"])
    accepted = reviewed["leader-cleanup-250"]
    require(chosen is not None and accepted["model_state_sha256"] == ACCEPTED_STATE
            and chosen["model_state_sha256"] == selection["model_state_sha256"]
            and chosen["checkpoint"] == selection["checkpoint"]
            and chosen["primary_plan"] == selection["primary_plan"], "Selected model identity differs")
    require(chosen["full_sdr_db"] >= TARGET and chosen["full_sdr_db"] >= accepted["full_sdr_db"],
            "Selected primary score is below the target or accepted model")
    return chosen


def validate_selection(binding, reservation_binding):
    evidence = {}
    reservation = validate_reservation(reservation_binding, evidence)
    require(Path(binding["path"]) == SELECTION_PATH, "Use the single canonical additional selection")
    selection = bound_read(binding, evidence)
    require(selection["reservation"] == reservation_binding, "Selection used another reservation")
    reviewed = {prefix: load_quality_bundle(prefix, evidence, reservation) for prefix in TRIALS}
    chosen = validate_choice(selection, reviewed)
    add_bindings(evidence, selection["source_bindings"])
    return reservation, selection, chosen, reviewed, evidence


def record_first_use(selection_binding, reservation_binding):
    """Consume the reserved material for this one frozen choice before inference.

    A retained marker permits its comparators or a retry of the same selection.
    A failed attempt still conservatively consumes the material for later
    adaptive choices. This function is never called by metadata preflight.
    """
    from research.direct.run_latency58_quality import write
    from research.direct.train_latency58 import verify_inputs
    reservation, selection, chosen, reviewed, evidence = validate_selection(selection_binding, reservation_binding)
    verify_inputs({"source_bindings": evidence})
    marker = RESERVATION_PATH.parent / "first-use.json"
    expected = {"schema": "latency58-additional-confirmation-first-use-v1", "reservation": reservation_binding,
                "selection": selection_binding, "selected_model_state_sha256": chosen["model_state_sha256"],
                "accepted_model_state_sha256": ACCEPTED_STATE,
                "material_consumed_for_later_adaptive_choices": True,
                "successful_scoring_claimed_by_this_marker": False}
    try:
        write(marker, expected)
    except FileExistsError:
        require(read(marker) == expected, "Additional confirmation material belongs to another frozen selection")
    return {"path": str(marker), "sha256": sha(marker)}
