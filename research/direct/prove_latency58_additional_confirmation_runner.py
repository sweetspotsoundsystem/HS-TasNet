"""Check the new confirmation control flow using already recorded primary data."""
from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.latency58_additional_confirmation_gate import (
    ACCEPTED_STATE, RESERVATION_PATH, RESERVATION_SHA256, SELECTION_PATH,
    quality_paths, validate_quality_contract, validate_reservation,
)
from research.direct import evaluate_latency58_additional_confirmation as evaluator
from research.direct import evaluate_latency58_sdr_confirmation_v5 as previous
from research.direct import evaluate_latency58_sdr_parallel as parallel
from research.direct.report_latency58_additional_confirmation import compare_pair
from research.direct import run_latency58_additional_confirmation as runner


def scoring_loop(path):
    parsed = ast.parse(path.read_text())
    main = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    loop = next(n for n in main.body if isinstance(n, ast.With)
                and "ProcessPoolExecutor" in ast.unparse(n))
    reduction = next(n for n in main.body if isinstance(n, ast.Assign)
                     and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                     and n.value.func.id == "combine_reports")
    return ast.dump(loop, include_attributes=False), ast.dump(reduction, include_attributes=False)


def main():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Use metadata-only CPU preflight")
    out = PHASE / "additional-confirmation-runner-preflight-001"
    require(not out.exists() and not SELECTION_PATH.exists() and not evaluator.ROOT_OUTPUT.exists()
            and not (RESERVATION_PATH.parent / "first-use.json").exists(), "Preserve unconsumed confirmation")
    require(evaluator.initialize_worker is previous.initialize_worker
            and evaluator.score_track is previous.score_track
            and evaluator.combine_reports is parallel.combine_reports
            and evaluator.initialize_worker.__globals__ is evaluator.score_track.__globals__,
            "Original worker functions or shared worker state changed")
    require(scoring_loop(ROOT / "research/direct/evaluate_latency58_additional_confirmation.py")
            == scoring_loop(ROOT / "research/direct/evaluate_latency58_sdr_confirmation_v5.py"),
            "Parallel scoring loop or aggregation changed")
    reservation_binding = {"path": str(RESERVATION_PATH), "sha256": RESERVATION_SHA256}
    reservation = validate_reservation(reservation_binding, {})
    working_path = PHASE / "teacher-half250-full14-001/result.json"
    candidate_path = PHASE / "leader-cleanup-250-full14-001/result.json"
    summary_path = PHASE / "leader-cleanup-250-summary-001/result.json"
    working, candidate, summary = (read(p) for p in (working_path, candidate_path, summary_path))
    repeated = compare_pair(working, candidate)
    require(repeated["comparison"] == summary["comparisons"]["working"]["full14"]
            and repeated["all_track_stem_band_absence"]
            == summary["comparisons"]["working"]["full14_all_track_stem_band_absence"],
            "Existing primary comparison did not reproduce exactly")
    identity = compare_pair(candidate, candidate)
    require(all(item["delta"] == 0 and item["paired_track_bootstrap_95_percent"] == [0., 0.]
                for item in identity["comparison"]["metrics"].values()),
            "Identical accepted/candidate report did not produce exact zero differences")

    # Only plan routing is substituted here. The real authentication function
    # is restored before exercising the actual preparation command below.
    metadata_paths = []
    models = {}
    for prefix in ("leader-cleanup-250", "cleanup-rebound-250"):
        paths = quality_paths(prefix)
        documents = {key: read(path) for key, path in paths.items()}
        metadata_paths.extend(paths.values())
        models[prefix] = validate_quality_contract(prefix, documents, reservation)
        models[prefix]["primary_plan"] = {"path": str(paths["primary_plan"]), "sha256": sha(paths["primary_plan"])}
    rejected, routing_passes = [], 0
    def reject(label, action):
        try:
            action()
        except (RuntimeError, KeyError) as error:
            rejected.append({"case": label, "error": str(error)})
        else:
            raise RuntimeError("Invalid routing request was admitted: " + label)
    authenticate = evaluator.authenticate_selection
    try:
        for prefix, chosen in models.items():
            fixture_selection = {"model_state_sha256": chosen["model_state_sha256"]}
            evidence = {str(candidate_path): sha(candidate_path)}
            evaluator.authenticate_selection = lambda *args: (reservation, fixture_selection, chosen, models, evidence)
            directories = evaluator.expected_directories(chosen)
            common = {"schema": evaluator.SCHEMA, "workers": 2, "track_indices": list(range(14)),
                      "reservation": reservation_binding, "selection": {"path": "routing-fixture-only", "sha256": "0" * 64},
                      "source_bindings": evidence, "counted_roots": reservation["counted_roots"],
                      "stop_counted_bytes": 79_500_000_000, "maximum_confirmation_report_bytes": 12_000_000,
                      "evaluation_directories": directories}
            roles = ["working", "candidate"] + ([] if chosen["model_state_sha256"] == ACCEPTED_STATE else ["accepted"])
            for role in roles:
                plan = {**common, "role": role, "model_kind": evaluator.ROLE_KINDS[role],
                        "output_directory": directories[role], "label": "additional-confirmation-" + role}
                if role == "working":
                    plan["expected_model_state_sha256"] = evaluator.STUDENT_STATE_SHA256
                else:
                    model = chosen if role == "candidate" else models["leader-cleanup-250"]
                    plan.update(expected_model_state_sha256=model["model_state_sha256"],
                                selected_primary_plan=model["primary_plan"],
                                evaluation_loader_module=model["evaluation_loader_module"])
                evaluator.validate_plan(plan)
                routing_passes += 1
                changes = {
                    "wrong_model_fingerprint": lambda p: p.update(expected_model_state_sha256="0" * 64),
                    "incomplete_track_coverage": lambda p: p.update(track_indices=[0]),
                    "changed_role_kind": lambda p: p.update(model_kind="unreviewed"),
                    "changed_output_location": lambda p: p.update(output_directory=str(PHASE / "unreviewed-output")),
                    "changed_artifact_budget": lambda p: p.update(maximum_confirmation_report_bytes=80_000_000_000),
                    "omitted_authentication_evidence": lambda p: p.update(source_bindings={}),
                }
                for name, change in changes.items():
                    corrupt = copy.deepcopy(plan)
                    change(corrupt)
                    reject(prefix + ":" + role + ":" + name, lambda: evaluator.validate_plan(corrupt))
                if role != "working":
                    corrupt = copy.deepcopy(plan)
                    corrupt["evaluation_loader_module"] = "research.direct.evaluate_latency58_unreviewed"
                    reject(prefix + ":" + role + ":unreviewed_loader", lambda: evaluator.validate_plan(corrupt))
    finally:
        evaluator.authenticate_selection = authenticate
    reject("actual_preparation_requires_uncompleted_quality", lambda: runner.prepare(
        "leader-cleanup-250", out / "not-an-authored-review.json"))
    require(evaluator.authenticate_selection is authenticate and not SELECTION_PATH.exists()
            and not evaluator.ROOT_OUTPUT.exists() and not (RESERVATION_PATH.parent / "first-use.json").exists(),
            "Preflight altered authentication, selected a model or consumed confirmation")
    import torch
    require(not torch.cuda.is_initialized(), "Preflight initialized CUDA")
    paths = [ROOT / path for path in runner.SOURCES] + metadata_paths + [RESERVATION_PATH,
             Path(reservation["manifest"]["path"]), Path(reservation["config"]["path"]), working_path,
             candidate_path, summary_path, PHASE / "additional-confirmation-gate-preflight-001/result.json"]
    evidence = {str(path): sha(path) for path in paths}
    out.mkdir()
    plan = {"schema": "latency58-additional-confirmation-runner-preflight-plan-v1",
            "reservation": reservation_binding, "source_bindings": evidence,
            "scope": "Original worker identity, AST comparison, stored primary metrics and routing fixtures; no reserved scoring."}
    write(out / "plan.json", plan)
    result = {"schema": "latency58-additional-confirmation-runner-preflight-result-v1", "status": "pass",
              "plan_sha256": sha(out / "plan.json"), "source_bindings": evidence, "source_bindings_unchanged": True,
              "worker_functions_reused": True, "unchanged_scoring_loop": True,
              "stored_primary_comparison_reproduced": True, "identical_candidate_reference_zero_delta": True,
              "routing_contract_fixtures_passed": routing_passes, "rejected_cases": rejected,
              "actual_choice_authenticated_by_routing_fixtures": False,
              "source_audio_rehashed": False, "new_audio_decoded": False, "model_instances": 0,
              "cuda_initialized": False, "reserved_windows_scored": False,
              "quality_selected": False, "reservation_consumed": False,
              "limitations": ["Actual positive selection still requires the completed remaining quality bundles.",
                              "No model was loaded and no confirmation inference was run by this preflight."]}
    require(all(sha(path) == digest for path, digest in evidence.items()), "Preflight source changed")
    write(out / "result.json", result)
    print(json.dumps({"status": "pass", "routing_fixtures": routing_passes, "rejected_cases": len(rejected),
                      "stored_primary_comparison_reproduced": True, "reserved_windows_scored": False}), flush=True)


if __name__ == "__main__":
    main()
