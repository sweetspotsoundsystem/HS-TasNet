"""Freeze one reviewed choice, then run bounded additional-window comparisons.

Preparation requires all six current quality bundles. Execution requires the
prepared plan hash. Neither command can silently choose another candidate
after the reservation has been consumed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, PYTHON, ROOT, execute, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_additional_confirmation_gate import (
    RESERVATION_PATH, RESERVATION_SHA256, SELECTION_PATH, TRIALS, add_bindings,
    load_quality_bundle, validate_choice, validate_reservation, validate_selection,
)
from research.direct.evaluate_latency58_additional_confirmation import (
    ROOT_OUTPUT, ROLE_KINDS, SCHEMA, expected_directories, require_capacity, validate_plan,
)
from research.direct.latency58_sdr_teacher import STUDENT_STATE_SHA256

PREFLIGHT = PHASE / "additional-confirmation-runner-preflight-001/result.json"
SOURCES = ["research/direct/latency58_additional_confirmation_gate.py",
           "research/direct/evaluate_latency58_additional_confirmation.py",
           "research/direct/report_latency58_additional_confirmation.py",
           "research/direct/run_latency58_additional_confirmation.py",
           "research/direct/prove_latency58_additional_confirmation_runner.py",
           "research/direct/evaluate_latency58_sdr_confirmation_v5.py",
           "research/direct/evaluate_latency58_sdr_parallel.py",
           "research/direct/report_latency58_confirmation.py",
           "research/direct/report_latency58_sdr.py", "research/direct/report_latency58_vocal_focus.py",
           "research/direct/compare.py", "research/direct/run_latency58_quality.py",
           "research/direct/train_latency58.py", "research/direct/latency58_sdr_teacher.py"]


def bind(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha(path)}


def prepare(candidate_prefix, review_path):
    require(candidate_prefix in TRIALS and not SELECTION_PATH.parent.exists() and not ROOT_OUTPUT.exists()
            and not (RESERVATION_PATH.parent / "first-use.json").exists(),
            "Preserve the single frozen selection and reserved material")
    evidence = {}
    reservation_binding = {"path": str(RESERVATION_PATH), "sha256": RESERVATION_SHA256}
    reservation = validate_reservation(reservation_binding, evidence)
    reviewed = {prefix: load_quality_bundle(prefix, evidence, reservation) for prefix in TRIALS}
    require(PREFLIGHT.is_file(), "Complete runner preflight before preparing confirmation")
    preflight = read(PREFLIGHT)
    require(preflight["status"] == "pass" and preflight["worker_functions_reused"]
            and preflight["unchanged_scoring_loop"] and preflight["stored_primary_comparison_reproduced"]
            and preflight["reservation_consumed"] is False, "Runner lacks complete metadata/arithmetic preflight")
    verify_inputs(preflight)
    add_bindings(evidence, preflight["source_bindings"])
    add_bindings(evidence, {str(PREFLIGHT): sha(PREFLIGHT), **{str(ROOT / p): sha(ROOT / p) for p in SOURCES}})
    review_binding = bind(review_path)
    add_bindings(evidence, {review_binding["path"]: review_binding["sha256"]})
    chosen = reviewed[candidate_prefix]
    choice = {"schema": "latency58-additional-primary-selection-v1", "status": "selected_for_confirmation",
              "reservation": reservation_binding, "candidate_prefix": candidate_prefix,
              "model_state_sha256": chosen["model_state_sha256"], "checkpoint": chosen["checkpoint"],
              "primary_plan": chosen["primary_plan"],
              "reviewed_model_states": {prefix: value["model_state_sha256"] for prefix, value in reviewed.items()},
              "confirmation_material_used_for_selection": False, "quality_and_probe_review_passed": True,
              "quality_review": read(review_path), "quality_review_source": review_binding,
              "source_bindings": evidence}
    validate_choice(choice, reviewed)
    verify_inputs(choice)
    directories = expected_directories(chosen)
    require(all(not Path(path).exists() for path in set(directories.values())), "Preserve prior comparator outputs")
    common = {"schema": SCHEMA, "reservation": reservation_binding, "workers": 2,
              "track_indices": list(range(14)), "counted_roots": reservation["counted_roots"],
              "stop_counted_bytes": reservation["counted_stop_bytes"],
              "maximum_confirmation_report_bytes": reservation["maximum_confirmation_report_bytes"],
              "evaluation_directories": directories}
    require_capacity(common, 6_000_000)
    SELECTION_PATH.parent.mkdir()
    write(SELECTION_PATH, choice)
    selection_binding = bind(SELECTION_PATH)
    _, _, selected, _, selected_evidence = validate_selection(selection_binding, reservation_binding)
    require(selected == chosen, "Selection readback differs")
    add_bindings(evidence, selected_evidence)
    common.update(selection=selection_binding, source_bindings=dict(evidence))
    ROOT_OUTPUT.mkdir()
    roles = ["working", "candidate"]
    if directories["accepted"] != directories["candidate"]:
        roles.append("accepted")
    plans = {}
    for role in roles:
        out = Path(directories[role])
        out.mkdir()
        plan = {**common, "role": role, "model_kind": ROLE_KINDS[role],
                "output_directory": str(out), "label": "additional-confirmation-" + role}
        if role == "working":
            plan["expected_model_state_sha256"] = STUDENT_STATE_SHA256
        else:
            model = chosen if role == "candidate" else reviewed["leader-cleanup-250"]
            plan.update(expected_model_state_sha256=model["model_state_sha256"],
                        selected_primary_plan=model["primary_plan"],
                        evaluation_loader_module=model["evaluation_loader_module"])
        validate_plan(plan)
        write(out / "plan.json", plan)
        plans[role] = bind(out / "plan.json")
        add_bindings(evidence, {plans[role]["path"]: plans[role]["sha256"]})
    queue = {"schema": "latency58-additional-confirmation-queue-plan-v1", "selection": selection_binding,
             "reservation": reservation_binding, "models": plans, "evaluation_directories": directories,
             "counted_roots": common["counted_roots"], "stop_counted_bytes": common["stop_counted_bytes"],
             "maximum_confirmation_report_bytes": common["maximum_confirmation_report_bytes"],
             "source_bindings": evidence, "maximum_seconds_per_evaluation": 5400,
             "maximum_seconds_for_summary": 1800, "new_audio_export": False}
    require_capacity(queue, len((json.dumps(queue, indent=2, allow_nan=False) + "\n").encode()))
    write(ROOT_OUTPUT / "plan.json", queue)
    require(not (RESERVATION_PATH.parent / "first-use.json").exists(), "Preparation consumed confirmation material")
    print(json.dumps({"status": "prepared_not_evaluated", "plan": bind(ROOT_OUTPUT / "plan.json"),
                      "selected_model_state_sha256": chosen["model_state_sha256"], "executed_comparators": 0}), flush=True)


def run(plan_path, plan_sha256):
    require(plan_path.resolve() == ROOT_OUTPUT / "plan.json" and sha(plan_path) == plan_sha256,
            "Use the frozen confirmation queue")
    plan = read(plan_path)
    require(plan["schema"] == "latency58-additional-confirmation-queue-plan-v1"
            and plan["maximum_seconds_per_evaluation"] == 5400 and plan["maximum_seconds_for_summary"] == 1800
            and plan["new_audio_export"] is False, "Confirmation queue scope differs")
    verify_inputs(plan)
    _, _, chosen, _, evidence = validate_selection(plan["selection"], plan["reservation"])
    require(all(plan["source_bindings"].get(p) == digest for p, digest in evidence.items())
            and plan["evaluation_directories"] == expected_directories(chosen), "Queue omits selected evidence")
    roles = ["working", "candidate"]
    if plan["evaluation_directories"]["accepted"] != plan["evaluation_directories"]["candidate"]:
        roles.append("accepted")
    require(set(plan["models"]) == set(roles), "Missing or additional comparator")
    for role in roles:
        binding = plan["models"][role]
        require(binding == bind(Path(plan["evaluation_directories"][role]) / "plan.json"), "Comparator plan changed")
        model_plan = read(binding["path"])
        validate_plan(model_plan)
        require(model_plan["selection"] == plan["selection"] and model_plan["reservation"] == plan["reservation"]
                and model_plan["role"] == role, "Comparator belongs to another queue")
    require_capacity(plan, 6_000_000)
    # Exclusive start recording prevents an observation timeout or a second
    # invocation from accidentally launching overlapping copies of this queue.
    write(ROOT_OUTPUT / "run-start.json", {"plan_sha256": plan_sha256, "pid": os.getpid(),
          "process_start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().split(")", 1)[1].split()[19]})
    for role in roles:
        binding = plan["models"][role]
        model_plan = read(binding["path"])
        execute([PYTHON, "-u", "-m", "research.direct.evaluate_latency58_additional_confirmation",
                 "--plan", binding["path"], "--plan-sha256", binding["sha256"]],
                Path(model_plan["output_directory"]), "evaluation", 5400, model_plan["source_bindings"],
                {"plan_sha256": binding["sha256"], "queue_plan_sha256": plan_sha256, "role": role})
    from research.direct.report_latency58_additional_confirmation import load_report
    completed_evidence = dict(plan["source_bindings"])
    for role in roles:
        load_report(role, Path(plan["evaluation_directories"][role]), completed_evidence)
    completed_evidence[str(plan_path.resolve())] = plan_sha256
    summary_plan = {"schema": "latency58-additional-confirmation-summary-plan-v1",
                    "selection": plan["selection"], "reservation": plan["reservation"],
                    "evaluations": plan["evaluation_directories"], "output_directory": str(ROOT_OUTPUT),
                    "source_bindings": completed_evidence}
    summary_path = ROOT_OUTPUT / "summary-plan.json"
    require_capacity(plan, len((json.dumps(summary_plan, indent=2, allow_nan=False) + "\n").encode()))
    write(summary_path, summary_plan)
    execute([PYTHON, "-u", "-m", "research.direct.report_latency58_additional_confirmation",
             "--plan", str(summary_path), "--plan-sha256", sha(summary_path)], ROOT_OUTPUT, "summary", 1800,
            completed_evidence, {"plan_sha256": sha(summary_path), "queue_plan_sha256": plan_sha256})
    verify_inputs(plan)
    storage = require_capacity(plan, 10_000)
    write(ROOT_OUTPUT / "queue-result.json", {"actual_exit_code": 0, "source_bindings_unchanged": True,
          "queue_plan_sha256": plan_sha256, "summary": bind(ROOT_OUTPUT / "result.json"), "storage_after": storage})
    print(json.dumps({"status": "additional_confirmation_complete", "summary": bind(ROOT_OUTPUT / "result.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--candidate-prefix", choices=TRIALS, required=True)
    prepare_parser.add_argument("--quality-review", type=Path, required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--plan", type=Path, required=True)
    run_parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Use CUDA-hidden confirmation control")
    if args.command == "prepare":
        prepare(args.candidate_prefix, args.quality_review)
    else:
        run(args.plan, args.plan_sha256)


if __name__ == "__main__":
    main()
