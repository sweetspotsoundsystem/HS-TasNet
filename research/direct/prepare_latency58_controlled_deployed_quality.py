"""Reserve unchanged quality work against the live controlled deployed-truth supervisor."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_controlled_deployed_checkpoint import validate_recipe, require_space
from research.direct.report_latency58_sdr import load_completed
from research.direct.report_latency58_controlled_deployed import REFERENCES, reference_views, load_views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--training-plan-sha256", required=True)
    parser.add_argument("--supervisor-pid", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.training_plan) == args.training_plan_sha256, "Changed training input")
    training = read(args.training_plan)
    validate_recipe(training)
    verify_inputs(training)
    require(not training["resource_only"] and training["teacher_mode"] == "ordinary_only", "Production only")
    require(training["additional_loss_weight"] == .5, "Different added supervision coefficient")
    out = args.output_directory
    require(out.parent == PHASE and out.is_dir() and not (out / "result.json").exists(), "Preserve preparation")
    prefix = "controlled-deployed-half-250"
    stage = PHASE / "controlled-deployed-half-to-000250-001"
    control_stage = PHASE / "counterfactual-teacher-ordinary-only-to-000250-001"
    queue = PHASE / (prefix + "-queued-quality-001")
    match = PHASE / (prefix + "-training-match-001")
    directories = [queue, match, *(PHASE / (prefix + "-" + mode + "-001")
                                  for mode in ("full14", "actions60", "probes", "views", "summary"))]
    require(all(not path.exists() for path in directories), "Preserve existing candidate evidence")
    proc = Path(f"/proc/{args.supervisor_pid}")
    stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
    argv = [part.decode() for part in (proc / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")]
    require(argv[argv.index("-m") + 1] == "research.direct.run_latency58_controlled_deployed_stage"
            and argv[argv.index("--plan") + 1] == str(args.training_plan)
            and argv[argv.index("--plan-sha256") + 1] == args.training_plan_sha256
            and argv[argv.index("--stop-step") + 1] == "250"
            and argv[argv.index("--output-directory") + 1] == str(stage), "Different live training supervisor")
    identity = {"pid": args.supervisor_pid, "start_ticks": int(stat[19]), "argv": argv}
    evidence = {**training["source_bindings"], str(args.training_plan): args.training_plan_sha256}
    # These source transformations change only checkpoint family, labels and schemas.
    # All streaming, source reads, metrics and aggregation remain byte-for-byte equal.
    for suffix in ("", "_parallel", "_probes", "_views"):
        old = ROOT / ("research/direct/evaluate_latency58_counterfactual" + suffix + ".py")
        new = ROOT / ("research/direct/evaluate_latency58_controlled_deployed" + suffix + ".py")
        require(old.read_text().replace("counterfactual", "controlled_deployed").replace("latency58-controlled_deployed", "latency58-controlled-deployed")
                == new.read_text(), "Evaluation changed beyond checkpoint loading and labels")
        evidence.update({str(p): sha(p) for p in (old, new)})
    modules = ("audit_latency58_controlled_deployed_training_match", "run_latency58_controlled_deployed_quality",
               "run_latency58_controlled_deployed_views", "report_latency58_controlled_deployed",
               "queue_latency58_controlled_deployed_quality", "prepare_latency58_controlled_deployed_quality",
               "report_latency58_sdr", "report_latency58_vocal_focus", "compare_latency58_vocal_views",
               "compare_latency58_sdr", "compare", "run_latency58_quality")
    for name in modules:
        path = ROOT / ("research/direct/" + name + ".py")
        ast.parse(path.read_text(), filename=str(path))
        evidence[str(path)] = sha(path)
    states = {}
    for label, reference_prefix in REFERENCES.items():
        fingerprints = set()
        for mode in ("full14", "actions60", "probes"):
            quality_plan, report = load_completed(PHASE / (reference_prefix + "-" + mode + "-001"),
                                                 evidence, canonical_baseline=label == "working")
            if label == "ordinary_control":
                require(quality_plan["training_plan"] == training["matched_ordinary_training_plan"],
                        "Quality control differs from the matched training recipe")
            fingerprints.add(report["model_state_sha256"] if mode == "probes"
                             else report["results"][0]["model"]["model_state_sha256"])
        fingerprints.add(load_views(reference_views(label), evidence)["model"]["model_state_sha256"])
        require(len(fingerprints) == 1, "Reference scores contain different models")
        states[label] = next(iter(fingerprints))
    require(states["working"] == training["parent"]["model_state_sha256"], "Different rollback reference")
    for path in (stage / "stage.json", stage / "watchdog-spec.json", stage / "command.json",
                 control_stage / "audit.json", control_stage / "audit-execution.json", control_stage / "execution.json"):
        evidence[str(path)] = sha(path)
    verify_inputs({"source_bindings": evidence})
    before = require_space(training, 450_000_000)
    reservation = {"schema": "latency58-controlled-deployed-views-reservation-v1",
                   "new_artifact_allowance_bytes": 10_000_000, "training_reserve_bytes": 350_000_000,
                   "counted_bytes_at_preparation": before, "counted_roots": training["counted_roots"],
                   "stop_counted_bytes": training["stop_counted_bytes"],
                   "evaluation_directories": [str(PHASE / (prefix + "-views-001"))], "source_bindings": evidence}
    write(out / "views-reservation.json", reservation)
    reserve_binding = {"path": str(out / "views-reservation.json"), "sha256": sha(out / "views-reservation.json")}
    queue_plan = {"schema": "latency58-controlled-deployed-queued-quality-plan-v1", "quality_prefix": prefix,
                  "training_plan": {"path": str(args.training_plan), "sha256": args.training_plan_sha256},
                  "stage_directory": str(stage), "control_stage_directory": str(control_stage),
                  "output_directory": str(queue), "training_match_directory": str(match),
                  "maximum_wait_seconds": 9000, "launch_next_training_arm": False, "supervisor": identity,
                  "views_reservation": reserve_binding,
                  "source_bindings": {**evidence, reserve_binding["path"]: reserve_binding["sha256"]}}
    queue.mkdir()
    write(queue / "plan.json", queue_plan)
    write(out / "result.json", {"schema": "latency58-controlled-deployed-quality-preparation-v1", "status": "pass",
          "source_bindings": evidence, "source_bindings_unchanged": True, "supervisor": identity,
          "queue_plan": {"path": str(queue / "plan.json"), "sha256": sha(queue / "plan.json")},
          "evaluation_math_and_streaming_unchanged": True, "reference_model_states": states,
          "training_updates_executed": 0, "quality_selected": False,
          "counted_bytes_before": before, "counted_bytes_after": require_space(training, 450_000_000)})
    print({"status": "pass", "queue_plan": str(queue / "plan.json"), "source_bindings": len(evidence)}, flush=True)


if __name__ == "__main__":
    main()
