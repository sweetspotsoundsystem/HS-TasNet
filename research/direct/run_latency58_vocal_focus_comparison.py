"""Run final training authentication or the two prescribed pilot comparisons."""
from __future__ import annotations

import argparse
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, ROOT, PYTHON, read, require, sha, write, execute
from research.direct.train_latency58 import verify_inputs
from research.direct.audit_latency58_vocal_focus_match import ARMS
from research.direct.latency58_vocal_focus_checkpoint import require_space
from research.direct.report_latency58_sdr import load_completed
from research.direct.report_latency58_vocal_focus import load_views
from research.direct.record_latency58_vocal_focus_review import completed_endpoint
from research.direct.capture_latency58_vocal_focus_skelpolu import completed


def binding(path):
    return {"path": str(path), "sha256": sha(path)}


def merge_sources(target, added):
    require(all(path not in target or target[path] == digest for path, digest in added.items()),
            "Different hashes for the same comparison input")
    target.update(added)


def training_evidence(arm):
    """Read completed production evidence without requiring a retired Adam file."""
    require(arm in ARMS, "Different pilot arm")
    training_path = PHASE / "vocal-focus-prep-001" / (arm + "-training-plan.json")
    stage = PHASE / ("vocal-focus-" + arm.replace("_", "-") + "-to-000250-001")
    training = read(training_path)
    require(training["arm"] == arm and not training["resource_only"] and training["config"]["steps"] == 250,
            "Different training endpoint")
    verify_inputs(training)
    spec = {"training_plan": binding(training_path), **{
        key: binding(stage / name) for key, name in (
            ("audit", "audit.json"), ("audit_execution", "audit-execution.json"), ("execution", "execution.json"))}}
    execution, audited, audit = (read(spec[key]["path"]) for key in ("execution", "audit_execution", "audit"))
    require(execution["actual_exit_code"] == audited["actual_exit_code"] == 0
            and execution["source_bindings_unchanged"] and audited["source_bindings_unchanged"]
            and not audited["timed_out"] and audit["status"] == "pass" and audit["step"] == 250
            and audit["arm"] == arm and audit["source_bindings_unchanged"]
            and execution["plan_sha256"] == audited["plan_sha256"] == audit["plan_sha256"] == spec["training_plan"]["sha256"],
            "Production training or its original saved-state audit is incomplete")
    sources = dict(training["source_bindings"])
    generation = Path(training["run_dir"]) / "checkpoints/step-000250"
    paths = [Path(item["path"]) for item in spec.values()]
    paths += [Path(execution["monitor_result"]), Path(training["run_dir"]) / "status.json"]
    paths += [generation / name for name in ("receipt.json", "model.pt", "rng.pt", "metrics.jsonl")]
    merge_sources(sources, {str(path): sha(path) for path in paths})
    return training, spec, sources


def run_plan(out, plan, module, label, timeout):
    require(not out.exists(), "Preserve existing comparison evidence")
    verify_inputs(plan)
    require_space(plan, 3_000_000)
    out.mkdir()
    path = out / "plan.json"
    write(path, plan)
    execute([PYTHON, "-u", "-m", module, "--plan", str(path), "--plan-sha256", sha(path)],
            out, label, timeout, plan["source_bindings"], {"plan_sha256": sha(path)})
    return out / "result.json", out / (label + "-execution.json")


def run_match(suffix):
    sources, arms, plans = {}, {}, {}
    for arm in ARMS:
        plans[arm], arms[arm], evidence = training_evidence(arm)
        merge_sources(sources, evidence)
    for name in ("run_latency58_vocal_focus_comparison.py", "audit_latency58_vocal_focus_training_match.py",
                 "audit_latency58_vocal_focus_match.py", "run_latency58_quality.py"):
        path = ROOT / "research/direct" / name
        merge_sources(sources, {str(path): sha(path)})
    space = {key: plans["original"][key] for key in ("counted_roots", "stop_counted_bytes")}
    require(all(all(plan[key] == value for key, value in space.items()) for plan in plans.values()),
            "Pilot storage boundaries differ")
    out = PHASE / ("vocal-focus-training-match-250-" + suffix)
    plan = {"schema": "latency58-vocal-focus-training-match-plan-v1", "arms": arms,
            "output_directory": str(out), "source_bindings": sources, **space}
    result, execution = run_plan(out, plan, "research.direct.audit_latency58_vocal_focus_training_match", "match", 240)
    print({"event": "full_training_match_complete", "result": binding(result), "execution": binding(execution)}, flush=True)


def run_comparisons(match_directory, suffix):
    match_directory = match_directory.resolve(strict=True)
    require(match_directory.parent == PHASE, "Use the final matching audit from this phase")
    result_path, execution_path = match_directory / "result.json", match_directory / "match-execution.json"
    match_plan_path = match_directory / "plan.json"
    match_plan, match = read(match_plan_path), read(result_path)
    sources = dict(match_plan["source_bindings"])
    merge_sources(sources, match["source_bindings"])
    merge_sources(sources, {str(path): sha(path) for path in (match_plan_path, result_path, execution_path)})
    match_items = {"match_audit": binding(result_path), "match_execution": binding(execution_path),
                   "match_plan": binding(match_plan_path)}
    completed({"source_bindings": sources}, match_items, "match_audit", "match_execution", "match_plan",
              "research.direct.audit_latency58_vocal_focus_training_match")
    require(match["schema"] == "latency58-vocal-focus-training-match-v1" and match["status"] == "pass"
            and match["updates_per_arm"] == 250 and match["final_rng_states_exact"], "Incomplete final training match")
    for arm in ARMS:
        prefix = "vocal-focus-" + arm.replace("_", "-") + "-250"
        directory = PHASE / (prefix + "-review-001")
        review_plan_path = directory / "plan.json"
        review_plan, review = read(review_plan_path), read(directory / "result.json")
        merge_sources(sources, review_plan["source_bindings"])
        merge_sources(sources, review["source_bindings"])
        review_items = {key: binding(directory / name) for key, name in (
            ("review", "result.json"), ("review_plan", "plan.json"), ("review_execution", "review-execution.json"))}
        merge_sources(sources, {item["path"]: item["sha256"] for item in review_items.values()})
        completed({"source_bindings": sources}, review_items, "review", "review_execution", "review_plan",
                  "research.direct.record_latency58_vocal_focus_review")
        _, _, receipt, _ = completed_endpoint(review_plan)
        require(review["training_closed"] and review["all_quality_metrics_reviewed"]
                and review["review"] == review_plan["review"] and review["arm"] == arm
                and review["training_plan"] == match["training_plans"][arm]
                and review["model_state_sha256"] == receipt["model_state_sha256"] == match["trained_model_states"][arm],
                "Different or unreviewed pilot endpoint")
        evidence = {}
        for mode in ("full14", "actions60", "probes"):
            load_completed(PHASE / (prefix + "-" + mode + "-001"), evidence)
        load_views(PHASE / (prefix + "-views-001"), evidence)
        merge_sources(sources, evidence)
    for name in ("run_latency58_vocal_focus_comparison.py", "compare_latency58_vocal_focus.py",
                 "report_latency58_vocal_focus.py", "compare_latency58_vocal_views.py", "run_latency58_quality.py",
                 "record_latency58_vocal_focus_review.py", "capture_latency58_vocal_focus_skelpolu.py"):
        path = ROOT / "research/direct" / name
        merge_sources(sources, {str(path): sha(path)})
    prepared = []
    for reference, candidate in (("original", "focused"), ("focused", "focused_mixer")):
        out = PHASE / ("vocal-focus-" + candidate.replace("_", "-") + "-versus-" + reference + "-250-" + suffix)
        require(not out.exists(), "Preserve existing endpoint comparison")
        plan = {"schema": "latency58-vocal-focus-comparison-plan-v1", "output_directory": str(out),
                "source_bindings": dict(sources), **match_items,
                **{key: match_plan[key] for key in ("counted_roots", "stop_counted_bytes")}}
        for side, arm in (("reference", reference), ("candidate", candidate)):
            plan.update({side + "_arm": arm, side + "_prefix": "vocal-focus-" + arm.replace("_", "-") + "-250",
                         side + "_step": 250, side + "_training_plan": match["training_plans"][arm],
                         side + "_model_state_sha256": match["trained_model_states"][arm]})
        prepared.append((out, plan))
    for out, plan in prepared:
        run_plan(out, plan, "research.direct.compare_latency58_vocal_focus", "comparison", 300)
    print({"event": "both_pilot_comparisons_complete", "quality_selected": False}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("training-match", "quality-comparisons"))
    parser.add_argument("--match-directory", type=Path)
    parser.add_argument("--suffix", default="001")
    args = parser.parse_args()
    require(Path.cwd() == ROOT and len(args.suffix) == 3 and args.suffix.isdecimal(), "Different cwd or suffix")
    if args.operation == "training-match":
        require(args.match_directory is None, "Training match creates its own output")
        run_match(args.suffix)
    else:
        require(args.match_directory is not None, "Quality comparisons require the completed training match")
        run_comparisons(args.match_directory, args.suffix)


if __name__ == "__main__":
    main()
