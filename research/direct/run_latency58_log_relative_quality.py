"""Run bounded quality scoring of one immutable, independently audited generation.

Generation bindings exclude the optimizer: subsequent training may retire its
superseded Adam file while this CPU evaluator reads the retained model.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from research.direct.run_latency58_quality import (
    PHASE, PYTHON, ROOT, checkpoint_bindings, execute, read, require, sha, write,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--training-plan-sha256", required=True)
    parser.add_argument("--stage-directory", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--modes", nargs="+", choices=("full14", "actions60", "probes"), default=["full14"])
    args = parser.parse_args()
    require(Path.cwd() == ROOT and args.output_prefix and all(c.isalnum() or c in "-_" for c in args.output_prefix)
            and len(set(args.modes)) == len(args.modes), "Invalid output prefix, cwd, or duplicate modes")
    require(sha(args.training_plan) == args.training_plan_sha256, "Training plan changed")
    training = read(args.training_plan)
    require(training["schema"] == "latency58-log-relative-training-v1", "Require a context trial plan")
    from research.direct.latency58_log_relative_checkpoint import read_generation, require_space
    require_space(training, 25_000_000 if "actions60" in args.modes else 2_000_000)
    stage = args.stage_directory.resolve(strict=True)
    require(stage.is_relative_to(PHASE), "Use a completed stage in this phase")
    audit_path, audit_exec_path = stage / "audit.json", stage / "audit-execution.json"
    audit, audit_execution, execution = read(audit_path), read(audit_exec_path), read(stage / "execution.json")
    monitor_path = Path(execution["monitor_result"])
    monitor = read(monitor_path)
    generation = Path(audit["generation"])
    require(generation.parent == Path(training["run_dir"]) / "checkpoints", "Generation belongs to another run")
    receipt = read_generation(generation, expected_plan_sha=args.training_plan_sha256, require_optimizer=False)
    require(audit["status"] == "pass" and audit["source_bindings_unchanged"]
            and audit["teacher_kind"] == training["teacher_kind"]
            and audit["step"] == receipt["step"] and receipt["step"] in (250, 500)
            and audit["carry_state"] == receipt["carry_state"] == training["carry_state"]
            and audit["plan_sha256"] == execution["plan_sha256"] == audit_execution["plan_sha256"] == args.training_plan_sha256
            and audit["generation_receipt_sha256"] == sha(generation / "receipt.json")
            and audit["model_state_sha256"] == receipt["model_state_sha256"]
            and audit["checkpoint"]["path"] == str(generation / "model.pt")
            and audit["checkpoint"]["sha256"] == receipt["files"]["model.pt"]["sha256"]
            and execution["actual_exit_code"] == audit_execution["actual_exit_code"] == monitor["child_exit_code"] == 0
            and not audit_execution["timed_out"] and execution["source_bindings_unchanged"]
            and audit_execution["source_bindings_unchanged"]
            and monitor["status"] == monitor["supervisor_health"] == "pass" and monitor["post_exit_quiet_completed"],
            "Require a complete generation with independently audited weights and a healthy GPU exit")
    bindings = {**training["source_bindings"], str(args.training_plan.resolve()): args.training_plan_sha256}
    paths = [audit_path, audit_exec_path, stage / "execution.json", stage / "stage.json", monitor_path,
             generation / "receipt.json", generation / "model.pt", generation / "rng.pt", generation / "metrics.jsonl",
             Path(__file__).resolve(), ROOT / "research/direct/evaluate_latency58_log_relative.py",
             ROOT / "research/direct/compare_latency58_sdr.py", ROOT / "research/direct/compare.py"]
    if "probes" in args.modes:
        paths.append(ROOT / "research/direct/evaluate_latency58_log_relative_probes.py")
    if "full14" in args.modes:
        proof_path = PHASE / "sdr-parallel-baseline-qualification-full14-001/qualification.json"
        proof = read(proof_path)
        require(proof["status"] == "pass" and proof["exact_track_reports"] and proof["exact_stream_metadata"]
                and proof["exact_original_aggregate"] and proof["track_indices"] == list(range(6))
                and all(sha(p) == v for p, v in proof["source_bindings"].items()), "Original parallel qualification changed")
        bindings.update(proof["source_bindings"])
        paths += [proof_path, ROOT / "research/direct/evaluate_latency58_log_relative_parallel.py"]
    bindings.update({str(p): sha(p) for p in paths})
    prepared = []
    for mode in args.modes:
        template_path = PHASE / ("pilot2000-" + mode + "-001") / "plan.json"
        template, template_execution = read(template_path), read(template_path.parent / "execution.json")
        require(template_execution["actual_exit_code"] == 0 and not template_execution["timed_out"]
                and template_execution["source_bindings_unchanged"]
                and template_execution["plan_sha256"] == sha(template_path), "Require a previously executed protocol")
        old = checkpoint_bindings(template["checkpoint"])
        require(all(template["source_bindings"].get(p) == s for p, s in old.items()), "Old checkpoint inventory differs")
        unchanged = {p: s for p, s in template["source_bindings"].items() if p not in old}
        require(all(sha(p) == s for p, s in unchanged.items()), "Preserved evaluation protocol changed")
        out = PHASE / (args.output_prefix + "-" + mode + "-001")
        require(not out.exists(), "Preserve existing evaluation: " + str(out))
        plan = copy.deepcopy(template)
        plan.update(schema="latency58-log-relative-" + ("probe" if mode == "probes" else "music") + "-plan-v1",
                    mode=mode, label=args.output_prefix, step=receipt["step"], generation=str(generation),
                    training_plan={"path": str(args.training_plan.resolve()), "sha256": args.training_plan_sha256},
                    checkpoint=audit["checkpoint"], output_directory=str(out),
                    source_bindings={**unchanged, **bindings, str(template_path): sha(template_path)},
                    protocol_template={"path": str(template_path), "sha256": sha(template_path)})
        if mode == "full14":
            plan.update(schema="latency58-log-relative-parallel-music-plan-v1", workers=2,
                        model_kind="log_relative_candidate", track_indices=list(range(14)),
                        expected_model_state_sha256=receipt["model_state_sha256"],
                        parallel_qualification={"path": str(proof_path), "sha256": sha(proof_path)})
        prepared.append((mode, out, plan))
    for mode, out, plan in prepared:
        out.mkdir()
        plan_path = out / "plan.json"
        write(plan_path, plan)
        module = "research.direct.evaluate_latency58_log_relative" + ("_probes" if mode == "probes" else "_parallel" if mode == "full14" else "")
        argv = [PYTHON, "-u", "-m", module, "--plan", str(plan_path), "--plan-sha256", sha(plan_path)]
        print(json.dumps({"event": "evaluate", "teacher": training["teacher_kind"], "step": receipt["step"],
                          "mode": mode, "plan_sha256": sha(plan_path)}), flush=True)
        execute(argv, out, "evaluation", plan["timeout_seconds"], plan["source_bindings"], {"plan_sha256": sha(plan_path)})
        if mode == "full14":
            from research.direct.compare_latency58_sdr import REFERENCE_PATHS
            source = ROOT / "research/direct/compare_latency58_sdr.py"
            comparison_bindings = {str(p): sha(p) for p in (source, ROOT / "research/direct/compare.py",
                out / "result.json", out / "execution.json", out / "plan.json", *REFERENCE_PATHS.values())}
            argv = [PYTHON, "-u", "-m", "research.direct.compare_latency58_sdr", "--candidate", str(out)]
            execute(argv, out, "comparison", 120, comparison_bindings, {"source_bindings": comparison_bindings})
    print(json.dumps({"event": "quality_bundle_completed", "teacher": training["teacher_kind"],
                      "step": receipt["step"], "quality_selected": False}), flush=True)


if __name__ == "__main__":
    main()
