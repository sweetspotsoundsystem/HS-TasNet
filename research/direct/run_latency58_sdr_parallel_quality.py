"""Qualify unchanged per-track scoring, then run bounded two-process panels."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from research.direct.run_latency58_quality import (
    PHASE, PYTHON, ROOT, checkpoint_bindings, execute, read, require, sha, write,
)


def protocol_template():
    path = PHASE / "pilot2000-full14-001/plan.json"
    plan, execution = read(path), read(path.parent / "execution.json")
    require(execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"] and execution["plan_sha256"] == sha(path),
            "Require a successfully executed serial protocol")
    old = checkpoint_bindings(plan["checkpoint"])
    require(all(plan["source_bindings"].get(p) == s for p, s in old.items()), "Old checkpoint inventory differs")
    plan = copy.deepcopy(plan)
    plan.pop("checkpoint")
    plan["source_bindings"] = {p: s for p, s in plan["source_bindings"].items() if p not in old}
    require(all(sha(p) == s for p, s in plan["source_bindings"].items()), "Preserved protocol changed")
    plan["protocol_template"] = {"path": str(path), "sha256": sha(path)}
    for p in (path, path.parent / "execution.json", Path(__file__).resolve(),
              ROOT / "research/direct/evaluate_latency58_sdr_parallel.py"):
        plan["source_bindings"][str(p)] = sha(p)
    return plan


def candidate_bindings(args):
    from research.direct.latency58_sdr_checkpoint import read_generation, require_space
    require(sha(args.training_plan) == args.training_plan_sha256, "Training plan changed")
    training = read(args.training_plan)
    require(training["schema"] == "latency58-sdr-training-v1", "Require an SDR training plan")
    require_space(training, 2_000_000)
    stage = args.stage_directory.resolve(strict=True)
    require(stage.is_relative_to(PHASE), "Use a completed stage in this phase")
    audit_path, audit_exec_path = stage / "audit.json", stage / "audit-execution.json"
    audit, audit_exec, execution = read(audit_path), read(audit_exec_path), read(stage / "execution.json")
    monitor_path = Path(execution["monitor_result"])
    monitor = read(monitor_path)
    generation = Path(audit["generation"])
    require(generation.parent == Path(training["run_dir"]) / "checkpoints", "Wrong training run")
    receipt = read_generation(generation, expected_plan_sha=args.training_plan_sha256, require_optimizer=False)
    require(audit["status"] == "pass" and audit["source_bindings_unchanged"]
            and audit["teacher_kind"] == training["teacher_kind"]
            and audit["step"] == receipt["step"] and receipt["step"] in (250, 500, 1000)
            and audit["plan_sha256"] == execution["plan_sha256"] == audit_exec["plan_sha256"] == args.training_plan_sha256
            and audit["generation_receipt_sha256"] == sha(generation / "receipt.json")
            and audit["model_state_sha256"] == receipt["model_state_sha256"]
            and audit["checkpoint"]["path"] == str(generation / "model.pt")
            and audit["checkpoint"]["sha256"] == receipt["files"]["model.pt"]["sha256"]
            and execution["actual_exit_code"] == audit_exec["actual_exit_code"] == monitor["child_exit_code"] == 0
            and not audit_exec["timed_out"] and execution["source_bindings_unchanged"]
            and audit_exec["source_bindings_unchanged"]
            and monitor["status"] == monitor["supervisor_health"] == "pass" and monitor["post_exit_quiet_completed"],
            "Require independently audited weights and healthy actual training exit")
    bindings = {**training["source_bindings"], str(args.training_plan.resolve()): args.training_plan_sha256}
    paths = [audit_path, audit_exec_path, stage / "execution.json", stage / "stage.json", monitor_path,
             generation / "receipt.json", generation / "model.pt", generation / "rng.pt", generation / "metrics.jsonl",
             ROOT / "research/direct/evaluate_latency58_sdr.py"]
    bindings.update({str(p): sha(p) for p in paths})
    return {"model_kind": "sdr_candidate", "step": receipt["step"], "generation": str(generation),
            "expected_model_state_sha256": receipt["model_state_sha256"], "checkpoint": audit["checkpoint"],
            "training_plan": {"path": str(args.training_plan.resolve()), "sha256": args.training_plan_sha256}}, bindings


def qualify(out):
    from research import evaluate as legacy
    reference_path = PHASE / "teacher-half250-full14-001/result.json"
    plan, report, execution = read(out / "plan.json"), read(out / "result.json"), read(out / "execution.json")
    reference = read(reference_path)
    indices = list(range(6))
    require(plan["track_indices"] == indices and plan["model_kind"] == "working_baseline"
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"] and report["inputs_unchanged"]
            and execution["plan_sha256"] == report["plan_sha256"] == sha(out / "plan.json"),
            "Require completed six-track qualification")
    candidate, baseline = report["results"][0], reference["results"][0]
    require(candidate["model"]["model_state_sha256"] == baseline["model"]["model_state_sha256"]
            == plan["expected_model_state_sha256"], "Qualification weights differ")
    for key in ("manifest_sha256", "config_sha256", "metric_source_sha256", "output_policy", "metrics",
                "precision", "torch_version", "unroll_hops", "io_block_hops", "excerpts"):
        require(report[key] == reference[key], "Qualification protocol differs: " + key)
    require(candidate["tracks"] == baseline["tracks"][:6]
            and candidate["stream_batches"] == baseline["stream_batches"][:6]
            and candidate["aggregate"] == legacy._aggregate_tracks(baseline["tracks"][:6])
            and len(report["parallel_track_execution"]["observed_worker_pids"]) == 2,
            "Two-process scoring does not exactly reproduce the stored baseline")
    paths = (reference_path, out / "plan.json", out / "result.json", out / "execution.json")
    bindings = {**plan["source_bindings"], **{str(p): sha(p) for p in paths}}
    require(all(sha(p) == s for p, s in bindings.items()), "Qualification input changed")
    write(out / "qualification.json", {
        "schema": "latency58-sdr-parallel-qualification-v1", "status": "pass", "source_bindings": bindings,
        "exact_track_reports": True, "exact_stream_metadata": True, "exact_original_aggregate": True,
        "track_indices": indices, "timing_qualification": False,
        "elapsed_seconds": execution["elapsed_seconds"],
    })
    print(json.dumps({"event": "parallel_qualification_pass", "elapsed_seconds": execution["elapsed_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualify", action="store_true")
    parser.add_argument("--qualification-directory", type=Path)
    parser.add_argument("--training-plan", type=Path)
    parser.add_argument("--training-plan-sha256")
    parser.add_argument("--stage-directory", type=Path)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and args.output_prefix
            and all(c.isalnum() or c in "-_" for c in args.output_prefix), "Invalid prefix or cwd")
    plan = protocol_template()
    out = PHASE / (args.output_prefix + "-full14-001")
    require(not out.exists(), "Preserve existing output")
    if args.qualify:
        from research.direct.latency58_sdr_teacher import STUDENT_STATE_SHA256
        training_path = PHASE / "sdr-teacher-prep-001/c91-plan.json"
        training = read(training_path)
        from research.direct.latency58_sdr_checkpoint import require_space
        require_space(training, 2_000_000)
        fields = {"model_kind": "working_baseline", "expected_model_state_sha256": STUDENT_STATE_SHA256}
        bindings = {**training["source_bindings"], str(training_path): sha(training_path)}
        reference = PHASE / "teacher-half250-full14-001/result.json"
        bindings[str(reference)] = sha(reference)
        indices = list(range(6))
    else:
        require(all((args.qualification_directory, args.training_plan, args.training_plan_sha256, args.stage_directory)),
                "Candidate scoring needs qualification and an audited generation")
        proof_path = args.qualification_directory / "qualification.json"
        proof = read(proof_path)
        require(proof["schema"] == "latency58-sdr-parallel-qualification-v1" and proof["status"] == "pass"
                and proof["exact_track_reports"] and proof["exact_stream_metadata"] and proof["exact_original_aggregate"]
                and proof["track_indices"] == list(range(6))
                and all(sha(p) == s for p, s in proof["source_bindings"].items()), "Parallel qualification changed")
        fields, bindings = candidate_bindings(args)
        bindings.update(proof["source_bindings"])
        bindings[str(proof_path.resolve())] = sha(proof_path)
        indices = list(range(14))
    plan.update(schema="latency58-sdr-parallel-music-plan-v1", workers=2, track_indices=indices,
                label=args.output_prefix, output_directory=str(out), **fields)
    plan["source_bindings"].update(bindings)
    require(all(sha(p) == s for p, s in plan["source_bindings"].items()), "Evaluation inputs changed")
    out.mkdir()
    path = out / "plan.json"
    write(path, plan)
    argv = [PYTHON, "-u", "-m", "research.direct.evaluate_latency58_sdr_parallel", "--plan", str(path),
            "--plan-sha256", sha(path)]
    print(json.dumps({"event": "parallel_evaluate", "indices": indices, "plan_sha256": sha(path)}), flush=True)
    execute(argv, out, "evaluation", plan["timeout_seconds"], plan["source_bindings"], {"plan_sha256": sha(path)})
    if args.qualify:
        qualify(out)
    else:
        from research.direct.compare_latency58_sdr import REFERENCE_PATHS
        paths = (ROOT / "research/direct/compare_latency58_sdr.py", ROOT / "research/direct/compare.py",
                 out / "result.json", out / "execution.json", path, *REFERENCE_PATHS.values())
        bindings = {str(p): sha(p) for p in paths}
        argv = [PYTHON, "-u", "-m", "research.direct.compare_latency58_sdr", "--candidate", str(out)]
        execute(argv, out, "comparison", 120, bindings, {"source_bindings": bindings})


if __name__ == "__main__":
    main()
