"""Score reduced ordinary-mixture distillation against C204 and the matched half-teacher trial."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, read, write, sha, require, execute
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_reduced_teacher_checkpoint import read_generation, require_space
from research.direct.report_latency58_sdr import load_completed
from research.direct.report_latency58_vocal_focus import music_cells

OUT = PHASE / "latency58-reduced-teacher-001"


def binding(path):
    return {"path": str(path.resolve()), "sha256": sha(path)}


def main():
    require(Path.cwd() == ROOT and OUT.is_dir(), "Missing reduced-teacher trial")
    source = Path(__file__).resolve()
    completion = read(OUT / "result.json")
    training_path = OUT / "training-plan.json"
    training = read(training_path)
    audit, audit_execution = read(OUT / "audit.json"), read(OUT / "audit-execution.json")
    execution = read(OUT / "production/execution.json")
    monitor_path = Path(execution["monitor_result"])
    monitor = read(monitor_path)
    require(completion["status"] == "training_and_audit_pass" and completion["updates"] == 250
            and completion["audit"] == binding(OUT / "audit.json")
            and completion["training_plan"] == binding(training_path)
            and audit["status"] == "pass" and audit["plan_sha256"] == sha(training_path)
            and audit_execution["actual_exit_code"] == 0 and not audit_execution["timed_out"]
            and audit_execution["source_bindings_unchanged"] and execution["actual_exit_code"] == 0
            and execution["source_bindings_unchanged"] and execution["plan_sha256"] == sha(training_path)
            and monitor["status"] == monitor["supervisor_health"] == "pass"
            and monitor["child_exit_code"] == 0 and monitor["post_exit_quiet_completed"],
            "Control endpoint lacks successful audit or monitoring")
    generation = Path(audit["generation"])
    receipt = read_generation(generation, expected_plan_sha=sha(training_path), require_optimizer=False)
    require(receipt["model_state_sha256"] == audit["model_state_sha256"]
            and audit["generation_receipt_sha256"] == sha(generation / "receipt.json")
            and audit["additional_loss_weight"] == .5 and audit["teacher_weight"] == .25
            and audit["matched_input_journal_verified"],
            "Scored control differs from the audited matched ablation")
    verify_inputs(training)
    require_space(training, 10_000_000)
    paths = [source, training_path, OUT / "result.json", OUT / "audit.json", OUT / "audit-execution.json",
             OUT / "production/execution.json", monitor_path,
             *[generation / name for name in ("model.pt", "receipt.json", "rng.pt", "metrics.jsonl")],
             *[ROOT / "research/direct" / name for name in (
                 "evaluate_latency58_reduced_teacher.py", "evaluate_latency58_reduced_teacher_parallel.py",
                 "report_latency58_sdr.py",
                 "report_latency58_vocal_focus.py", "compare.py")]]
    bindings = {**training["source_bindings"], **{str(p): sha(p) for p in paths}}
    quality_dir = OUT / "full14"
    positive_plan_path = PHASE / "leader-cleanup-250-full14-001/plan.json"
    positive_plan, positive_report = load_completed(positive_plan_path.parent, bindings)
    require(positive_plan["checkpoint"] == training["parent"]["checkpoint"],
            "C204 score does not belong to the training parent")
    bindings.update(positive_plan["source_bindings"])
    proof_path = PHASE / "sdr-parallel-baseline-qualification-full14-001/qualification.json"
    proof = read(proof_path)
    require(proof["status"] == "pass" and proof["exact_track_reports"] and proof["exact_stream_metadata"]
            and proof["exact_original_aggregate"], "Parallel scoring is not qualified")
    verify_inputs(proof)
    bindings.update(proof["source_bindings"])
    bindings[str(proof_path)] = sha(proof_path)
    quality = {"schema": "latency58-reduced-teacher-parallel-music-plan-v1", "workers": 2,
               "model_kind": "latency58_reduced_teacher_candidate", "track_indices": list(range(14)),
               "label": "latency58_reduced_teacher_quarter", "mode": "full14", "step": 250,
               "training_plan": binding(training_path), "generation": str(generation),
               "checkpoint": audit["checkpoint"], "expected_model_state_sha256": receipt["model_state_sha256"],
               "output_directory": str(quality_dir), "protocol_template": binding(positive_plan_path),
               "parallel_qualification": binding(proof_path), "source_bindings": bindings}
    verify_inputs(quality)
    quality_dir.mkdir()
    write(quality_dir / "plan.json", quality)
    execute([PYTHON, "-u", "-m", "research.direct.evaluate_latency58_reduced_teacher_parallel",
             "--plan", str(quality_dir / "plan.json"), "--plan-sha256", sha(quality_dir / "plan.json")],
            quality_dir, "evaluation", 2400, bindings, {"plan_sha256": sha(quality_dir / "plan.json")})

    from research.direct.compare import compare
    evidence = dict(bindings)
    _, candidate = load_completed(quality_dir, evidence)
    comparisons = {}
    reference_states = {}
    for label, directory in (
        ("current_best_c204", PHASE / "leader-cleanup-250-full14-001"),
        ("matched_half_teacher", PHASE / "cleanup-rebound-250-full14-001")):
        _, reference = load_completed(directory, evidence)
        require(reference["inputs_unchanged"] and candidate["inputs_unchanged"]
                and all(reference[k] == candidate[k] for k in (
                    "manifest_sha256", "output_policy", "precision", "metrics", "metric_source_sha256")),
                "Compared primary music protocols differ")
        reference_states[label] = reference["results"][0]["model"]["model_state_sha256"]
        comparisons[label] = {"candidate_minus_reference": compare(reference["results"][0], candidate["results"][0]),
                              "all_track_stem_cells": music_cells(reference["results"][0], candidate["results"][0])}
    require(reference_states["current_best_c204"] == training["parent"]["model_state_sha256"]
            and reference_states["matched_half_teacher"] == read(training["matched_positive_receipt"]["path"])["model_state_sha256"]
            and candidate["results"][0]["model"]["model_state_sha256"] == receipt["model_state_sha256"],
            "Candidate or reference identities differ")
    verify_inputs({"source_bindings": evidence})
    summary = {"schema": "latency58-sdr-improvement-comparison-v1", "status": "pass",
               "source_bindings": evidence, "source_bindings_unchanged": True,
               "training_plan": binding(training_path), "model_state_sha256": receipt["model_state_sha256"],
               "reference_model_states": reference_states,
               "full_mixture_aggregate": candidate["results"][0]["aggregate"], "comparisons": comparisons,
               "automatic_model_replacement": False,
               "limitations": ["Existing development protocol; no unseen-song claim.",
                   "Track intervals omit training-seed variation and repeated model-selection effects."]}
    write(OUT / "sdr-comparison.json", summary)
    print(json.dumps({"event": "sdr_scoring_complete", "summary": str(OUT / "sdr-comparison.json"),
                      "aggregate": summary["full_mixture_aggregate"]}), flush=True)


if __name__ == "__main__":
    main()
