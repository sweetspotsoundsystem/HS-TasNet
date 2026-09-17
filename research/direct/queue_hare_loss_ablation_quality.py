"""After a successful audited control, score the frozen primary and source-view protocols."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, read, write, sha, require, execute
from research.direct.train_latency58 import verify_inputs
from research.direct.hare_loss_ablation_checkpoint import read_generation, require_space
from research.direct.report_latency58_sdr import load_completed
from research.direct.report_latency58_vocal_focus import music_cells, load_views

OUT = PHASE / "hare-loss-ablation-001"


def binding(path):
    return {"path": str(path.resolve()), "sha256": sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-pid", type=int, required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and OUT.is_dir(), "Missing prepared ablation")
    decision_sha = sha(OUT / "decision.json")
    source = Path(__file__).resolve()
    source_sha = sha(source)
    began = time.monotonic()
    while not (OUT / "result.json").exists():
        require(time.monotonic() - began < 9500, "Training wait exceeded its fixed deadline")
        require(sha(OUT / "decision.json") == decision_sha and sha(source) == source_sha,
                "Decision or quality launcher changed while waiting")
        proc = Path(f"/proc/{args.training_pid}/cmdline")
        require(proc.exists() and b"research.direct.run_hare_loss_ablation" in proc.read_bytes(),
                "Training root exited without an audited endpoint")
        for name in ("resource", "production"):
            execution_path = OUT / name / "execution.json"
            if execution_path.exists():
                require(read(execution_path)["actual_exit_code"] == 0, "Monitored training failed")
        time.sleep(5)
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
            and audit["additional_loss_weight"] == 0 and audit["matched_input_journal_verified"],
            "Scored control differs from the audited matched ablation")
    verify_inputs(training)
    require_space(training, 10_000_000)
    paths = [source, training_path, OUT / "result.json", OUT / "audit.json", OUT / "audit-execution.json",
             OUT / "production/execution.json", monitor_path,
             *[generation / name for name in ("model.pt", "receipt.json", "rng.pt", "metrics.jsonl")],
             *[ROOT / "research/direct" / name for name in (
                 "evaluate_hare_loss_ablation.py", "evaluate_hare_loss_ablation_parallel.py",
                 "evaluate_hare_loss_ablation_views.py", "report_latency58_sdr.py",
                 "report_latency58_vocal_focus.py", "compare.py", "compare_latency58_vocal_views.py")]]
    bindings = {**training["source_bindings"], **{str(p): sha(p) for p in paths}}
    quality_dir, views_dir = OUT / "full14", OUT / "views"
    positive_plan_path = PHASE / "leader-cleanup-250-full14-001/plan.json"
    positive_plan, positive_report = load_completed(positive_plan_path.parent, bindings)
    require(positive_plan["training_plan"] == training["matched_positive_training_plan"],
            "Positive score does not belong to the matched positive training arm")
    bindings.update(positive_plan["source_bindings"])
    proof_path = PHASE / "sdr-parallel-baseline-qualification-full14-001/qualification.json"
    proof = read(proof_path)
    require(proof["status"] == "pass" and proof["exact_track_reports"] and proof["exact_stream_metadata"]
            and proof["exact_original_aggregate"], "Parallel scoring is not qualified")
    verify_inputs(proof)
    bindings.update(proof["source_bindings"])
    bindings[str(proof_path)] = sha(proof_path)
    quality = {"schema": "hare-loss-ablation-parallel-music-plan-v1", "workers": 2,
               "model_kind": "hare_loss_ablation_candidate", "track_indices": list(range(14)),
               "label": "hare_loss_ablation_zero", "mode": "full14", "step": 250,
               "training_plan": binding(training_path), "generation": str(generation),
               "checkpoint": audit["checkpoint"], "expected_model_state_sha256": receipt["model_state_sha256"],
               "output_directory": str(quality_dir), "protocol_template": binding(positive_plan_path),
               "parallel_qualification": binding(proof_path), "source_bindings": bindings}
    verify_inputs(quality)
    quality_dir.mkdir()
    write(quality_dir / "plan.json", quality)
    execute([PYTHON, "-u", "-m", "research.direct.evaluate_hare_loss_ablation_parallel",
             "--plan", str(quality_dir / "plan.json"), "--plan-sha256", sha(quality_dir / "plan.json")],
            quality_dir, "evaluation", 2400, bindings, {"plan_sha256": sha(quality_dir / "plan.json")})

    template_path = PHASE / "vocal-views-working-001/plan.json"
    template = read(template_path)
    verify_inputs(template)
    template_execution = read(template_path.parent / "execution.json")
    require(template_execution["actual_exit_code"] == 0 and not template_execution["timed_out"]
            and template_execution["plan_sha256"] == sha(template_path)
            and template_execution["source_bindings_unchanged"], "Source-view template failed")
    allocation = {"new_artifact_allowance_bytes": 10_000_000, "training_reserve_bytes": 350_000_000,
                  "counted_bytes_at_preparation": read(OUT / "preflight.json")["counted_bytes"],
                  "evaluation_directories": [str(quality_dir), str(views_dir)],
                  "source_bindings": {str(OUT / "decision.json"): decision_sha}}
    write(OUT / "evaluation-reservation.json", allocation)
    view_bindings = {**bindings, **template["source_bindings"], **{str(p): sha(p) for p in (
        template_path, template_path.parent / "execution.json", quality_dir / "plan.json",
        quality_dir / "result.json", quality_dir / "execution.json", OUT / "evaluation-reservation.json")}}
    views = copy.deepcopy(template)
    views.update(schema="hare-loss-ablation-views-evaluation-plan-v1", output_directory=str(views_dir),
                 model={"kind": "hare_loss_ablation", "label": "hare_loss_ablation_zero",
                        "model_state_sha256": receipt["model_state_sha256"],
                        "quality_plan": binding(quality_dir / "plan.json")},
                 reservation=binding(OUT / "evaluation-reservation.json"), source_bindings=view_bindings,
                 protocol_template=binding(template_path), counted_roots=training["counted_roots"],
                 stop_counted_bytes=training["stop_counted_bytes"])
    views_dir.mkdir()
    write(views_dir / "plan.json", views)
    execute([PYTHON, "-u", "-m", "research.direct.evaluate_hare_loss_ablation_views",
             "--plan", str(views_dir / "plan.json"), "--plan-sha256", sha(views_dir / "plan.json")],
            views_dir, "evaluation", 3000, view_bindings, {"plan_sha256": sha(views_dir / "plan.json")})

    from research.direct.compare import compare
    from research.direct.compare_latency58_vocal_views import compare_reports
    evidence = dict(view_bindings)
    _, control = load_completed(quality_dir, evidence)
    control_views = load_views(views_dir, evidence)
    comparisons = {}
    for label, music_path, views_path in (
        ("positive_weight_half", positive_plan_path.parent, PHASE / "leader-cleanup-250-views-001"),
        ("shared_parent", PHASE / "sdr-drum-accum-500-full14-001", PHASE / "vocal-views-drum500-001")):
        _, reference = load_completed(music_path, evidence)
        reference_views = load_views(views_path, evidence)
        require(reference["inputs_unchanged"] and control["inputs_unchanged"]
                and all(reference[k] == control[k] for k in (
                    "manifest_sha256", "output_policy", "precision", "metrics", "metric_source_sha256")),
                "Compared primary music protocols differ")
        require(reference_views["model"]["model_state_sha256"] == reference["results"][0]["model"]["model_state_sha256"],
                "Reference view and music model identities differ")
        comparisons[label] = {"control_minus_reference": compare(reference["results"][0], control["results"][0]),
                              "all_track_stem_cells": music_cells(reference["results"][0], control["results"][0]),
                              "views_control_minus_reference": compare_reports(reference_views, control_views)}
    require(control_views["model"]["model_state_sha256"] == control["results"][0]["model"]["model_state_sha256"]
            == receipt["model_state_sha256"], "Control view and primary music identities differ")
    verify_inputs({"source_bindings": evidence})
    summary = {"schema": "hare-loss-ablation-quality-summary-v1", "status": "pass",
               "source_bindings": evidence, "source_bindings_unchanged": True,
               "training_plan": binding(training_path), "model_state_sha256": receipt["model_state_sha256"],
               "comparison_variable": "additional_loss_weight", "control_weight": 0, "positive_weight": .5,
               "full_mixture_aggregate": control["results"][0]["aggregate"],
               "source_views_aggregate": control_views["aggregate"], "comparisons": comparisons,
               "automatic_model_replacement": False, "human_listening_completed": False,
               "limitations": ["Single-seed retrospective ablation; positive arm was selected during development.",
                   "Track bootstrap intervals omit training-seed uncertainty and model-selection correction.",
                   "These are familiar development songs; no untouched test-song result is claimed.",
                   "Controlled views retain wanted fidelity and gain alongside spill; no audibility claim follows."]}
    write(OUT / "quality-summary.json", summary)
    print(json.dumps({"event": "quality_complete", "summary": str(OUT / "quality-summary.json"),
                      "aggregate": summary["full_mixture_aggregate"]}), flush=True)


if __name__ == "__main__":
    main()
