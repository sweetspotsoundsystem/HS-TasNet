"""Prepare and execute fixed vocal views for an audited, fully scored pilot."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, ROOT, PYTHON, read, require, sha, write, execute
from research.direct.report_latency58_sdr import load_completed
from research.direct.train_latency58 import verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--reservation", type=Path, required=True)
    parser.add_argument("--reservation-sha256", required=True)
    args = parser.parse_args()
    require(args.prefix and all(c.isalnum() or c in "-_" for c in args.prefix)
            and sha(args.reservation) == args.reservation_sha256, "Invalid prefix or changed allocation")
    reservation = read(args.reservation)
    verify_inputs(reservation)
    out = PHASE / (args.prefix + "-views-001")
    require(not out.exists() and str(out) in reservation["evaluation_directories"], "Unreserved or existing diagnostic")
    bindings = dict(reservation["source_bindings"])
    quality_dir = PHASE / (args.prefix + "-full14-001")
    quality_plan, report = load_completed(quality_dir, bindings)
    require(quality_plan["schema"] == "latency58-cleanup-successor-parallel-music-plan-v1"
            and quality_plan["step"] == 250 and report["inputs_unchanged"], "Incomplete full-mixture pilot score")
    training_binding = quality_plan["training_plan"]
    training = read(training_binding["path"])
    require(sha(training_binding["path"]) == training_binding["sha256"]
            and training["schema"] == "latency58-cleanup-successor-training-v1" and not training["resource_only"],
            "Different pilot training plan")
    verify_inputs(training)
    from research.direct.latency58_cleanup_successor_checkpoint_v2 import read_generation, require_space
    receipt = read_generation(quality_plan["generation"], expected_plan_sha=training_binding["sha256"],
                              require_optimizer=False)
    fingerprint = report["results"][0]["model"]["model_state_sha256"]
    require(receipt["step"] == 250 and receipt["model_state_sha256"] == fingerprint,
            "Full-mixture score and saved pilot differ")
    require_space(training, 5_000_000)
    template_path = PHASE / "vocal-views-working-001/plan.json"
    template = read(template_path)
    template_execution = read(template_path.parent / "execution.json")
    require(template_execution["actual_exit_code"] == 0 and not template_execution["timed_out"]
            and template_execution["source_bindings_unchanged"]
            and template_execution["plan_sha256"] == sha(template_path), "Vocal baseline protocol did not complete")
    verify_inputs(template)
    bindings.update(template["source_bindings"])
    bindings.update(quality_plan["source_bindings"])
    for path in (template_path, template_path.parent / "execution.json", args.reservation,
                 Path(__file__).resolve(), ROOT / "research/direct/evaluate_latency58_cleanup_successor_views.py",
                 ROOT / "research/direct/evaluate_latency58_cleanup_successor.py",
                 ROOT / "research/direct/latency58_vocal_views.py"):
        bindings[str(path)] = sha(path)
    plan = copy.deepcopy(template)
    plan.update(schema="latency58-cleanup-successor-views-evaluation-plan-v1", output_directory=str(out),
                model={"kind": "cleanup_successor", "label": args.prefix, "arm": training["arm"], "teacher_mode": training["teacher_mode"],
                       "model_state_sha256": fingerprint,
                       "quality_plan": {"path": str(quality_dir / "plan.json"), "sha256": sha(quality_dir / "plan.json")}},
                reservation={"path": str(args.reservation), "sha256": args.reservation_sha256},
                source_bindings=bindings, protocol_template={"path": str(template_path), "sha256": sha(template_path)})
    verify_inputs(plan)
    out.mkdir()
    write(out / "plan.json", plan)
    execute([PYTHON, "-u", "-m", "research.direct.evaluate_latency58_cleanup_successor_views",
             "--plan", str(out / "plan.json"), "--plan-sha256", sha(out / "plan.json")],
            out, "evaluation", 3000, bindings, {"plan_sha256": sha(out / "plan.json")})


if __name__ == "__main__":
    main()
