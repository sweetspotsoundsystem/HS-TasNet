"""Freeze, rehearse and run one loss-only control under the existing GPU supervisor."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import time

from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, execute, write, sha, read, require
from research.direct.train_latency58 import verify_inputs, load_source, state_sha256
from research.direct.hare_loss_ablation_checkpoint import validate_recipe, validate_journal, require_space, load_parent

OUT = PHASE / "hare-loss-ablation-001"


def binding(path):
    return {"path": str(path.resolve()), "sha256": sha(path)}


def launch(plan_path, previous_execution_path, name):
    plan = read(plan_path)
    verify_inputs(plan)
    validate_recipe(plan)
    require_space(plan, 400_000_000)
    previous_execution = read(previous_execution_path)
    previous_path = Path(previous_execution["monitor_result"])
    previous = read(previous_path)
    require(previous_execution["actual_exit_code"] == 0 and previous_execution["source_bindings_unchanged"]
            and previous["status"] == previous["supervisor_health"] == "pass"
            and previous["child_exit_code"] == 0 and previous["post_exit_quiet_completed"],
            "Previous monitored stage failed")
    out = OUT / name
    out.mkdir()
    stop = 2 if plan["resource_only"] else 250
    stage = {"schema": "hare-loss-ablation-stage-v1", "plan_sha256": sha(plan_path),
             "start_step": 0, "stop_step": stop, "output_directory": str(out),
             "previous_event_record_id": previous["last_event_record_id"],
             "previous_monitor": binding(previous_path),
             "source_bindings": {str(p.resolve()): sha(p) for p in (plan_path, previous_path, previous_execution_path)}}
    write(out / "stage.json", stage)
    spec = {"schema": "gpu-watchdog-launch-v1", "cwd": str(ROOT), "environment": plan["environment"],
            "progress_path": str(Path(plan["run_dir"]) / "metrics.jsonl"),
            "argv": [PYTHON, "-u", "-m", "research.direct.train_hare_loss_ablation",
                     "--plan", str(plan_path), "--plan-sha256", sha(plan_path),
                     "--stage", str(out / "stage.json"), "--stage-sha256", sha(out / "stage.json")]}
    write(out / "watchdog-spec.json", spec)
    monitor_out = Path(plan["watchdog_source"]).parent / ("hare-loss-ablation-001-" + name)
    require(not monitor_out.exists(), "Preserve existing monitor")
    argv = [PYTHON, plan["watchdog_source"], "--launch-spec", str(out / "watchdog-spec.json"),
            "--launch-spec-sha256", sha(out / "watchdog-spec.json"), "--output-dir", str(monitor_out),
            "--max-runtime-seconds", str(max(300, stop * 32 + 180)), "--poll-seconds", "2",
            "--query-timeout-seconds", "10", "--startup-grace-seconds", "120",
            "--progress-timeout-seconds", "60", "--stop-grace-seconds", "15",
            "--post-exit-quiet-seconds", "10", "--max-temperature-c", "80", "--memory-headroom-mib", "4096"]
    write(out / "command.json", {"argv": argv})
    began = time.monotonic()
    print(json.dumps({"event": "launch", "stage": name, "updates": stop}), flush=True)
    with (out / "console.log").open("x") as log:
        child = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    unchanged = all(sha(p) == s for p, s in {**plan["source_bindings"], **stage["source_bindings"]}.items())
    result = {"actual_exit_code": child.returncode, "elapsed_seconds": time.monotonic() - began,
              "source_bindings_unchanged": unchanged, "plan_sha256": sha(plan_path),
              "monitor_result": str(monitor_out / "result.json")}
    write(out / "execution.json", result)
    require(child.returncode == 0 and unchanged, "Ablation monitored stage failed")
    terminal = read(monitor_out / "result.json")
    require(terminal["status"] == terminal["supervisor_health"] == "pass"
            and terminal["child_exit_code"] == 0 and terminal["post_exit_quiet_completed"],
            "Ablation supervisor failed")
    print(json.dumps({"event": "stage_pass", "stage": name, **result}), flush=True)
    return out / "execution.json"


def main():
    require(Path.cwd() == ROOT and not OUT.exists(), "Use a new frozen experiment directory")
    reference_path = PHASE / "leader-cleanup-production-prep-001/training-plan.json"
    reference = read(reference_path)
    generation = Path(reference["run_dir"]) / "checkpoints/step-000250"
    original_sources = reference["source_bindings"]
    verify_inputs(reference)
    # Every dependency of the positive arm remains bound; new sources are additional.
    sources = [ROOT / "research/HARE_PAPER_EXPERIMENTS.md", Path(__file__).resolve(), reference_path,
               generation / "metrics.jsonl", generation / "receipt.json",
               ROOT / "research/direct/train_hare_loss_ablation.py",
               ROOT / "research/direct/hare_loss_ablation_checkpoint.py",
               ROOT / "research/direct/audit_hare_loss_ablation.py"]
    bindings = {**original_sources, **{str(p): sha(p) for p in sources}}
    OUT.mkdir()
    decision = {"schema": "hare-loss-ablation-decision-v1", "comparison_variable": "additional_loss_weight",
                "control_weight": 0, "positive_weight": .5,
                "matched_positive_training_plan": binding(reference_path),
                "maximum_production_updates": 250, "quality_endpoints": [250],
                "automatic_continuation": False, "automatic_model_replacement": False,
                "selection_scope": "Retrospective single-seed component ablation; positive arm already selected.",
                "evaluation": "Primary full14 and controlled views only; report all per-stem and local costs.",
                "confirmation_policy": "Previously consumed confirmation intervals cannot become new test material.",
                "outside_roots_reservation_bytes": 800_000_000,
                "checkpoint_and_quality_reserve_bytes": 400_000_000,
                "source_bindings": bindings}
    write(OUT / "decision.json", decision)
    plan = copy.deepcopy(reference)
    for key in ("resource_plan", "full_resource", "full_resource_execution", "resource_comparison",
                "resource_comparison_execution"):
        plan.pop(key, None)
    plan.update(schema="hare-loss-ablation-training-v1", resource_only=True,
                comparison_variable="additional_loss_weight", additional_loss_weight=0,
                matched_positive_training_plan=binding(reference_path),
                matched_positive_journal=binding(generation / "metrics.jsonl"),
                matched_positive_receipt=binding(generation / "receipt.json"),
                preparation_decision=binding(OUT / "decision.json"),
                run_dir=str(PHASE / "hare-loss-ablation-resource-run-001"),
                stop_counted_bytes=79_200_000_000, outside_roots_reservation_bytes=800_000_000,
                source_bindings={**bindings, str(OUT / "decision.json"): sha(OUT / "decision.json")})
    validate_recipe(plan)
    counted = require_space(plan, 400_000_000)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = load_parent(plan)
    require(state_sha256(model.state_dict()) == plan["initialized_model_state_sha256"]
            and not torch.cuda.is_initialized(), "CPU parent reconstruction failed")
    del model
    helpers = load_source("hare_ablation_preflight_helpers", plan["helper_source"])
    fixture = [json.loads(line) for line in (generation / "metrics.jsonl").read_text().splitlines()[:2]]
    for row in fixture:
        row.update(additional_loss_weight=0, controlled_deployed_contribution=0, loss=row["base_loss"])
        for micro in row["microbatches"]:
            micro.update(additional_loss_weight=0, controlled_deployed_contribution=0, loss=micro["base_loss"])
    def validate(rows):
        return validate_journal(("\n".join(json.dumps(row) for row in rows) + "\n").encode(), 2, plan, helpers)
    validate(fixture)
    rejected = 0
    for field, value in (("augmented_batch_sha256", "0" * 64), ("additional_loss_weight", .5),
                         ("view_codes", [2, 2, 2, 2]), ("teacher_targets_sha256", "0" * 64)):
        malformed = copy.deepcopy(fixture)
        malformed[0]["microbatches"][0][field] = value
        try:
            validate(malformed)
        except RuntimeError:
            rejected += 1
    require(rejected == 4, "Malformed ablation fixtures were accepted")
    write(OUT / "preflight.json", {"status": "pass", "cpu_parent_state_exact": True,
          "zero_weight_arithmetic_fixture_pass": True, "negative_journal_fixtures_rejected": rejected,
          "fixture_is_training_replay": False, "counted_bytes": counted,
          "combined_forecast_bytes": counted + 800_000_000 + 400_000_000,
          "cuda_initialized": False, "source_bindings": bindings})
    plan["source_bindings"][str(OUT / "preflight.json")] = sha(OUT / "preflight.json")
    write(OUT / "resource-plan.json", plan)
    previous = PHASE / "public-cuda-qualification-001/execution.json"
    resource_execution = launch(OUT / "resource-plan.json", previous, "resource")
    resource_path = Path(plan["run_dir"]) / "resource.json"
    resource = read(resource_path)
    require(resource["status"] == "pass" and resource["training_updates_executed"] == 2
            and not resource["checkpoint_written"] and resource["source_bindings_unchanged"],
            "Resource rehearsal incomplete")
    production = copy.deepcopy(plan)
    production.update(resource_only=False, run_dir=str(PHASE / "hare-loss-ablation-zero-b16-250-001"),
                      full_resource=binding(resource_path), full_resource_execution=binding(resource_execution),
                      resource_plan=binding(OUT / "resource-plan.json"))
    production["source_bindings"].update({str(p): sha(p) for p in (
        resource_path, resource_execution, OUT / "resource-plan.json")})
    write(OUT / "training-plan.json", production)
    launch(OUT / "training-plan.json", resource_execution, "production")
    generation = Path(production["run_dir"]) / "checkpoints/step-000250"
    audit_bindings = {**production["source_bindings"], str(OUT / "training-plan.json"): sha(OUT / "training-plan.json"),
                      **{str(p): sha(p) for p in generation.iterdir() if p.is_file()}}
    execute([PYTHON, "-u", "-m", "research.direct.audit_hare_loss_ablation", "--plan", str(OUT / "training-plan.json"),
             "--plan-sha256", sha(OUT / "training-plan.json"), "--generation", str(generation),
             "--output", str(OUT / "audit.json")], OUT, "audit", 180, audit_bindings, {})
    require(read(OUT / "audit.json")["status"] == "pass", "Saved endpoint audit failed")
    write(OUT / "result.json", {"status": "training_and_audit_pass", "updates": 250,
          "audit": binding(OUT / "audit.json"), "training_plan": binding(OUT / "training-plan.json"),
          "quality_evaluated": False, "automatic_model_replacement": False})
    print(json.dumps(read(OUT / "result.json")), flush=True)


if __name__ == "__main__":
    main()
