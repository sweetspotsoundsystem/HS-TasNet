"""Score one audited matched teacher/control endpoint using the same fixed protocol.

Load only the model file whose complete split endpoint passed the separate
CPU audit. Optimizer tensors are not needed for inference.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from research.direct.run_latency58_quality import (
    PHASE, PYTHON, ROOT, checkpoint_bindings, execute, read, require, sha, write,
)
from research.direct.train_latency58 import disk_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-plan", required=True, type=Path)
    parser.add_argument("--stage-directory", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--modes", nargs="+", choices=("actions60", "full14", "probes"),
                        default=["actions60", "full14", "probes"])
    args = parser.parse_args()
    require(len(set(args.modes)) == len(args.modes), "Duplicate evaluation modes")
    require(Path.cwd() == ROOT and args.output_prefix and all(
        c.isalnum() or c in "-_" for c in args.output_prefix), "Invalid directory prefix or cwd")
    training = read(args.training_plan)
    require(training["schema"] == "latency58-teacher-training-plan-v1", "Use a matched-loss follow-up plan")
    stage = args.stage_directory.resolve(strict=True)
    run = Path(training["run_dir"])
    require(stage.is_relative_to(PHASE) and run.parent == PHASE, "Endpoint must belong to this phase")
    audit_path, audit_execution_path = stage / "checkpoint-audit.json", stage / "audit-execution.json"
    audit, audit_execution = read(audit_path), read(audit_execution_path)
    execution, command = read(stage / "root-execution.json"), read(stage / "root-command.json")
    argv = command["argv"]
    monitor = read(Path(argv[argv.index("--output-dir") + 1]) / "result.json")
    latest = read(run / "latest.json")
    receipt_path = run / "endpoint/receipt.json"
    receipt = read(receipt_path)
    checkpoint = audit["checkpoint"]
    require(audit["step"] == latest["step"] == receipt["step"] == 250 and audit["status"] == "pass"
            and audit["plan_sha256"] == latest["plan_sha256"] == sha(args.training_plan)
            and audit["objective"] == training["objective"] and audit["teacher_weight"] == training["teacher_weight"]
            and audit["model_state_sha256"] == receipt["model_state_sha256"]
            and audit["endpoint_receipt_sha256"] == latest["receipt_sha256"] == sha(receipt_path)
            and checkpoint["path"] == str(run / "endpoint/model.pt")
            and checkpoint["sha256"] == receipt["files"]["model.pt"]["sha256"] == sha(checkpoint["path"])
            and execution["actual_exit_code"] == audit_execution["actual_exit_code"] == monitor["child_exit_code"] == 0
            and not audit_execution["timed_out"] and execution["source_bindings_unchanged"]
            and audit_execution["source_bindings_unchanged"]
            and monitor["status"] == monitor["supervisor_health"] == "pass" and monitor["post_exit_quiet_completed"],
            "Require the completed arm and separately audited exact model")
    bindings = {checkpoint["path"]: checkpoint["sha256"]}
    for path in (receipt_path, audit_path, audit_execution_path, stage / "root-execution.json", stage / "root-command.json"):
        bindings[str(path)] = sha(path)
    require(disk_bytes(PHASE) + 25_000_000 < training["artifact_allowance_bytes"],
            "Insufficient room for the four Actions WAVs and reports")
    bindings.update({str(args.training_plan.resolve()): sha(args.training_plan),
                     str(Path(__file__).resolve()): sha(__file__)})
    for name in ("latency58_asymmetric.py", "latency58_encoder_window.py", "latency58_gpu.py",
                 "latency58_asymmetric_checkpoint.py", "latency58_teacher_checkpoint.py",
                 "evaluate_latency58_teacher.py", "evaluate_latency58_teacher_probes.py"):
        path = ROOT / "research/direct" / name
        bindings[str(path)] = sha(path)
    bindings[training["hann_parent_checkpoint"]["path"]] = training["hann_parent_checkpoint"]["sha256"]
    prepared = []
    for mode in args.modes:
        template_dir = PHASE / ("pilot2000-" + mode + "-001")
        template_path = template_dir / "plan.json"
        template_execution = read(template_dir / "execution.json")
        require(template_execution["actual_exit_code"] == 0 and not template_execution["timed_out"]
                and template_execution["source_bindings_unchanged"]
                and template_execution["plan_sha256"] == sha(template_path), "Template lacks successful actual execution")
        template = read(template_path)
        old = checkpoint_bindings(template["checkpoint"])
        require(all(template["source_bindings"].get(path) == digest for path, digest in old.items()),
                "Template checkpoint binding inventory differs")
        unchanged = {path: digest for path, digest in template["source_bindings"].items() if path not in old}
        require(all(sha(path) == digest for path, digest in unchanged.items()), "Protocol source or input changed")
        out = PHASE / (args.output_prefix + "-" + mode + "-001")
        require(not out.exists(), "Preserve existing evaluation: " + str(out))
        plan = copy.deepcopy(template)
        plan.update(schema=plan["schema"].replace("latency58-", "latency58-teacher-"),
                    parent_checkpoint=training["hann_parent_checkpoint"],
                    checkpoint=checkpoint, output_directory=str(out), source_bindings={**unchanged, **bindings},
                    protocol_template={"path": str(template_path), "sha256": sha(template_path)})
        if mode != "probes":
            plan["label"] = args.output_prefix
        prepared.append((mode, out, plan))
    for mode, out, plan in prepared:
        out.mkdir()
        plan_path = out / "plan.json"
        write(plan_path, plan)
        module = "research.direct.evaluate_latency58_teacher_probes" if mode == "probes" else "research.direct.evaluate_latency58_teacher"
        argv = [PYTHON, "-u", "-m", module, "--plan", str(plan_path), "--plan-sha256", sha(plan_path)]
        print(json.dumps({"event": "evaluate", "mode": mode, "step": audit["step"],
                          "objective": training["objective"], "plan_sha256": sha(plan_path)}), flush=True)
        execute(argv, out, "evaluation", plan["timeout_seconds"], plan["source_bindings"],
                {"plan_sha256": sha(plan_path)})
        if mode != "probes":
            comparator = ROOT / "research/direct/compare_latency58.py"
            comparison_bindings = {str(comparator): sha(comparator), str(out / "result.json"): sha(out / "result.json")}
            argv = [PYTHON, "-u", "-m", "research.direct.compare_latency58", "--candidate", str(out), "--mode", mode]
            execute(argv, out, "comparison", 120, comparison_bindings, {"source_bindings": comparison_bindings})
    print(json.dumps({"event": "quality_bundle_completed", "step": audit["step"],
                      "objective": training["objective"], "quality_selected": False}), flush=True)


if __name__ == "__main__":
    main()
