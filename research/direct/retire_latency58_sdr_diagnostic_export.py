"""Retire the unselected C91-student diagnostic graph; retain its recreation and failure evidence."""
from __future__ import annotations

import argparse
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_sdr_checkpoint import read_generation, require_space


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Retirement plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-sdr-diagnostic-export-retirement-plan-v1"
            and plan["reserve_after_bytes"] == 400_000_000, "Different retirement scope")
    verify_inputs(plan)
    folder = PHASE / "sdr-c91-1000-onnx-diagnostic-001"
    target = folder / "sdr-c91-1000-hop128.onnx"
    original_path = folder / "plan.json"
    original = read(original_path)
    verify_inputs(original)
    execution = read(folder / "export-execution.json")
    verified = read(folder / "sdr-c91-1000-hop128.verification.json")
    require(original["purpose"] == "development_parity" and original["model_kind"] == "sdr_candidate"
            and original["output"] == str(target) and original["step"] == 1000
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"]
            and execution["plan_sha256"] == verified["plan_sha256"] == sha(original_path)
            and verified["source_bindings_unchanged"] and verified["verification"]["passed"]
            and verified["export_purpose"] == "development_parity"
            and not verified["native_host_qualified"] and verified["quality_retention_decision"] is None,
            "Not the completed, unselected development export")
    conclusion = read(PHASE / "sdr-c91-1000-long-parity-diagnostic-001/conclusion.json")
    require(not conclusion["model_selected"] and not conclusion["native_qualified"]
            and not conclusion["literal_vs_ort_within_tolerance"], "Different parity conclusion")
    audit = read(original["audit"]["path"])
    audit_execution = read(original["audit_execution"]["path"])
    receipt = read_generation(original["generation"], expected_plan_sha=original["training_plan"]["sha256"],
                              require_optimizer=False)
    require(audit["status"] == "pass" and audit_execution["actual_exit_code"] == 0
            and audit["source_bindings_unchanged"] and audit_execution["source_bindings_unchanged"]
            and audit["checkpoint"] == original["checkpoint"] == verified["checkpoint"]
            and receipt["model_state_sha256"] == audit["model_state_sha256"] == original["model_state_sha256"]
            and receipt["files"]["model.pt"]["sha256"] == sha(original["checkpoint"]["path"]),
            "Retained checkpoint or audit differs")
    expected = {"path": str(target), "sha256": verified["onnx_sha256"], "bytes": verified["onnx_bytes"]}
    require(plan["target"] == expected and expected["bytes"] == 111342446
            and target.is_file() and not target.is_symlink() and target.stat().st_size == expected["bytes"]
            and sha(target) == expected["sha256"] and str(target) not in plan["source_bindings"],
            "Different disposable graph")
    for item in plan["retained_input_plans"]:
        document = read(item["path"])
        require(plan["source_bindings"].get(item["path"]) == item["sha256"] == sha(item["path"])
                and str(target) not in document["source_bindings"], "Graph remains a retained input")
        verify_inputs(document)
    protected = plan["protected_files"]
    required = {str(original_path), str(folder / "export-execution.json"),
                str(folder / "sdr-c91-1000-hop128.verification.json"),
                str(PHASE / "sdr-c91-1000-long-parity-diagnostic-001/conclusion.json"),
                str(PHASE / "sdr-c91-1000-native-diagnostic-prep-001/native-qualification-001.json"),
                str(PHASE / "teacher-half250-onnx-001/teacher-half250-hop128.onnx"),
                original["checkpoint"]["path"]}
    require(required.issubset(protected) and str(target) not in protected
            and all(plan["source_bindings"].get(p) == s == sha(p) for p, s in protected.items()),
            "Incomplete protected model and evidence inventory")
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "intent.json").exists(), "Preserve retirement")
    before = require_space(plan, 0)
    require(before - expected["bytes"] + 400_000_000 < plan["stop_counted_bytes"], "Insufficient forecast headroom")
    write(out / "intent.json", {"plan_sha256": args.plan_sha256, "target": expected,
                               "protected_files": protected, "counted_bytes_before": before})
    require(sha(target) == expected["sha256"], "Graph changed immediately before retirement")
    target.unlink()
    require(not target.exists(), "Retirement failed")
    verify_inputs(plan)
    verify_inputs(original)
    for item in plan["retained_input_plans"]:
        verify_inputs(read(item["path"]))
    after = require_space(plan, 400_000_000)
    write(out / "receipt.json", {"schema": "latency58-sdr-diagnostic-export-retirement-v1", "status": "complete",
          "plan_sha256": args.plan_sha256, "intent_sha256": sha(out / "intent.json"), "retired_export": expected,
          "freed_bytes": expected["bytes"], "protected_files": protected, "source_bindings_unchanged": True,
          "retained_model_inputs_unchanged": True, "counted_bytes_after": after,
          "headroom_before_stop_bytes": plan["stop_counted_bytes"] - after, "reserved_bytes": 400_000_000,
          "source_audio_deleted": False, "trained_checkpoint_deleted": False, "accepted_model_deleted": False,
          "training_updates_executed": 0, "native_failure_evidence_retained": True,
          "recreation": "Original exporter, checkpoint, export plan, verification and failed long/native parity are retained. Re-export requires a fresh output and a new reservation; this retirement does not revise the failed parity result."})
    print({"status": "complete", "freed_bytes": expected["bytes"], "counted_bytes_after": after}, flush=True)


if __name__ == "__main__":
    main()
