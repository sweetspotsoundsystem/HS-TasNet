"""Release one closed, lower-SDR Adam state if the first direct-SDR pilot needs a successor."""
from __future__ import annotations

import json
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import disk_bytes, verify_inputs


def main():
    out = PHASE / "direct-sdr-storage-002"
    require(Path.cwd() == ROOT and not out.exists(), "Preserve retirement receipts")
    pilot = PHASE / "direct-sdr-001"
    completed = read(pilot / "result.json")
    quality_execution = read(pilot / "full14/execution.json")
    require(completed["status"] == "training_audit_and_full14_complete" and not completed["target_reached"]
            and quality_execution["actual_exit_code"] == 0 and not quality_execution["timed_out"]
            and quality_execution["source_bindings_unchanged"], "Wait for the actual first-pilot result")
    protected = {str(Path(__file__).resolve()): sha(__file__)}
    for path in (pilot / "plan.json", pilot / "full14/result.json", pilot / "result.json", pilot / "checkpoint-audit.json",
                 PHASE / "c204-residual-model-001/qualification.json", PHASE / "c204-residual-model-full14-001/result.json",
                 PHASE / "c204-residual-model-onnx-001/verification.json", PHASE / "c204-residual-model-native-001/review.json"):
        report = read(path)
        protected.update(report.get("source_bindings", {}))
        protected[str(path)] = sha(path)
    baselines = read(PHASE / "c204-residual-model-001/final-review.json")
    for key in ("checkpoint", "onnx", "c204_checkpoint_preserved", "working_plugin_graph_preserved"):
        binding = baselines[key]
        require(sha(binding["path"]) == binding["sha256"], "A rollback baseline changed")
        protected[binding["path"]] = binding["sha256"]
    for path in (pilot / "production-run/checkpoint").iterdir():
        protected[str(path)] = sha(path)
    run = PHASE / "cleanup-successor-b16-micro4-lr1e5-250"
    generation = run / "checkpoints/step-000250"
    latest = read(run / "audit-latest.json")
    for key in ("audit", "execution"):
        binding = latest[key]
        require(sha(binding["path"]) == binding["sha256"], "Closed optimizer audit changed")
        protected[binding["path"]] = binding["sha256"]
    audit, execution = read(latest["audit"]["path"]), read(latest["execution"]["path"])
    status, receipt = read(run / "status.json"), read(generation / "receipt.json")
    score = read(PHASE / "cleanup-successor-250-full14-001/result.json")["results"][0]
    require(status["status"] == "complete" and status["step"] == receipt["step"] == audit["step"] == 250
            and audit["status"] == "pass" and audit["source_bindings_unchanged"]
            and execution["actual_exit_code"] == 0 and not execution["timed_out"] and execution["source_bindings_unchanged"]
            and score["model"]["model_state_sha256"] == receipt["model_state_sha256"] == audit["model_state_sha256"]
            and score["aggregate"]["full_sdr_db"] < 4.069078803302578, "Optimizer is not from the closed lower-SDR endpoint")
    for name, binding in receipt["files"].items():
        path = generation / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == binding["bytes"]
                and sha(path) == binding["sha256"], "A closed-generation artifact changed")
        if name != "optimizer.pt":
            protected[str(path)] = binding["sha256"]
    target = generation / "optimizer.pt"
    require(str(target) not in protected, "Optimizer is required by preserved evidence")
    for proc in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            arguments = proc.read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        require(not any(value == str(run).encode() or value.startswith(str(generation).encode()) for value in arguments),
                "The retired generation is still used by a live command")
    protected[str(generation / "receipt.json")] = sha(generation / "receipt.json")
    verify_inputs({"source_bindings": protected})
    budget = read(pilot / "plan.json")
    before = sum(disk_bytes(Path(p)) for p in budget["counted_roots"])
    retired = {"path": str(target), **receipt["files"]["optimizer.pt"]}
    require(before - retired["bytes"] + 800_000_000 + 400_000_000 < 80_000_000_000,
            "Retirement does not fund the successor")
    out.mkdir()
    write(out / "intent.json", {"retired": retired, "protected_bindings": protected,
          "counted_bytes_before": before, "reason": "Closed lower-SDR Adam state; retain every model, RNG, metric and receipt."})
    require(sha(target) == retired["sha256"], "Optimizer changed before removal")
    target.unlink()
    verify_inputs({"source_bindings": protected})
    after = sum(disk_bytes(Path(p)) for p in budget["counted_roots"])
    write(out / "receipt.json", {"status": "complete", "intent_sha256": sha(out / "intent.json"), "retired": retired,
          "freed_bytes": retired["bytes"], "protected_bindings_unchanged": True, "counted_bytes_after": after,
          "forecast_including_outside_and_successor": after + 800_000_000 + 400_000_000})
    print(json.dumps(read(out / "receipt.json")), flush=True)


if __name__ == "__main__":
    main()
