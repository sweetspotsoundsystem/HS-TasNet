"""Release obsolete optimizer state for the new SDR training goal."""
from __future__ import annotations

import json
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import disk_bytes, verify_inputs


def main():
    out = PHASE / "direct-sdr-storage-001"
    require(Path.cwd() == ROOT and not out.exists(), "Preserve existing retirement receipts")
    protected = {str(Path(__file__).resolve()): sha(__file__)}
    for relative in ("leader-cleanup-250-full14-001/plan.json", "c204-residual-model-001/qualification.json",
                     "c204-residual-model-full14-001/result.json", "c204-residual-model-onnx-001/verification.json",
                     "c204-residual-model-native-001/review.json"):
        path = PHASE / relative
        report = read(path)
        protected.update(report["source_bindings"])
        protected[str(path)] = sha(path)
    baselines = read(PHASE / "c204-residual-model-001/final-review.json")
    for key in ("checkpoint", "onnx", "c204_checkpoint_preserved", "working_plugin_graph_preserved"):
        binding = baselines[key]
        require(sha(binding["path"]) == binding["sha256"], "Rollback artifact changed")
        protected[binding["path"]] = binding["sha256"]
    retire = []
    for run_name, prefix in (("hare-loss-ablation-zero-b16-250-001", "hare-loss-ablation-001"),
                             ("latency58-reduced-teacher-quarter-b16-250-001", "latency58-reduced-teacher-001")):
        run, stage = PHASE / run_name, PHASE / prefix
        status, audit, execution, score = [read(p) for p in (
            run / "status.json", stage / "audit.json", stage / "production/execution.json", stage / "sdr-comparison.json")]
        require(status["status"] == "complete" and status["step"] == audit["step"] == 250
                and audit["status"] == "pass" and execution["actual_exit_code"] == 0
                and score["status"] == "pass" and score["source_bindings_unchanged"]
                and score["full_mixture_aggregate"]["full_sdr_db"] < 4.069078803302578,
                "Optimizer belongs to an unfinished, unaudited, or selected trial")
        for proc in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                args = proc.read_bytes().split(b"\0")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            require(not any(value == str(run).encode() or value.startswith(str(run / "checkpoints").encode()) for value in args),
                    "Retired generation is still in a live command")
        generation = run / "checkpoints/step-000250"
        receipt = read(generation / "receipt.json")
        require(receipt["model_state_sha256"] == audit["model_state_sha256"] == score["model_state_sha256"],
                "Saved checkpoint and full14 report differ")
        for name, binding in receipt["files"].items():
            path = generation / name
            require(path.is_file() and not path.is_symlink() and path.stat().st_size == binding["bytes"]
                    and sha(path) == binding["sha256"], "Retirement candidate bytes changed")
            if name == "optimizer.pt":
                retire.append({"path": str(path), **binding, "model_state_sha256": receipt["model_state_sha256"]})
            else:
                protected[str(path)] = binding["sha256"]
        protected[str(generation / "receipt.json")] = sha(generation / "receipt.json")
    require(len(retire) == 2 and set(protected).isdisjoint(row["path"] for row in retire),
            "An optimizer is required by a preserved inference baseline")
    verify_inputs({"source_bindings": protected})
    budget = read(PHASE / "latency58-reduced-teacher-001/training-plan.json")
    before = sum(disk_bytes(Path(p)) for p in budget["counted_roots"])
    require(before - sum(r["bytes"] for r in retire) + 800_000_000 + 400_000_000 < 80_000_000_000,
            "Retirement cannot fund the bounded pilot")
    out.mkdir()
    write(out / "intent.json", {"retire": retire, "protected_bindings": protected, "counted_bytes_before": before,
          "reason": "Closed lower-SDR optimizer states; preserve every inference model, RNG, journal and receipt."})
    for row in retire:
        path = Path(row["path"])
        require(sha(path) == row["sha256"], "Optimizer changed before removal")
        path.unlink()
    verify_inputs({"source_bindings": protected})
    after = sum(disk_bytes(Path(p)) for p in budget["counted_roots"])
    write(out / "receipt.json", {"status": "complete", "intent_sha256": sha(out / "intent.json"),
          "retired": retire, "freed_bytes": sum(r["bytes"] for r in retire),
          "protected_bindings_unchanged": True, "counted_bytes_after": after,
          "forecast_including_outside_and_pilot": after + 800_000_000 + 400_000_000})
    print(json.dumps(read(out / "receipt.json")), flush=True)


if __name__ == "__main__":
    main()
