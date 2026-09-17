"""Retire explicitly named rejected artifacts to fund full-model magnitude training."""
from __future__ import annotations

import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import disk_bytes, verify_inputs
from research.direct.reserve_latency58_wave_spectral import same


def main():
    import torch
    torch.set_num_threads(1)
    out = PHASE / "full-magnitude-storage-001"
    require(Path.cwd() == ROOT and not out.exists(), "Preserve storage evidence")
    targets = {
        PHASE / "wave-spectral-001/production-run/checkpoint/optimizer.pt": "optimizer",
        PHASE / "direct-sdr-musdb-001/production-run/checkpoint/model.pt": "model",
        PHASE / "direct-sdr-001/production-run/checkpoint/model.pt": "model",
    }
    protected = {str(Path(__file__).resolve()): sha(__file__)}
    rejected = []
    for target in targets:
        root = target.parents[2]
        result, execution, audit = (read(root / p) for p in ("result.json", "full14/execution.json", "checkpoint-audit.json"))
        require(result["status"] == "training_audit_and_full14_complete" and not result["target_reached"]
                and result["full_sdr_db"] < 4.0846618609770395 and execution["actual_exit_code"] == 0
                and not execution["timed_out"] and execution["source_bindings_unchanged"] and audit["status"] == "pass",
                "Only closed, rejected lower-SDR experiments may be retired")
        rejected.append({"name": root.name, "full_sdr_db": result["full_sdr_db"]})
        training = read(root / "production-stage/execution.json")
        monitor = read(training["monitor_result"])
        require(training["actual_exit_code"] == 0 and training["source_bindings_unchanged"]
                and monitor["status"] == monitor["supervisor_health"] == "pass"
                and monitor["child_exit_code"] == 0 and monitor["post_exit_quiet_completed"],
                "Rejected training stage must be closed")
        for relative in ("plan.json", "result.json", "checkpoint-audit.json", "full14/plan.json", "full14/result.json",
                         "full14/execution.json", "production-stage/execution.json", "production-run/result.json"):
            path = root / relative
            protected.update(read(path).get("source_bindings", {}))
            protected[str(path)] = sha(path)
        for path in target.parent.iterdir():
            protected[str(path)] = sha(path)
    # Historical scoring plans bind the rejected model bytes. Keep those plans
    # unchanged; the new receipt explicitly records loss of replay for these two.
    superseded = {str(p): protected.pop(str(p), None) for p in targets}
    baselines = read(PHASE / "c204-residual-model-001/final-review.json")
    for key in ("checkpoint", "onnx", "c204_checkpoint_preserved", "working_plugin_graph_preserved"):
        binding = baselines[key]
        require(Path(binding["path"]) not in targets and sha(binding["path"]) == binding["sha256"], "Rollback baseline differs")
        protected[binding["path"]] = binding["sha256"]
    bindings = {}
    for target in targets:
        binding = read(target.parent / "receipt.json")["files"][target.name]
        require(target.is_file() and not target.is_symlink() and sha(target) == binding["sha256"]
                and target.stat().st_size == binding["bytes"], "Retirement target differs")
        bindings[str(target)] = binding
    for proc in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            arguments = proc.read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        require(not any(arg.startswith(str(p.parent).encode()) for p in targets for arg in arguments),
                "A live process uses a retirement target")
    verify_inputs({"source_bindings": protected})
    budget = read(PHASE / "wave-spectral-001/plan.json")
    before = sum(disk_bytes(Path(p)) for p in budget["counted_roots"])
    require(before - sum(b["bytes"] for b in bindings.values()) + 800_000_000 + 370_000_000 + 2_000_000 < 80_000_000_000,
            "Retirement cannot fund the next run")
    out.mkdir()
    write(out / "intent.json", {"targets": bindings, "superseded_failed_model_bindings": superseded,
          "protected_bindings": protected, "counted_bytes_before": before, "rejected_scores": rejected,
          "reason": "Retain accepted baselines, source audio and all negative-result evidence; retire two failed model weight sets and a closed failed optimizer."})
    retained = []
    for target, kind in targets.items():
        payload = torch.load(target, map_location="cpu", weights_only=True)
        metadata = {k: v for k, v in payload.items() if k != kind}
        saved = out / (target.parents[2].name + ("-rng.pt" if kind == "optimizer" else "-metadata.pt"))
        with saved.open("xb") as stream:
            torch.save(metadata, stream)
            stream.flush()
            os.fsync(stream.fileno())
        require(same(metadata, torch.load(saved, map_location="cpu", weights_only=True)), "Metadata round trip differs")
        retained.append({"path": str(saved), "sha256": sha(saved), "omitted_payload_key": kind})
        del payload, metadata
    verify_inputs({"source_bindings": protected})
    for target in targets:
        require(sha(target) == bindings[str(target)]["sha256"], "Target changed before retirement")
    for target in targets:
        target.unlink()
    verify_inputs({"source_bindings": protected})
    after = sum(disk_bytes(Path(p)) for p in budget["counted_roots"])
    write(out / "receipt.json", {"status": "complete", "intent_sha256": sha(out / "intent.json"),
          "retired": bindings, "retained_metadata_and_rng": retained, "metadata_round_trip_exact": True,
          "failed_model_replay_unavailable": [str(p) for p, kind in targets.items() if kind == "model"],
          "wave_spectral_adam_resume_available": False, "protected_bindings_unchanged": True,
          "counted_bytes_after": after, "forecast_including_outside_and_full_magnitude": after + 800_000_000 + 370_000_000})
    print(json.dumps(read(out / "receipt.json")), flush=True)


if __name__ == "__main__":
    main()
