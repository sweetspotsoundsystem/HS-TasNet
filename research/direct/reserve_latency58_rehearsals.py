"""Retire four superseded two-update rehearsal tensors, preserving their metadata."""
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, write, sha, require
from research.direct.train_latency58 import verify_inputs


def main():
    import torch
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Use the CPU workspace")
    names = ("sdr-context-warm-to-000002-001", "sdr-context-reset-to-000002-001",
             "sdr-cropped11-to-000002-001", "sdr-c91-to-000002-001")
    plans = [read(PHASE / name) for name in ("full-magnitude-001/plan.json", "full-magnitude-001/full14/plan.json",
                                           "m4-int8-ort126-full14-001/plan.json")]
    protected = {path: digest for plan in plans for path, digest in plan["source_bindings"].items()}
    # Include the saved endpoint and its resumable optimizer explicitly.
    for path in (PHASE / "full-magnitude-001/production-run/checkpoint").glob("*"):
        if path.is_file():
            protected[str(path.resolve())] = sha(path)
    verify_inputs({"source_bindings": protected})
    out = PHASE / "rehearsal-storage-001"
    require(not out.exists(), "Preserve retirement evidence")
    before = require_space(plans[0], 0)
    out.mkdir()
    rows = []
    for name in names:
        audit_path = PHASE / name / "audit.json"
        audit = read(audit_path)
        path = Path(audit["checkpoint"]["path"])
        require(audit["status"] == "pass" and audit["step"] == 2 and path.parent.name == "step-000002"
                and path.name == "model.pt" and path.is_file() and not path.is_symlink()
                and path.is_relative_to(PHASE) and str(path) not in protected
                and sha(path) == audit["checkpoint"]["sha256"], "Invalid or protected rehearsal target")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        require(payload["step"] == 2 and payload["model_state_sha256"] == audit["model_state_sha256"],
                "Rehearsal identity differs")
        metadata = {key: value for key, value in payload.items() if key != "model"}
        preserved = out / (name + "-metadata.pt")
        with preserved.open("xb") as stream:
            torch.save(metadata, stream)
            stream.flush()
            os.fsync(stream.fileno())
        require(torch.load(preserved, map_location="cpu", weights_only=True) == metadata,
                "Metadata round trip differs")
        rows.append({"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size,
                     "step": 2, "audit": {"path": str(audit_path), "sha256": sha(audit_path)},
                     "metadata": {"path": str(preserved), "sha256": sha(preserved)},
                     "reason": "Superseded two-update rehearsal; later checkpoints and all audits retained"})
    intent = {"schema": "latency58-rehearsal-retirement-v1", "targets": rows, "counted_bytes_before": before,
              "protected_source_bindings": protected, "source_audio_touched": False,
              "accepted_models_and_current_checkpoint_preserved": True,
              "model_tensor_replay_for_retired_rehearsals_available_afterwards": False}
    write(out / "intent.json", intent)
    for row in rows:
        path = Path(row["path"])
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "Target changed before retirement")
        path.unlink()
    verify_inputs({"source_bindings": protected})
    after = require_space(plans[0], 370_000_000)
    receipt = {"status": "complete", "intent_sha256": sha(out / "intent.json"), "retired_files": len(rows),
               "retired_bytes": sum(row["bytes"] for row in rows), "counted_bytes_after": after,
               "forecast_including_outside_and_next_checkpoint": after + 800_000_000 + 370_000_000,
               "protected_sources_unchanged": True, "no_cuda": not torch.cuda.is_initialized()}
    write(out / "receipt.json", receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
