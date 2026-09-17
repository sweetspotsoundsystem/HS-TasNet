"""Retire only the two completed, unselected continuation optimizers."""
from pathlib import Path
import os
import json

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs


def main():
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Use the CPU workspace")
    source_path = PHASE / "full-magnitude-sdr-001/plan.json"
    source = read(source_path)
    active_path = PHASE / "m4-int8-precise-core-full14-001/plan.json"
    active = read(active_path)
    verify_inputs(source)
    verify_inputs(active)
    before = require_space(source, 0)
    trials = [
        ("full-magnitude-fast16-001", 4000, 4.099680774398102,
         "f3ebc30b9a6f8fdc14905b56599ff0789a986c267a60bbd826ba40713768fde2"),
        ("full-magnitude-sdr-001", 2000, 4.0584543602451255,
         "8809c4574029143380073255d15ce3653821329dcc185baaff61c665eada9b0e"),
    ]
    bindings = {**source["source_bindings"], **active["source_bindings"],
                str(source_path): sha(source_path), str(active_path): sha(active_path),
                str(Path(__file__).resolve()): sha(__file__)}
    protected = [PHASE / "full-magnitude-001/production-run/checkpoint/model.pt",
                 PHASE / "full-magnitude-001/production-run/checkpoint/optimizer.pt"]
    bindings.update({str(p): sha(p) for p in protected})
    targets = []
    for name, steps, score, expected_sha in trials:
        root = PHASE / name
        result, audit = read(root / "result.json"), read(root / "checkpoint-audit.json")
        generation = root / "production-run/checkpoint"
        receipt = read(generation / "receipt.json")
        optimizer = generation / "optimizer.pt"
        model = generation / "model.pt"
        require(result["status"] == "training_audit_and_full14_complete"
                and result["full_sdr_db"] == score < 4.114394535058717
                and not result["target_reached"] and not result["plugin_replaced"]
                and audit["status"] == "pass" and audit["step"] == steps
                and read(root / "production-stage/execution.json")["actual_exit_code"] == 0
                and read(root / "full14/execution.json")["actual_exit_code"] == 0,
                "Require a closed, unselected lower-scoring run")
        require(optimizer.is_file() and not optimizer.is_symlink() and str(optimizer) not in bindings
                and optimizer.stat().st_size == 226_722_452
                and sha(optimizer) == expected_sha == audit["optimizer_sha256"]
                == receipt["files"]["optimizer.pt"]["sha256"], "Optimizer identity or live binding differs")
        require(sha(model) == audit["checkpoint"]["sha256"] == result["checkpoint"]["sha256"]
                == receipt["files"]["model.pt"]["sha256"], "Preserved model differs")
        paths = [model, generation / "receipt.json", root / "result.json", root / "checkpoint-audit.json",
                 root / "production-stage/execution.json", root / "full14/execution.json", root / "full14/result.json"]
        bindings.update({str(p): sha(p) for p in paths})
        targets.append({"path": str(optimizer), "sha256": expected_sha, "bytes": optimizer.stat().st_size,
                        "preserved_model": str(model), "full_sdr_db": score, "optimizer_resume_retained": False})
    out = PHASE / "failed-full-optimizers-retirement-001"
    require(not out.exists(), "Preserve earlier retirement records")
    out.mkdir()
    write(out / "intent.json", {"targets": targets, "preserved_bindings": bindings,
          "counted_bytes_before": before, "source_audio_and_all_model_weights_preserved": True})
    verify_inputs({"source_bindings": bindings})
    for target in targets:
        path = Path(target["path"])
        require(path.is_file() and not path.is_symlink() and sha(path) == target["sha256"], "Target changed before removal")
        path.unlink()
    verify_inputs({"source_bindings": bindings})
    require(all(not Path(t["path"]).exists() for t in targets), "Retirement incomplete")
    receipt = {"status": "complete", "intent_sha256": sha(out / "intent.json"),
               "removed": targets, "reclaimed_bytes": sum(t["bytes"] for t in targets),
               "preserved_bindings_unchanged": True, "source_audio_and_all_model_weights_preserved": True,
               "counted_bytes_after": require_space(source, 380_000_000),
               "next_training_reservation_bytes": 380_000_000}
    write(out / "receipt.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
