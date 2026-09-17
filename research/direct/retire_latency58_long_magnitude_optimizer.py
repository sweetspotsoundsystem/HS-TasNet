"""Reclaim only the closed, rejected long-history trial's optimizer snapshot."""
import fcntl
import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs


def main():
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Use the CPU workspace")
    source_path = PHASE / "quadrature-001/plan.json"
    source = read(source_path)
    active_paths = [PHASE / name for name in (
        "quadrature-001/selection-review.json", "m4-remix-int8-full14-001/plan.json",
        "m4-remix-candidates-native-001/plan.json")]
    bindings = {**source["source_bindings"], str(source_path): sha(source_path),
                str(Path(__file__).resolve()): sha(__file__)}
    for path in active_paths:
        record = read(path)
        verify_inputs(record)
        bindings.update(record["source_bindings"])
        bindings[str(path)] = sha(path)
    root = PHASE / "long-magnitude-001"
    result, review, audit = (read(root / name) for name in
                            ("result.json", "selection-review.json", "checkpoint-audit.json"))
    require(result["status"] == "training_audit_and_full14_complete"
            and result["full_sdr_db"] == 4.082512807757332 and not result["target_reached"]
            and not result["plugin_replaced"] and review["status"] == "not_selected"
            and audit["status"] == "pass" and audit["step"] == 2000,
            "Require the completed, rejected long-history endpoint")
    for name in ("production-stage/execution.json", "full14/execution.json"):
        execution = read(root / name)
        require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"],
                "The rejected trial did not close successfully")
    lock = (root / "production-run/trainer.lock").open("r")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        require(str(root).encode() not in command, "A process still references this trial")
    generation = root / "production-run/checkpoint"
    receipt = read(generation / "receipt.json")
    target, model = generation / "optimizer.pt", generation / "model.pt"
    expected = "8e73f9d58537cc61dd88b0ee3a4a2d9da6d63f8e40c5a3f11e3be6f43b746977"
    require(receipt["schema"] == "latency58-long-magnitude-generation-v1" and receipt["step"] == 2000
            and target.is_file() and not target.is_symlink() and target.stat().st_size == 229_675_211
            and str(target) not in bindings
            and sha(target) == expected == receipt["files"]["optimizer.pt"]["sha256"]
            == audit["optimizer_sha256"], "Optimizer identity or active binding differs")
    require(sha(model) == receipt["files"]["model.pt"]["sha256"]
            == audit["checkpoint"]["sha256"] == result["checkpoint"]["sha256"], "Inference model changed")
    paths = [root / name for name in (
        "result.json", "selection-review.json", "checkpoint-audit.json", "production-stage/execution.json",
        "full14/execution.json", "full14/result.json", "production-run/metrics.jsonl",
        "production-run/checkpoint/receipt.json", "production-run/checkpoint/model.pt")]
    paths.extend(PHASE / parent / "production-run/checkpoint" / name
                 for parent in ("full-magnitude-001", "remix-magnitude-001", "quadrature-001")
                 for name in ("model.pt", "optimizer.pt"))
    bindings.update({str(path): sha(path) for path in paths})
    require(str(target) not in bindings, "Target became a preserved input")
    verify_inputs({"source_bindings": bindings})
    before = require_space(source, 0)
    out = PHASE / "long-magnitude-optimizer-retirement-001"
    require(not out.exists(), "Preserve retirement records")
    out.mkdir()
    removed = {"path": str(target), "sha256": expected, "bytes": target.stat().st_size,
               "preserved_model": str(model), "optimizer_resume_retained": False}
    write(out / "intent.json", {"target": removed, "preserved_bindings": bindings,
          "counted_bytes_before": before, "limitation": "Exact optimizer/RNG restart of the rejected long-history trial will no longer be available. Its original receipt, review and all inference weights remain unchanged; this availability receipt supersedes their historical optimizer-retained statement."})
    require(target.is_file() and not target.is_symlink() and sha(target) == expected, "Target changed before removal")
    target.unlink()
    verify_inputs({"source_bindings": bindings})
    require(not target.exists(), "Retirement incomplete")
    write(out / "receipt.json", {"status": "complete", "intent_sha256": sha(out / "intent.json"),
          "removed": removed, "reclaimed_bytes": removed["bytes"], "preserved_bindings_unchanged": True,
          "all_inference_weights_and_source_audio_preserved": True,
          "selected_parent_optimizers_preserved": True,
          "counted_bytes_after": require_space(source, 452_000_000),
          "next_training_and_archive_reservation_bytes": 452_000_000})
    print(json.dumps(read(out / "receipt.json")), flush=True)


if __name__ == "__main__":
    main()
