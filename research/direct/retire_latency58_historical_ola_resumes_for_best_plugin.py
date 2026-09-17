"""Recover space from two historical intermediate OLA training resumes.

Keep every inference checkpoint, all journals and receipts, and the terminal
training resume. These intermediate optimizer/RNG snapshots are not used by
the current hop128 experiment or its rollback checkpoints.
"""
from pathlib import Path
import fcntl
import json
import os

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs


TARGETS = {
    1175: "468be6448f4e863679684f54a36003a596374f4b497f2b66c0b96a8b45b445ee",
    1200: "c3e33070db43672b10b269c5edd93ad1118ecaf90fdafd1f5fa7a06a77907920",
}


def main():
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Use CPU workspace")
    source_path = PHASE / "temporal-attention-001/plan.json"
    source = read(source_path)
    verify_inputs(source)
    previous = PHASE / "temporal-attention-001"
    review = read(previous / "selection-review.json")
    result = read(previous / "result.json")
    require(result["status"] == "training_audit_and_full14_complete"
            and review["status"] == "selected_for_research" and review["actual_root_exit_code"] == 0
            and review["best_research_checkpoint"] == result["checkpoint"], "Require the selected completed attention trial")
    for path in (previous / "production-stage/execution.json", previous / "full14/execution.json",
                 PHASE / "quadrature-all-s8-quiet-native-stage-001/execution.json"):
        execution = read(path)
        require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"],
                "Training, scoring or quiet native timing did not close successfully")
    quiet = read(PHASE / "quadrature-all-s8-quiet-native-001/analysis.json")
    require(quiet["status"] == "pass" and quiet["actual_root_exit_code"] == 0
            and not quiet["our_training_and_scoring_concurrent"], "Require the completed quiet comparison")
    old = ROOT / "research/direct/runs/latency11/ola512-right-baked-gpu-b4-bf16-projection01-lr3e-5"
    status, latest = read(old / "status.json"), read(old / "latest.json")
    require(status["status"] == "complete" and status["step"] == latest["step"] == 2000
            and status["latest_checkpoint"] == latest, "Historical training is not complete")
    lock = (old / "trainer.lock").open("r")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        require(str(old).encode() not in command, "A process still references the historical run")
    bindings = {**source["source_bindings"], str(source_path): sha(source_path),
                str(Path(__file__).resolve()): sha(__file__)}
    for name in ("check_latency58_attention_int8.py", "check_latency58_attention_int8_long.py",
                 "evaluate_latency58_attention_int8_memory.py", "latency58_attention_int8.py",
                 "latency58_attention_int8_precision.py", "latency58_attention_int8_verify.py",
                 "save_latency58_best_onnx.py", "latency58_best_onnx.py",
                 "latency58_best_onnx_export.py", "latency58_best_onnx_verify.py"):
        path = ROOT / "research/direct" / name
        bindings[str(path)] = sha(path)
    for relative in ("quadrature-continuation-001/selection-review.json",
                     "fusion-refinement-001/selection-review.json",
                     "temporal-attention-001/selection-review.json",
                     "attention-int8-screen-001/plan.json",
                     "attention-int8-long-001/plan.json",
                     "attention-int8-full14-memory-001/plan.json",
                     "best-model-onnx-memory-001/plan.json",
                     "best-model-onnx-long-001/plan.json",
                     "temporal-attention-functional-001/plan.json",
                     "temporal-attention-checkpoint-memory-functional-001/plan.json",
                     "quadrature-all-s8-quiet-native-001/plan.json",
                     "quadrature-all-s8-quiet-native-001/analysis.json",
                     "quadrature-all-s8-quiet-native-stage-001/command.json",
                     "m4-quadrature-magint8-saved-001/plan.json",
                     "m4-quadrature-candidates-benchmark-package-001/plan.json"):
        path = PHASE / relative
        record = read(path)
        verify_inputs(record)
        for bound, digest in record["source_bindings"].items():
            require(bound not in bindings or bindings[bound] == digest, "Conflicting preserved identity")
            bindings[bound] = digest
        bindings[str(path)] = sha(path)
    for prefix in ("attention-int8-screen", "attention-int8-long"):
        result_path = PHASE / (prefix + "-001") / "result.json"
        execution_path = PHASE / (prefix + "-stage-001") / "execution.json"
        proof, execution = read(result_path), read(execution_path)
        require(proof["status"] == "pass" and proof["source_bindings_unchanged"]
                and proof["checkpoint"] == result["checkpoint"]
                and execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                and not execution["timed_out"], "The selected model's export checks must have completed")
        bindings.update({str(p): sha(p) for p in (result_path, execution_path)})
    quality_path = PHASE / "attention-int8-full14-memory-001/result.json"
    quality_execution_path = PHASE / "attention-int8-full14-memory-stage-001/execution.json"
    quality, quality_execution = read(quality_path), read(quality_execution_path)
    require(quality["status"] == "pass" and quality["quality_handoff_gate_passed"]
            and quality["source_bindings_unchanged"] and quality["track_count"] == 14 and quality["excerpt_count"] == 28
            and quality["graph_sha256"] == read(PHASE / "attention-int8-screen-001/result.json")["graph_sha256"]
            and quality_execution["actual_exit_code"] == 0 and quality_execution["source_bindings_unchanged"]
            and not quality_execution["timed_out"], "Require complete passing deployment quality before reclaiming handoff space")
    bindings.update({str(p): sha(p) for p in (quality_path, quality_execution_path)})
    for name in ("status.json", "latest.json", "config.json", "metrics.jsonl"):
        bindings[str(old / name)] = sha(old / name)
    terminal = Path(latest["generation"])
    require(terminal == old / "checkpoints/step-002000", "Terminal location changed")
    for name, digest in latest["files"].items():
        require(sha(terminal / name) == digest, "Preserved terminal checkpoint differs")
        bindings[str(terminal / name)] = digest
    require(sha(terminal / "receipt.json") == latest["receipt_sha256"], "Terminal receipt changed")
    bindings[str(terminal / "receipt.json")] = latest["receipt_sha256"]
    for parent in ("full-magnitude-001", "remix-magnitude-001", "quadrature-001", "quadrature-continuation-001", "fusion-refinement-001", "temporal-attention-001"):
        for name in ("model.pt", "optimizer.pt"):
            path = PHASE / parent / "production-run/checkpoint" / name
            bindings[str(path)] = sha(path)
    targets = []
    for step, expected in TARGETS.items():
        generation = old / "checkpoints" / f"step-{step:06d}"
        receipt = read(generation / "receipt.json")
        path = generation / "resume.pt"
        require(receipt["status"] == "pass" and receipt["step"] == step
                and receipt["plan_sha256"] == latest["plan_sha256"], "Historical generation differs")
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == 318_519_698
                and str(path) not in bindings
                and sha(path) == expected == receipt["files"]["resume.pt"], "Resume identity/binding differs")
        for name in ("model.pt", "metrics.jsonl"):
            preserved = generation / name
            require(preserved.is_file() and not preserved.is_symlink()
                    and sha(preserved) == receipt["files"][name], "Preserved inference/journal differs")
            bindings[str(preserved)] = receipt["files"][name]
        bindings[str(generation / "receipt.json")] = sha(generation / "receipt.json")
        targets.append({"path": str(path), "sha256": expected, "bytes": path.stat().st_size,
                        "step": step, "preserved_inference_checkpoint": str(generation / "model.pt"),
                        "preserved_model_state_sha256": receipt["model_state_sha256"]})
    out = PHASE / "historical-ola-resume-retirement-005"
    require(not out.exists(), "Preserve earlier retirement records")
    before = require_space(source, 0)
    reclaimed = sum(t["bytes"] for t in targets)
    require(before + 800_000_000 + 600_000_000 > 80_000_000_000
            and before - reclaimed + 800_000_000 + 600_000_000 < 80_000_000_000,
            "Retirement must be necessary and sufficient for the handoff reservation")
    require(before - reclaimed + min(t["bytes"] for t in targets) + 800_000_000 + 600_000_000 > 80_000_000_000,
            "Do not retire more intermediate snapshots than the handoff needs")
    out.mkdir()
    write(out / "intent.json", {"targets": targets, "preserved_bindings": bindings,
          "counted_bytes_before": before, "all_inference_models_and_source_audio_preserved": True,
          "next_handoff": {"family": "temporal_attention_signed_integer", "checkpoint": result["checkpoint"],
                           "graph_sha256": read(PHASE / "attention-int8-screen-001/result.json")["graph_sha256"],
                           "reservation_bytes": 600_000_000,
                           "scope": "Exact graph export, plugin source worktree, model copy, native build, fixtures and prospective Git LFS object; selected-model M4/M4 Pro PR authorized by user."},
          "terminal_training_resume_preserved": True,
          "limitation": "The historical steps 1175 and 1200 optimizer/RNG snapshots will no longer be available for exact training restarts. Their separate inference checkpoints, journals and original receipts remain byte-identical."})
    verify_inputs({"source_bindings": bindings})
    for target in targets:
        path = Path(target["path"])
        require(path.is_file() and not path.is_symlink() and sha(path) == target["sha256"], "Resume changed before retirement")
        path.unlink()
    verify_inputs({"source_bindings": bindings})
    require(all(not Path(t["path"]).exists() for t in targets), "Retirement incomplete")
    result = {"status": "complete", "intent_sha256": sha(out / "intent.json"), "removed": targets,
              "reclaimed_bytes": sum(t["bytes"] for t in targets), "preserved_bindings_unchanged": True,
              "all_inference_models_and_source_audio_preserved": True, "terminal_training_resume_preserved": True,
              "counted_bytes_after": require_space(source, 600_000_000), "reserved_bytes": 600_000_000}
    write(out / "receipt.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
