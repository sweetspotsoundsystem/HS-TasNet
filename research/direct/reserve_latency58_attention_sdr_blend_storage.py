"""Reserve the SDR-blend trial after review, preserving models, audio and selected optimizers.

Prefer remaining generated plugin objects when sufficient. If necessary, retire
only the authenticated historical OLA step-1250 intermediate optimizer, then the
smallest prefix of largest eligible objects. Never remove source or test binaries.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess

from research.direct.run_latency58_quality import ROOT, PHASE, read, write, sha, require
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_sdr_checkpoint import require_space
from research.direct.reclaim_latency58_attention_build_objects import require_no_build

RESERVE, MANIFEST_MARGIN = 400_000_000, 4_000_000
OLA_STEP = 1250
OLA_RESUME_SHA = "43c6338a29b2b5a675f984c59fb1e471ea339197bdbeb3cd805da451010607d4"
OLA_RESUME_BYTES = 318_519_698


def no_historical_user(old):
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        require(str(old).encode() not in command, "A process still references the historical OLA run")


def main():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Use the CPU workspace")
    completed = PHASE / "attention-grouped-001"
    required = [completed / name for name in (
        "selection-review.json", "root-execution.json", "result.json", "production-stage/execution.json",
        "full14/execution.json", "full14/result.json", "checkpoint-audit.json")]
    require(all(path.is_file() for path in required),
            "Finish the continuation and saved full14 review before reserving the next trial")
    source_path = completed / "plan.json"
    source, review = read(source_path), read(completed / "selection-review.json")
    root_execution = read(completed / "root-execution.json")
    require(review["status"] in ("selected_for_research", "not_selected")
            and review["actual_root_exit_code"] == root_execution["actual_exit_code"] == 0
            and root_execution["source_bindings_unchanged"]
            and root_execution["result_sha256"] == sha(completed / "result.json")
            and 4.288099064999147 <= review["best_full_sdr_db"] < 5.0,
            "Require a closed, reviewed continuation below the target")
    verify_inputs(source)
    verify_inputs(review)
    for name in ("production-stage/execution.json", "full14/execution.json"):
        execution = read(completed / name)
        require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                and not execution.get("timed_out", False), "Prior execution did not close successfully")
    monitor_path = Path(read(completed / "production-stage/execution.json")["monitor_result"])
    monitor = read(monitor_path)
    require(monitor["status"] == monitor["supervisor_health"] == "pass" and monitor["child_exit_code"] == 0
            and monitor["post_exit_quiet_completed"], "The preceding GPU monitor must close successfully")
    worktree = PHASE / "best-model-stemgen-rt-001"
    require(worktree.resolve() == worktree and worktree.is_dir(), "Wrong published plugin worktree")
    require_no_build(worktree)
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=worktree, text=True).strip()
            == "c84805083c1127f6749f140ce8a49ae8b5acbf54"
            and not subprocess.check_output(["git", "status", "--porcelain"], cwd=worktree),
            "Preserve the clean published plugin candidate")
    previous_intent_path = PHASE / "attention-grouped-storage-001/intent.json"
    previous_receipt_path = previous_intent_path.parent / "receipt.json"
    previous_intent, previous_receipt = read(previous_intent_path), read(previous_receipt_path)
    require(previous_receipt["status"] == "complete" and previous_receipt["preserved_bindings_unchanged"]
            and previous_receipt["intent_sha256"] == sha(previous_intent_path), "Previous object cleanup is incomplete")
    memory_plan_path = PHASE / "attention-grouped-memory-functional-001/plan.json"
    memory_result_path = memory_plan_path.parent / "result.json"
    memory_execution_path = PHASE / "attention-grouped-memory-functional-stage-001/execution.json"
    memory_plan, memory, memory_execution = read(memory_plan_path), read(memory_result_path), read(memory_execution_path)
    require(memory["status"] == "pass" and memory["source_bindings_unchanged"]
            and memory_execution["actual_exit_code"] == 0 and memory_execution["source_bindings_unchanged"]
            and not memory_execution["timed_out"], "Require the completed grouped-optimizer checkpoint proof")
    verify_inputs(memory_plan)
    protected = {}
    for bindings in (previous_intent["preserved_bindings"], source["source_bindings"],
                     review["source_bindings"], memory_plan["source_bindings"]):
        for path, digest in bindings.items():
            require(path not in protected or protected[path] == digest, "Conflicting protected file identities")
            protected[path] = digest
    paths = [Path(__file__).resolve(), source_path, *required, monitor_path,
             previous_intent_path, previous_receipt_path, memory_plan_path, memory_result_path, memory_execution_path,
             completed / "production-run/checkpoint/model.pt", completed / "production-run/checkpoint/receipt.json"]
    if review["status"] == "selected_for_research":
        paths.append(completed / "production-run/checkpoint/optimizer.pt")
    paths.extend(ROOT / "research/direct" / name for name in (
        "reclaim_latency58_attention_build_objects.py", "prepare_latency58_attention_sdr_blend.py",
        "latency58_attention_sdr_blend.py", "train_latency58_attention_sdr_blend.py",
        "run_latency58_attention_sdr_blend.py"))
    for path in paths:
        digest = sha(path)
        require(str(path) not in protected or protected[str(path)] == digest, "A protected file changed")
        protected[str(path)] = digest
    verify_inputs({"source_bindings": protected})
    before = require_space(source, 0)
    limit = source["stop_counted_bytes"]
    build = worktree / "build-release"
    require(build.is_dir() and not build.is_symlink() and build.resolve() == build, "Wrong generated build directory")
    objects = sorted((path for path in build.rglob("*") if path.suffix in (".o", ".a")
                      and path.is_file() and not path.is_symlink() and path.resolve() == path
                      and path.stat().st_nlink == 1 and str(path) not in protected),
                     key=lambda path: (-path.stat().st_size, str(path)))
    object_bytes = sum(path.stat().st_size for path in objects)
    need_historical = before - object_bytes + RESERVE + MANIFEST_MARGIN >= limit
    chosen, reclaimed = [], 0
    old, lock = None, None
    if need_historical:
        old = ROOT / "research/direct/runs/latency11/ola512-right-baked-gpu-b4-bf16-projection01-lr3e-5"
        status, latest = read(old / "status.json"), read(old / "latest.json")
        require(status["status"] == "complete" and status["step"] == latest["step"] == 2000
                and status["latest_checkpoint"] == latest, "Historical training is incomplete")
        lock = (old / "trainer.lock").open("r")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        no_historical_user(old)
        terminal = Path(latest["generation"])
        require(terminal == old / "checkpoints/step-002000", "Historical terminal location changed")
        for name, digest in latest["files"].items():
            require(sha(terminal / name) == digest, "Historical terminal checkpoint changed")
            protected[str(terminal / name)] = digest
        require(sha(terminal / "receipt.json") == latest["receipt_sha256"], "Historical terminal receipt changed")
        protected[str(terminal / "receipt.json")] = latest["receipt_sha256"]
        for name in ("status.json", "latest.json", "config.json", "metrics.jsonl"):
            protected[str(old / name)] = sha(old / name)
        generation = old / "checkpoints" / f"step-{OLA_STEP:06d}"
        receipt = read(generation / "receipt.json")
        resume = generation / "resume.pt"
        require(receipt["status"] == "pass" and receipt["step"] == OLA_STEP
                and receipt["plan_sha256"] == latest["plan_sha256"]
                and resume.is_file() and not resume.is_symlink() and resume.resolve() == resume
                and resume.stat().st_nlink == 1 and resume.stat().st_size == OLA_RESUME_BYTES
                and str(resume) not in protected
                and sha(resume) == receipt["files"]["resume.pt"] == OLA_RESUME_SHA,
                "Historical intermediate resume identity or protection differs")
        for name in ("model.pt", "metrics.jsonl"):
            path = generation / name
            require(path.is_file() and not path.is_symlink() and sha(path) == receipt["files"][name],
                    "Historical inference model or journal changed")
            protected[str(path)] = receipt["files"][name]
        protected[str(generation / "receipt.json")] = sha(generation / "receipt.json")
        chosen.append({"kind": "historical_intermediate_optimizer", "path": str(resume),
                       "bytes": OLA_RESUME_BYTES, "sha256": OLA_RESUME_SHA, "step": OLA_STEP,
                       "preserved_inference_checkpoint": str(generation / "model.pt"),
                       "preserved_model_state_sha256": receipt["model_state_sha256"]})
        reclaimed = OLA_RESUME_BYTES
    chosen_objects = []
    for path in objects:
        if before - reclaimed + RESERVE + MANIFEST_MARGIN < limit:
            break
        item = {"kind": "generated_object", "path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}
        chosen_objects.append(item)
        chosen.append(item)
        reclaimed += item["bytes"]
    require(before - reclaimed + RESERVE + MANIFEST_MARGIN < limit,
            "The permitted historical resume and generated objects cannot fund the reservation")
    if chosen_objects:
        require(before - reclaimed + chosen_objects[-1]["bytes"] + RESERVE + MANIFEST_MARGIN >= limit,
                "Do not delete more generated objects than needed")
    out = PHASE / "attention-sdr-blend-storage-001"
    require(not out.exists(), "Preserve previous reservation records")
    require_no_build(worktree)
    if old is not None:
        no_historical_user(old)
    verify_inputs({"source_bindings": protected})
    out.mkdir()
    write(out / "intent.json", {"targets": chosen, "preserved_bindings": protected,
          "counted_bytes_before": before, "reserved_bytes": RESERVE, "manifest_margin_bytes": MANIFEST_MARGIN,
          "eligible_generated_bytes": object_bytes, "historical_resume_needed_after_all_eligible_objects": need_historical,
          "all_inference_models_and_source_audio_preserved": True, "all_selected_optimizers_preserved": True,
          "exact_native_test_binaries_preserved": True, "pr13_commit_preserved": "c84805083c1127f6749f140ce8a49ae8b5acbf54",
          "limitation": "If selected, the historical step-1250 intermediate optimizer/RNG snapshot will no longer support exact restart; its separate inference model, journals, receipt and step-2000 terminal resume remain unchanged. Generated objects are rebuildable from retained sources."})
    verify_inputs({"source_bindings": protected})
    for item in chosen:
        path = Path(item["path"])
        require(path.is_file() and not path.is_symlink() and path.resolve() == path
                and path.stat().st_nlink == 1 and path.stat().st_size == item["bytes"]
                and sha(path) == item["sha256"] and str(path) not in protected, "A cleanup target changed")
        if item["kind"] == "generated_object":
            require(path.is_relative_to(build) and path.suffix in (".o", ".a"), "Invalid generated target")
        else:
            require(old is not None and path == old / "checkpoints/step-001250/resume.pt", "Invalid historical target")
        path.unlink()
    verify_inputs({"source_bindings": protected})
    require(all(not Path(item["path"]).exists() for item in chosen), "Reservation cleanup is incomplete")
    result = {"status": "complete", "intent_sha256": sha(out / "intent.json"), "removed": chosen,
              "reclaimed_bytes": reclaimed, "preserved_bindings_unchanged": True,
              "all_inference_models_and_source_audio_preserved": True, "all_selected_optimizers_preserved": True,
              "native_test_binaries_preserved": True, "historical_terminal_resume_preserved": True,
              "historical_intermediate_resume_retired": need_historical, "reserved_bytes": RESERVE,
              "counted_bytes_after": require_space(source, RESERVE)}
    write(out / "receipt.json", result)
    if lock is not None:
        lock.close()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
