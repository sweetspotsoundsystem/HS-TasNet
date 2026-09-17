"""Reclaim only enough newly generated Linux object files to reserve the next quality trial.

Preserve all model weights, optimizer checkpoints, source audio, sources and the
exact native test binaries used for the published measurements. Removed object
files and static archives can be regenerated from retained sources/dependencies.
"""
import json
import os
from pathlib import Path
import subprocess

from research.direct.run_latency58_quality import ROOT, PHASE, read, write, sha, require
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_sdr_checkpoint import require_space

RESERVE = 420_000_000
MANIFEST_MARGIN = 5_000_000


def require_no_build(worktree):
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            argv = (process / "cmdline").read_bytes().split(b"\0")
            cwd = (process / "cwd").resolve(strict=True)
        except (OSError, ProcessLookupError, PermissionError):
            continue
        if not argv or not argv[0]:
            continue
        binary = Path(os.fsdecode(argv[0])).name
        if cwd.is_relative_to(worktree) and (binary in {
            "cmake", "ninja", "make", "gmake", "c++", "g++", "clang++", "cc1plus", "cc1",
            "collect2", "ld", "ld.lld", "lto1", "lto-wrapper", "AudioPluginTest", "RealtimeSafetyTest"}):
            raise RuntimeError("A build or native test still uses the new worktree")


def main():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Use the CPU workspace")
    worktree = PHASE / "best-model-stemgen-rt-001"
    require(worktree.resolve() == worktree and worktree.is_dir(), "Wrong plugin worktree")
    require_no_build(worktree)
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=worktree, text=True).strip()
            == "c84805083c1127f6749f140ce8a49ae8b5acbf54"
            and not subprocess.check_output(["git", "status", "--porcelain"], cwd=worktree),
            "Preserve the published clean plugin candidate")
    source_path = PHASE / "temporal-attention-001/plan.json"
    source = read(source_path)
    verify_inputs(source)
    functional_root = PHASE / "attention-continuation-memory-functional-001"
    functional, functional_plan = read(functional_root / "result.json"), read(functional_root / "plan.json")
    functional_execution = PHASE / "attention-continuation-memory-functional-stage-001/execution.json"
    execution = read(functional_execution)
    require(functional["status"] == "pass" and functional["source_bindings_unchanged"]
            and execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
            and not execution["timed_out"], "Require the trained-parent continuation proof")
    verify_inputs(functional_plan)
    validation = PHASE / "attention-plugin-native-validation-001"
    native_plan = read(validation / "plan-v2.json")
    native_review = read(validation / "review.json")
    ci_path = PHASE / "attention-plugin-ci-001/result.json"
    ci = read(ci_path)
    require(ci["status"] == "pass" and ci["head_sha"] == "c84805083c1127f6749f140ce8a49ae8b5acbf54"
            and ci["macos_passed"] and ci["windows_passed"]
            and native_review["status"] == "correctness_pass_timing_unqualified"
            and native_review["source_bindings_unchanged"], "Plugin verification has not completed")
    execution_paths = []
    for relative, expected in (("correctness-stage-001", 0), ("direct-timing-stage-001", 1), ("paced-timing-stage-001", 1)):
        path = validation / relative / "execution.json"
        result = read(path)
        require(result["actual_exit_code"] == expected and result["source_bindings_unchanged"]
                and not result["timed_out"], "Native measurements must be closed, retaining timing failures")
        execution_paths.append(path)
    protected = {**source["source_bindings"], **functional_plan["source_bindings"],
                 **{str(worktree / p): digest for p, digest in
                    {**native_plan["source_bindings"], **native_plan["binary_sha256"]}.items()}}
    paths = [Path(__file__).resolve(), source_path, functional_root / "plan.json", functional_root / "result.json",
             functional_execution, validation / "plan-v2.json", validation / "review.json", ci_path, *execution_paths]
    paths.extend(ROOT / "research/direct" / name for name in (
        "prepare_latency58_attention_continuation.py", "train_latency58_temporal_attention_continuation.py",
        "run_latency58_temporal_attention_continuation.py", "check_latency58_attention_continuation_memory.py"))
    for parent in ("full-magnitude-001", "remix-magnitude-001", "quadrature-001", "quadrature-continuation-001",
                   "fusion-refinement-001", "temporal-attention-001"):
        paths.extend(PHASE / parent / "production-run/checkpoint" / name for name in ("model.pt", "optimizer.pt"))
    paths.extend((worktree / "model/model.onnx", PHASE / "best-model-onnx-saved-001/model.onnx",
                  PHASE / "best-model-onnx-saved-001/result.json", PHASE / "attention-model-pr-created.json"))
    protected.update({str(path): sha(path) for path in paths})
    require(sha(worktree / "model/model.onnx") == native_plan["model_sha256"], "Plugin model changed")
    verify_inputs({"source_bindings": protected})
    before = require_space(source, 0)
    require(before + RESERVE + MANIFEST_MARGIN >= source["stop_counted_bytes"], "Object cleanup is not needed")
    build = worktree / "build-release"
    require(build.is_dir() and not build.is_symlink() and build.resolve().is_relative_to(worktree), "Wrong generated build path")
    candidates = sorted((p for p in build.rglob("*") if p.suffix in (".o", ".a")
                         and p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(build)),
                        key=lambda p: (-p.stat().st_size, str(p)))
    chosen, reclaimed = [], 0
    for path in candidates:
        require(str(path) not in protected, "A generated object is bound as preserved evidence")
        chosen.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)})
        reclaimed += path.stat().st_size
        if before - reclaimed + RESERVE + MANIFEST_MARGIN < source["stop_counted_bytes"]:
            break
    require(chosen and before - reclaimed + RESERVE + MANIFEST_MARGIN < source["stop_counted_bytes"]
            and before - reclaimed + chosen[-1]["bytes"] + RESERVE + MANIFEST_MARGIN >= source["stop_counted_bytes"],
            "Require sufficient space using the fewest largest generated files")
    out = PHASE / "attention-plugin-build-object-retirement-001"
    require(not out.exists(), "Preserve previous cleanup records")
    require_no_build(worktree)
    out.mkdir()
    write(out / "intent.json", {"targets": chosen, "preserved_bindings": protected,
          "counted_bytes_before": before, "reserved_training_bytes": RESERVE, "manifest_margin_bytes": MANIFEST_MARGIN,
          "only_generated_object_files_and_static_archives": True, "all_source_audio_and_checkpoints_preserved": True,
          "exact_test_binaries_preserved": True, "pr13_commit_preserved": "c84805083c1127f6749f140ce8a49ae8b5acbf54",
          "rebuild_effect": "A later local build recompiles missing objects and relinks missing static archives. Existing test binaries and the published plugin source/model remain unchanged."})
    verify_inputs({"source_bindings": protected})
    for item in chosen:
        path = Path(item["path"])
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(build)
                and path.suffix in (".o", ".a") and path.stat().st_size == item["bytes"]
                and sha(path) == item["sha256"], "Generated object changed before cleanup")
        path.unlink()
    verify_inputs({"source_bindings": protected})
    require(all(not Path(item["path"]).exists() for item in chosen), "Generated object cleanup is incomplete")
    result = {"status": "complete", "intent_sha256": sha(out / "intent.json"), "removed": chosen,
              "reclaimed_bytes": reclaimed, "preserved_bindings_unchanged": True,
              "all_inference_models_and_source_audio_preserved": True, "all_optimizer_checkpoints_preserved": True,
              "native_test_binaries_preserved": True, "reserved_bytes": RESERVE,
              "counted_bytes_after": require_space(source, RESERVE)}
    write(out / "receipt.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
