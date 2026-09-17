"""Recover space from one historical intermediate OLA training resume.

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


TARGETS = {1025: "4c4a89dd2a31fb131ad63c5118918637ce5407f7ab2a439ccf364c1423c09f38"}


def main():
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Use CPU workspace")
    source_path = PHASE / "quadrature-continuation-001/plan.json"
    source = read(source_path)
    verify_inputs(source)
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
    for relative in ("m4-quadrature-memory-native-001/plan.json",
                     "m4-quadrature-memory-native-001/selection-review.json",
                     "m4-quadrature-magint8-full14-memory-001/plan.json"):
        path = PHASE / relative
        record = read(path)
        verify_inputs(record)
        for bound, digest in record["source_bindings"].items():
            require(bound not in bindings or bindings[bound] == digest, "Conflicting preserved identity")
            bindings[bound] = digest
        bindings[str(path)] = sha(path)
    for name in ("status.json", "latest.json", "config.json", "metrics.jsonl"):
        bindings[str(old / name)] = sha(old / name)
    terminal = Path(latest["generation"])
    require(terminal == old / "checkpoints/step-002000", "Terminal location changed")
    for name, digest in latest["files"].items():
        require(sha(terminal / name) == digest, "Preserved terminal checkpoint differs")
        bindings[str(terminal / name)] = digest
    require(sha(terminal / "receipt.json") == latest["receipt_sha256"], "Terminal receipt changed")
    bindings[str(terminal / "receipt.json")] = latest["receipt_sha256"]
    for parent in ("full-magnitude-001", "remix-magnitude-001", "quadrature-001"):
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
    out = PHASE / "historical-ola-resume-retirement-002"
    require(not out.exists(), "Preserve earlier retirement records")
    before = require_space(source, 0)
    out.mkdir()
    write(out / "intent.json", {"targets": targets, "preserved_bindings": bindings,
          "counted_bytes_before": before, "all_inference_models_and_source_audio_preserved": True,
          "terminal_training_resume_preserved": True,
          "limitation": "The historical step-1025 optimizer/RNG snapshot will no longer be available for exact training restarts. Its separate inference checkpoint, journals and original receipts remain byte-identical."})
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
              "counted_bytes_after": require_space(source, 477_000_000), "reserved_bytes": 477_000_000}
    write(out / "receipt.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
