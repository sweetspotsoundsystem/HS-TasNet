"""Reclaim rebuildable Linux object files while retaining all native deliverables."""
import os
from pathlib import Path
import subprocess

from research.direct.run_latency58_quality import ROOT, PHASE, read, write, sha, require
from research.direct.train_latency58 import verify_inputs


def main():
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Use the CPU workspace")
    source = read(PHASE / "full-magnitude-fast16-001/plan.json")
    complete = read(PHASE / "full-magnitude-fast16-001/result.json")
    require(complete["status"] == "training_audit_and_full14_complete", "Current training is unfinished")
    native = Path("/home/axel/autoresearch/codex/stemgen-rt-hop128-5ms")
    build = native / "build-release"
    require(build.is_dir() and not build.is_symlink(), "Unexpected build root")
    tracked = subprocess.run(["git", "ls-files", "--", "build-release"], cwd=native,
                             capture_output=True, text=True, check=True).stdout
    require(not tracked.strip(), "Do not remove tracked build files")
    for process in Path("/proc").iterdir():
        if not process.name.isdecimal():
            continue
        try:
            arguments = (process / "cmdline").read_bytes().split(b"\0")
            executable = Path(os.fsdecode(arguments[0])).name if arguments[0] else ""
            if executable in {"cmake", "ninja", "make", "gmake", "cc1plus", "clang", "clang++", "g++", "ld", "ld.lld"}:
                require(False, "Wait for the active build process: " + process.name)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    protected = dict(source["source_bindings"])
    for plan_path in (PHASE / "full-magnitude-fast16-001/full14/plan.json",
                      PHASE / "m4-int8-ort126-full14-001/plan.json"):
        protected.update(read(plan_path)["source_bindings"])
    targets = []
    retained = {}
    for directory, dirs, files in os.walk(native, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(directory) / name).is_symlink()]
        for name in files:
            path = Path(directory) / name
            if path.is_symlink() or not path.is_file():
                continue
            if path.is_relative_to(build) and path.suffix == ".o":
                with path.open("rb") as stream:
                    header = stream.read(18)
                require(path.resolve() == path and str(path) not in protected and header[:4] == b"\x7fELF"
                        and header[5] == 1 and int.from_bytes(header[16:18], "little") == 1,
                        "Require an unprotected ELF relocatable object")
                targets.append({"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size})
            else:
                retained[str(path)] = sha(path)
    require(targets and sum(row["bytes"] for row in targets) >= 370_000_000, "Insufficient eligible build cache")
    protected.update(retained)
    verify_inputs({"source_bindings": protected})
    out = PHASE / "native-object-storage-001"
    require(not out.exists(), "Preserve storage receipts")
    before = require_space(source, 0)
    out.mkdir()
    write(out / "intent.json", {"targets": targets, "counted_bytes_before": before,
          "protected_source_bindings": protected, "source_audio_touched": False,
          "native_models_binaries_archives_sources_and_build_configuration_preserved": True,
          "recovery": "Recompile missing objects with cmake --build --preset release"})
    for row in targets:
        path = Path(row["path"])
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "Object changed before cleanup")
        path.unlink()
    verify_inputs({"source_bindings": protected})
    after = require_space(source, 370_000_000)
    receipt = {"status": "complete", "retired_files": len(targets), "retired_bytes": sum(r["bytes"] for r in targets),
               "intent_sha256": sha(out / "intent.json"), "counted_bytes_after": after,
               "forecast_including_outside_and_next_checkpoint": after + 800_000_000 + 370_000_000,
               "protected_sources_unchanged": True, "compiled_deliverables_unchanged": True}
    write(out / "receipt.json", receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
