"""Reclaim authenticated generated objects for the explicitly allocated trial."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess

import psutil

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_weighted_storage import policy, snapshot

OUT = PHASE / "weighted-vocal-quarter-storage-001"
RECORD_ALLOWANCE = 2_000_000
POST_RECORD_MARGIN = 2_000_000
BUILD_NAMES = {"cmake", "ninja", "make", "gmake", "c++", "g++", "clang++", "cc1plus", "cc1",
               "collect2", "ld", "ld.lld", "lto1", "lto-wrapper", "juceaide", "cl.exe", "link.exe", "MSBuild.exe"}


def owners(targets, build_roots):
    hits = []
    for proc in psutil.process_iter(["pid", "name", "exe", "cwd"]):
        try:
            info = proc.info
            if info["exe"] in targets or (info["name"] in BUILD_NAMES and info["cwd"]
                    and any(Path(info["cwd"]).is_relative_to(root) for root in build_roots)):
                hits.append(info)
            for opened in proc.open_files():
                if opened.path in targets:
                    hits.append({"pid": info["pid"], "path": opened.path})
            if info["pid"] != os.getpid():
                maps = Path("/proc") / str(info["pid"]) / "maps"
                if maps.is_file():
                    for line in maps.read_text().splitlines():
                        fields = line.split(maxsplit=5)
                        if len(fields) == 6 and fields[5] in targets:
                            hits.append({"pid": info["pid"], "mapped_path": fields[5]})
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, PermissionError):
            continue
    powershell = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    query = "$ErrorActionPreference='Stop'; $names=@('cl.exe','link.exe','MSBuild.exe','cmake.exe','ninja.exe','make.exe','g++.exe','clang++.exe'); @(Get-CimInstance Win32_Process | Where-Object { $_.Name -in $names } | Select-Object Name,ProcessId) | ConvertTo-Json -Compress"
    host = subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-Command", query],
                          capture_output=True, text=True, timeout=20)
    require(host.returncode == 0, "Windows build-owner query failed")
    windows = json.loads(host.stdout) if host.stdout.strip() else []
    if isinstance(windows, dict):
        windows = [windows]
    require(not hits and windows == [], "A build cache target or host compiler is active")
    return {"observed_utc": datetime.now(timezone.utc).isoformat(), "linux_target_users": hits,
            "windows_compiler_processes": windows, "windows_query_exit_code": host.returncode}


def git_states(build_roots):
    roots = {subprocess.check_output(["git", "-C", str(path), "rev-parse", "--show-toplevel"], text=True).strip()
             for path in build_roots}
    return {root: {"head": subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip(),
                   "status": subprocess.check_output(["git", "-C", root, "status", "--porcelain"], text=True)}
            for root in sorted(roots)}


def main():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Use the CPU workspace")
    require(OUT.is_dir() and not (OUT / "retirement-intent.json").exists()
            and not (OUT / "retirement-receipt.json").exists(), "Preserve previous retirement evidence")
    lock = (OUT / "retirement.lock").open("x")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inventory = read(OUT / "inventory.json")
    references = read(OUT / "historical-references.json")
    qualified = read(OUT / "allocation-qualification.json")
    require(inventory["candidate_bytes"] == 166_070_686 and len(inventory["candidates"]) == 95
            and qualified["status"] == "pass" and qualified["policy"] == policy()
            and qualified["all_standing_reservation_capacity_retained"]
            and references["inventory_sha256"] == sha(OUT / "inventory.json"), "Storage evidence differs")
    bindings = {**inventory["current_plan_bindings"], **qualified["source_bindings"]}
    for path in (Path(__file__).resolve(), OUT / "inventory.json", OUT / "historical-references.json",
                 OUT / "allocation-qualification.json"):
        bindings[str(path)] = sha(path)
    build_roots = [Path(path) for path in inventory["build_roots"]]
    states = git_states(build_roots)
    candidates = inventory["candidates"]
    for row in candidates:
        path = Path(row["path"])
        require(path.resolve() == path and path.is_file() and not path.is_symlink()
                and path.stat().st_nlink == 1 and path.stat().st_size == row["bytes"]
                and sha(path) == row["sha256"] and str(path) not in bindings
                and row["build_root"] in inventory["build_roots"]
                and path.is_relative_to(Path(row["build_root"]))
                and path.suffix.lower() in (".o", ".obj", ".a", ".lib"), "Generated cache identity changed")
        require(not subprocess.check_output(["git", "-C", row["build_root"], "ls-files", "--", str(path)], text=True),
                "A cache target is tracked source")
    annotations = []
    candidate_paths = {row["path"] for row in candidates}
    for reference in references["matching_documents"]:
        path = Path(reference["path"])
        require(sha(path) == reference["sha256"], "Historical cache annotation changed")
        value = read(path)
        locations = []
        def inspect(item, parents):
            if isinstance(item, dict):
                for key, child in item.items():
                    if key in candidate_paths:
                        locations.append({"path": key, "location": parents})
                    inspect(child, [*parents, key])
            elif isinstance(item, list):
                for index, child in enumerate(item):
                    inspect(child, [*parents, index])
            elif isinstance(item, str) and item in candidate_paths:
                locations.append({"path": item, "location": parents})
        inspect(value, [])
        require(all(row["location"] and row["location"][0] in
                    ("protected_files", "preserved_files", "protected_source_bindings", "targets", "removed")
                    for row in locations), "Generated cache remains a historical execution input")
        require("storage" in str(path) or "retirement" in str(path), "Unexpected historical cache consumer")
        annotations.append({**reference, "classified_locations": locations,
            "interpretation": "Earlier cleanup preservation or generated-build inventory; superseded only for this operation's selected cache files. Original receipt remains unchanged."})
        bindings[str(path)] = sha(path)
    for build in build_roots:
        for path in (build / "CMakeCache.txt", build.parent / "CMakeLists.txt"):
            require(path.is_file() and not path.is_symlink(), "Retain the generated build's configuration and source entry point")
            bindings[str(path)] = sha(path)
    verify_inputs({"source_bindings": bindings})
    budget_plan = {"storage_budget": read(PHASE / "branch-grouped-vocal-013/plan.json")["storage_budget"],
                   "weighted_storage": policy()}
    before = snapshot(budget_plan, require_room=False)
    require(before["headroom_after_complete_peak_bytes"] < 0, "Cache reclamation is not required")
    chosen, reclaimed = [], 0
    for row in candidates:
        chosen.append(row); reclaimed += row["bytes"]
        if before["projected_peak_bytes"] - reclaimed + RECORD_ALLOWANCE + POST_RECORD_MARGIN < before["authorized_cap_bytes"]:
            break
    require(chosen and before["projected_peak_bytes"] - reclaimed + RECORD_ALLOWANCE + POST_RECORD_MARGIN
            < before["authorized_cap_bytes"], "Generated caches do not cover the complete trial allocation")
    targets = {row["path"] for row in chosen}
    require(not targets.intersection(bindings), "A cache target is a current immutable input")
    protected = {}
    for build in build_roots:
        for path in build.rglob("*"):
            if path.is_file() and not path.is_symlink() and str(path) not in targets:
                protected[str(path)] = sha(path)
        print(json.dumps({"event": "build_non_targets_authenticated", "build": str(build),
                          "protected_file_count": len(protected)}), flush=True)
    live = owners(targets, build_roots)
    intent = {"schema": "latency58-quarter-vocal-generated-cache-retirement-v1",
              "observed_utc": datetime.now(timezone.utc).isoformat(), "executor_sha256": sha(__file__),
              "inventory_sha256": sha(OUT / "inventory.json"), "targets": chosen, "removed_bytes_planned": reclaimed,
              "protected_files": protected, "source_bindings": bindings, "tracked_worktree_states": states,
              "historical_cache_annotations": annotations, "live_process_check": live,
              "complete_trial_peak_before": before, "retirement_record_allowance_bytes": RECORD_ALLOWANCE,
              "post_record_headroom_margin_bytes": POST_RECORD_MARGIN, "trial_allocation": policy(),
              "models_optimizers_plugin_binaries_source_audio_sources_tests_or_logs_removed": False,
              "rebuild": inventory["rebuild"]}
    require(len((json.dumps(intent, indent=2, allow_nan=False) + "\n").encode()) < RECORD_ALLOWANCE - 20_000,
            "Retirement evidence exceeds its own allowance")
    require(git_states(build_roots) == states, "A worktree changed before cache retirement")
    write(OUT / "retirement-intent.json", intent)
    verify_inputs({"source_bindings": bindings})
    require(all(sha(path) == digest for path, digest in protected.items()), "Protected build artifact changed")
    owners(targets, build_roots)
    for row in chosen:
        path = Path(row["path"])
        require(path.resolve() == path and not path.is_symlink() and path.stat().st_nlink == 1
                and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "Cache changed immediately before removal")
        path.unlink()
    require(all(not Path(row["path"]).exists() for row in chosen), "Cache retirement incomplete")
    require(all(sha(path) == digest for path, digest in protected.items()), "A nonselected build file changed")
    verify_inputs({"source_bindings": bindings})
    require(git_states(build_roots) == states, "Tracked worktree state changed")
    after = snapshot(budget_plan)
    require(after["headroom_after_complete_peak_bytes"] > POST_RECORD_MARGIN,
            "Retirement did not leave the complete trial allocation and margin")
    receipt = {"schema": "latency58-quarter-vocal-generated-cache-retirement-result-v1", "status": "complete",
               "observed_utc": datetime.now(timezone.utc).isoformat(), "intent_sha256": sha(OUT / "retirement-intent.json"),
               "removed_files": len(chosen), "removed_bytes": reclaimed, "all_selected_files_absent": True,
               "all_preserved_files_unchanged": True, "protected_file_count": len(protected),
               "source_bindings_unchanged": True, "verified_source_binding_count": len(bindings),
               "tracked_worktree_states_unchanged": True, "complete_trial_peak_after": after,
               "all_models_optimizers_plugin_binaries_sources_tests_and_logs_preserved": True,
               "gpu_used": False}
    write(OUT / "retirement-receipt.json", receipt)
    print(json.dumps({"status": "complete", "removed_files": len(chosen), "removed_bytes": reclaimed,
                      "protected_files": len(protected), "projected_peak_bytes": after["projected_peak_bytes"],
                      "headroom_after_complete_peak_bytes": after["headroom_after_complete_peak_bytes"],
                      "receipt_sha256": sha(OUT / "retirement-receipt.json")}), flush=True)
    lock.close()


if __name__ == "__main__":
    main()
