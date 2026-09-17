"""Account explicitly for occupants of the next trial's artifact reservations."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, require
from research.direct.train_latency58 import disk_bytes

NAME = "branch-weighted-vocal-014"
MONITOR_ROOT = ROOT / "research/direct/runs/latency11/smoke/gpu-crash-followup"
PERMANENT_BYTES = 380_000_000
TRANSIENT_BYTES = 600_000_000
STANDING_DIAGNOSTIC_BYTES = 250_000_000
ADDITIONAL_DIAGNOSTIC_BYTES = 30_000_000
DIAGNOSTIC_BYTES = STANDING_DIAGNOSTIC_BYTES + ADDITIONAL_DIAGNOSTIC_BYTES
OUTSIDE_BYTES = 800_000_000
CAP_BYTES = 90_000_000_000
METADATA_SUFFIXES = {".json", ".jsonl", ".log", ".txt", ".md", ".py", ".lock"}


def policy():
    return {"schema": "latency58-quarter-vocal-explicit-storage-allocation-v1",
            "trial": NAME, "permanent_packed_generation_bytes": PERMANENT_BYTES,
            "standing_transient_save_bytes": TRANSIENT_BYTES,
            "standing_diagnostic_reservation_bytes": STANDING_DIAGNOSTIC_BYTES,
            "additional_diagnostic_reservation_bytes": ADDITIONAL_DIAGNOSTIC_BYTES,
            "maximum_trial_diagnostic_bytes": DIAGNOSTIC_BYTES,
            "other_outside_allowance_bytes": OUTSIDE_BYTES,
            "authorized_cap_bytes": CAP_BYTES,
            "diagnostic_roots": [str(PHASE / NAME), str(PHASE / "paired-vocal-weighted-014"),
                str(PHASE / "weighted-vocal-review-014"), str(MONITOR_ROOT / (NAME + "-resource")),
                str(MONITOR_ROOT / (NAME + "-production"))],
            "permanent_paths": [str(PHASE / NAME / "production-run/recovery.packed.pt"),
                                str(PHASE / NAME / "production-run/checkpoint.packed.pt")],
            "transient_path": str(PHASE / NAME / "production-run/recovery.packed.pending.pt"),
            "metadata_suffixes": sorted(METADATA_SUFFIXES),
            "allocation_rule": "Actual occupants plus unused capacity; prior artifacts never consume this trial's reservations",
            "quality_or_precision_change": False}


def peak_accounting(actual_bytes, *, permanent_occupied, transient_occupied, diagnostic_occupied):
    values = (actual_bytes, permanent_occupied, transient_occupied, diagnostic_occupied)
    require(all(type(value) is int and value >= 0 for value in values)
            and permanent_occupied <= PERMANENT_BYTES and transient_occupied <= PERMANENT_BYTES
            and diagnostic_occupied <= DIAGNOSTIC_BYTES
            and actual_bytes >= permanent_occupied + transient_occupied + diagnostic_occupied,
            "Invalid allocated artifact inventory or exceeded reservation")
    remaining = {"permanent_generation": PERMANENT_BYTES - permanent_occupied,
                 "transient_save": TRANSIENT_BYTES - transient_occupied,
                 "diagnostics": DIAGNOSTIC_BYTES - diagnostic_occupied,
                 "other_outside": OUTSIDE_BYTES}
    return {"actual_counted_and_external_git_bytes": actual_bytes,
            "occupied_permanent_generation_bytes": permanent_occupied,
            "occupied_transient_save_bytes": transient_occupied,
            "occupied_trial_diagnostic_bytes": diagnostic_occupied,
            "remaining_reserved_bytes": remaining,
            "projected_peak_bytes": actual_bytes + sum(remaining.values()),
            "authorized_cap_bytes": CAP_BYTES,
            "headroom_after_complete_peak_bytes": CAP_BYTES - actual_bytes - sum(remaining.values())}


def snapshot(plan, *, require_room=True):
    require(plan["weighted_storage"] == policy(), "Unprepared trial storage allocation")
    budget = plan["storage_budget"]
    require(budget["authorized_cap_bytes"] == CAP_BYTES
            and budget["live_training_save_reservation_bytes"] == TRANSIENT_BYTES
            and budget["diagnostic_artifact_allowance_bytes"] == STANDING_DIAGNOSTIC_BYTES
            and budget["other_outside_allowance_bytes"] == OUTSIDE_BYTES,
            "Standing storage reservations changed")
    counted_roots = [Path(path).resolve(strict=True) for path in budget["counted_roots"]]
    external = Path(budget["external_git_common_directory"]).resolve(strict=True)
    all_roots = [*counted_roots, external]
    require(all(not a.is_relative_to(b) for i, a in enumerate(all_roots)
                for j, b in enumerate(all_roots) if i != j), "Counted storage roots overlap")
    declared = policy()
    diagnostic_roots = [Path(path) for path in declared["diagnostic_roots"]]
    permanent_paths = {Path(path) for path in declared["permanent_paths"]}
    transient_path = Path(declared["transient_path"])
    require(all(path.resolve() == path and any(path.is_relative_to(root) for root in counted_roots)
                for path in diagnostic_roots)
            and all(not a.is_relative_to(b) for i, a in enumerate(diagnostic_roots)
                    for j, b in enumerate(diagnostic_roots) if i != j), "Allocated diagnostic roots overlap or escape accounting")
    metadata = {}
    permanent = {}; transient = 0
    for root in diagnostic_roots:
        if not root.exists():
            continue
        require(root.is_dir() and not root.is_symlink(), "Allocated diagnostic root must be a real directory")
        size = 0
        for path in root.rglob("*"):
            require(not path.is_symlink(), "Symlink inside allocated trial artifacts")
            if not path.is_file():
                continue
            require(path.resolve() == path, "Aliased trial artifact")
            count = path.stat().st_size
            if path in permanent_paths:
                permanent[str(path)] = count
            elif path == transient_path:
                transient += count
            else:
                require(path.suffix in METADATA_SUFFIXES, "Unallocated artifact type in diagnostic reservation: " + str(path))
                size += count
        metadata[str(root)] = size
    require(len(permanent) <= 1, "Packed finalization must not duplicate a permanent generation")
    counted = {str(root): disk_bytes(root) for root in counted_roots}
    git_bytes = disk_bytes(external)
    result = peak_accounting(sum(counted.values()) + git_bytes,
        permanent_occupied=sum(permanent.values()), transient_occupied=transient,
        diagnostic_occupied=sum(metadata.values()))
    result.update(observed_utc=datetime.now(timezone.utc).isoformat(), counted_roots=counted,
                  external_git_common_directory=str(external), external_git_common_bytes=git_bytes,
                  allocated_diagnostic_roots_bytes=metadata, allocated_permanent_files_bytes=permanent,
                  allocation=declared)
    if require_room:
        require(result["projected_peak_bytes"] < CAP_BYTES, "Trial peak exceeds the all-inclusive artifact cap")
    return result


def before_publication(directory, plan, serialized_bytes):
    require(Path(directory).resolve(strict=True) == PHASE / NAME / "production-run"
            and type(serialized_bytes) is int and 0 < serialized_bytes <= PERMANENT_BYTES,
            "Packed publication differs from the allocated generation")
    return snapshot(plan)
