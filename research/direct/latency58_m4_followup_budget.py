"""Count the new M4 allocation in addition to the frozen training forecast."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from research.direct.latency58_weighted_storage import snapshot as training_snapshot

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "research/direct/latency58_artifact_allowance.json"


def snapshot():
    policy = json.loads(POLICY.read_text())
    if policy["authorized_cap_bytes"] != 100_000_000_000:
        raise RuntimeError("Unexpected authorized ceiling")
    plan_path = ROOT / policy["active_training_plan"]
    data = plan_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != policy["active_training_plan_sha256"]:
        raise RuntimeError("Frozen training plan changed; revise combined accounting")
    base = training_snapshot(json.loads(data))
    counted = [Path(p).resolve() for p in base["counted_roots"]]
    counted.append(Path(base["external_git_common_directory"]).resolve())
    allocations = []
    roots = []
    for allocation in policy["additional_allocations"]:
        root = (ROOT / allocation["root"]).resolve()
        if any(root.is_relative_to(p) or p.is_relative_to(root) for p in [*counted, *roots]):
            raise RuntimeError("Additional allocation overlaps another counted root")
        roots.append(root)
        actual = 0
        for directory, dirs, files in os.walk(root, followlinks=False):
            for name in [*dirs, *files]:
                path = Path(directory) / name
                if path.is_symlink():
                    raise RuntimeError(f"Allocation has an uncounted symlink: {path}")
            actual += sum((Path(directory) / name).stat().st_size for name in files)
        reserved = allocation["reserved_peak_bytes"]
        if actual > reserved:
            raise RuntimeError("Additional M4 allocation exceeded")
        allocations.append({"root": str(root), "actual_bytes": actual,
                            "reserved_peak_bytes": reserved,
                            "unused_bytes": reserved - actual})
    peak = base["projected_peak_bytes"] + sum(v["reserved_peak_bytes"] for v in allocations)
    if peak > policy["authorized_cap_bytes"]:
        raise RuntimeError("Combined forecast exceeds user-authorized allowance")
    return {"observed_utc": datetime.now(timezone.utc).isoformat(),
            "authorized_cap_bytes": policy["authorized_cap_bytes"],
            "policy_sha256": hashlib.sha256(POLICY.read_bytes()).hexdigest(),
            "frozen_training_forecast": base,
            "additional_allocations": allocations,
            "combined_peak_bytes": peak,
            "headroom_bytes": policy["authorized_cap_bytes"] - peak}
