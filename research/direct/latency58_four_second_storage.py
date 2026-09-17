"""Reserve the next research root on top of all frozen training and M4 reserves.

The old 014 allocation and the full 4 GB M4 allocation remain counted by the
existing auditor. This additional disjoint root receives its complete 2.5 GB
reservation even while it is empty; its current occupants are not added twice.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, require, sha
from research.direct.latency58_m4_followup_budget import snapshot as prior_snapshot

ARTIFACT_ROOT = ROOT / 'research/four_second_20260916'
POLICY = ROOT / 'research/direct/latency58_four_second_artifact_allowance.json'
RESERVED_PEAK_BYTES = 2_500_000_000


def combined_peak(prior_peak, occupied, *, reserved=RESERVED_PEAK_BYTES):
    require(type(prior_peak) is type(occupied) is type(reserved) is int
            and 0 <= occupied <= reserved and prior_peak >= 0 and reserved == RESERVED_PEAK_BYTES,
            'Invalid new research allocation or exceeded root reservation')
    value = prior_peak + reserved
    require(value <= 100_000_000_000, 'Combined artifact forecast exceeds the authorized 100 GB')
    return value


def snapshot():
    policy = json.loads(POLICY.read_text())
    require(policy['authorized_cap_bytes'] == 100_000_000_000
            and policy['new_root'] == str(ARTIFACT_ROOT)
            and policy['reserved_peak_bytes'] == RESERVED_PEAK_BYTES,
            'New research allocation changed')
    prior = prior_snapshot()
    roots = [Path(p).resolve() for p in prior['frozen_training_forecast']['counted_roots']]
    roots.append(Path(prior['frozen_training_forecast']['external_git_common_directory']).resolve())
    roots.extend(Path(v['root']).resolve() for v in prior['additional_allocations'])
    require(ARTIFACT_ROOT.resolve() == ARTIFACT_ROOT
            and all(not ARTIFACT_ROOT.is_relative_to(p) and not p.is_relative_to(ARTIFACT_ROOT) for p in roots),
            'New research root overlaps or aliases a previously counted root')
    occupied = 0
    if ARTIFACT_ROOT.exists():
        require(ARTIFACT_ROOT.is_dir() and not ARTIFACT_ROOT.is_symlink(), 'Invalid research root')
        for path in ARTIFACT_ROOT.rglob('*'):
            require(not path.is_symlink(), 'Uncounted symlink in new research artifacts')
            if path.is_file():
                occupied += path.stat().st_size
    total = combined_peak(prior['combined_peak_bytes'], occupied)
    return {'observed_utc': datetime.now(timezone.utc).isoformat(),
            'authorized_cap_bytes': 100_000_000_000, 'policy_sha256': sha(POLICY),
            'prior_training_and_m4_forecast': prior, 'new_root': str(ARTIFACT_ROOT),
            'new_root_actual_bytes': occupied, 'new_root_reserved_peak_bytes': RESERVED_PEAK_BYTES,
            'new_root_unused_reservation_bytes': RESERVED_PEAK_BYTES - occupied,
            'combined_peak_bytes': total, 'headroom_bytes': 100_000_000_000 - total}
