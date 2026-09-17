"""Inventory quiet source material on the unchanged primary physical intervals."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.report_latency58_vocal_focus import load_views
from research.direct.latency58_sdr_checkpoint import require_space


def level_bucket(dbfs, nonzero):
    if not nonzero:
        return "exact_zero"
    if dbfs > -50:
        return "above_activity_threshold"
    for lower in (-60, -70, -80):
        if dbfs > lower:
            return f"({lower},{lower + 10}]_dbfs"
    return "at_or_below_minus80_nonzero"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use a frozen CPU1 source inventory")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-quiet-reference-inventory-plan-v1", "Different inventory scope")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "result.json").exists(), "Preserve source inventory")
    before = require_space(plan, 2_000_000)
    evidence = {}
    views = load_views(Path(plan["reference_views_directory"]), evidence)
    template = read(Path(plan["reference_views_directory"]) / "plan.json")
    require(template["track_indices"] == list(range(14)), "Require the unchanged primary panel")
    require(all(plan["source_bindings"].get(p) == digest for p, digest in evidence.items()), "Unbound existing source geometry")
    import numpy as np
    import torch
    from research import evaluate as legacy
    from research.direct import evaluate as shared
    from research.metrics import SOURCE_ORDER, MetricConfig, frame_ranges, rms_dbfs
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest, config = read(template["manifest"]["path"]), read(template["config"]["path"])
    tracks, config = shared.select_panel(manifest, config, panel="full", track_indices=list(range(14)),
                                        excerpt_starts=None, duration=15.0, alignment_samples=128)
    metric = MetricConfig.from_mapping(config["metrics"])
    require(metric.activity_dbfs == -50 and metric.window_samples == metric.hop_samples == 44100,
            "Different source activity definition")
    records, counts, by_track = [], {stem: Counter() for stem in SOURCE_ORDER}, {}
    for track_index, (track, prior) in enumerate(zip(tracks, views["tracks"], strict=True)):
        intervals = legacy._reference_intervals(track, config)
        require(track["name"] == prior["name"] and intervals == prior["intervals"], "Changed physical source intervals")
        paths = [legacy._safe_dataset_path(Path(manifest["root"]), track["stems"][stem]) for stem in SOURCE_ORDER]
        require(all(plan["source_bindings"].get(str(p)) == sha(p) for p in paths), "Unbound original source audio")
        by_track[track["name"]] = {stem: Counter() for stem in SOURCE_ORDER}
        for excerpt_index, interval in enumerate(intervals):
            references = np.stack([legacy._read_excerpt(p, interval["reference_start"], interval["reference_end"],
                                     expected_frames=int(track["frames"])) for p in paths]).astype(np.float32)
            for window_index, (start, stop) in enumerate(frame_ranges(
                    references.shape[-1], metric.window_samples, metric.hop_samples)):
                require(stop - start == 44100, "Unexpected partial primary window")
                for stem_index, stem in enumerate(SOURCE_ORDER):
                    source = np.ascontiguousarray(references[stem_index, :, start:stop])
                    dbfs = rms_dbfs(source, metric.epsilon)
                    nonzero = int(np.count_nonzero(source))
                    bucket = level_bucket(dbfs, nonzero)
                    view = prior["views"]["vocals_only" if stem == "vocals" else "instrumental"]
                    previous = view["windows"][excerpt_index * 15 + window_index]
                    physical_start = int(interval["reference_start"]) + start
                    require(previous["physical_start"] == physical_start
                            and previous["physical_end"] == int(interval["reference_start"]) + stop
                            and previous["per_stem"][stem]["desired_active"] == (dbfs > metric.activity_dbfs),
                            "Source activity or window positions differ from the completed protocol")
                    records.append({"track": track["name"], "track_index": track_index, "stem": stem,
                                    "excerpt_index": excerpt_index, "physical_start": physical_start,
                                    "physical_end": int(interval["reference_start"]) + stop,
                                    "source_rms_dbfs": dbfs, "source_peak_abs": float(np.max(np.abs(source))),
                                    "nonzero_channel_samples": nonzero, "bucket": bucket,
                                    "source_float32_sha256": hashlib.sha256(source.tobytes()).hexdigest()})
                    counts[stem][bucket] += 1
                    by_track[track["name"]][stem][bucket] += 1
    require(len(records) == 1680 and all(sum(c.values()) == 420 for c in counts.values())
            and not torch.cuda.is_initialized(), "Incomplete source inventory or changed CPU scope")
    verify_inputs(plan)
    write(out / "result.json", {"schema": "latency58-quiet-reference-inventory-v1", "status": "pass",
          "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
          "counts_per_stem": counts, "counts_per_track": by_track, "source_windows": records,
          "source_activity_and_physical_intervals_exact": True, "reference_only": True,
          "training_updates_executed": 0, "inference_executed": False, "cuda_initialized": False,
          "confirmation_material_used": False, "primary_protocol_changed": False, "quality_selected": False,
          "counted_bytes_before": before, "counted_bytes_after": require_space(plan, 0),
          "limitations": ["Source assignment follows dataset stems; nonzero quiet audio can include recorded bleed or noise.",
                          "This inventory does not establish fidelity or audibility of any model output.",
                          "Level buckets describe source support only and do not change the primary activity threshold."]})
    print({"status": "pass", "counts_per_stem": counts, "source_windows": len(records)}, flush=True)


if __name__ == "__main__":
    main()
