"""Compare quiet wanted-source measurements without changing primary scores."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import statistics

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_sdr_checkpoint import require_space
from research.direct.evaluate_latency58_quiet_wanted import QUIET_BUCKETS


def load_completed(directory, evidence):
    paths = [directory / name for name in ("plan.json", "result.json", "execution.json")]
    plan, result, execution = [read(path) for path in paths]
    require(plan["schema"] == "latency58-quiet-wanted-evaluation-plan-v1"
            and result["schema"] == "latency58-quiet-wanted-evaluation-v1" and result["status"] == "pass"
            and result["all_primary_track_scores_and_streams_exact"] and result["all_1680_source_windows_exact"]
            and result["analytic_quiet_metric_check"]["status"] == "pass"
            and result["source_bindings"] == plan["source_bindings"] and result["source_bindings_unchanged"]
            and not result["cuda_initialized"] and not result["primary_protocol_changed"]
            and not result["confirmation_material_used"] and result["model"] == plan["model"]
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"]
            and execution["plan_sha256"] == result["plan_sha256"] == sha(paths[0]),
            "Quiet wanted-source evidence has not completed exactly")
    verify_inputs(plan)
    evidence.update(plan["source_bindings"])
    evidence.update({str(path): sha(path) for path in paths})
    return result


def compare_reports(reference, candidate):
    require(reference["reference_inventory"] == candidate["reference_inventory"], "Different source inventory")
    all_pairs = []
    for a, b in zip(reference["tracks"], candidate["tracks"], strict=True):
        require(a["track"] == b["track"] and a["track_index"] == b["track_index"], "Different primary track order")
        for x, y in zip(a["windows"], b["windows"], strict=True):
            source = {k: v for k, v in x.items() if k != "bands"}
            require(source == {k: v for k, v in y.items() if k != "bands"}, "Different quiet source or physical window")
            require(set(x["bands"]) == set(y["bands"]) == {"full", "low_20_250"}, "Different frequency bands")
            bands = {}
            for band in x["bands"]:
                left, right = x["bands"][band], y["bands"][band]
                require(left.keys() == right.keys() and left["reference_rms_dbfs"] == right["reference_rms_dbfs"],
                        "Different desired source level or measurement definition")
                pairs = {}
                for field in left:
                    u, v = left[field], right[field]
                    require((u is None) == (v is None)
                            and (u is None or (math.isfinite(u) and math.isfinite(v))), "Different metric support")
                    pairs[field] = {"reference": u, "candidate": v, "delta": None if u is None else v - u}
                u, v = left["signed_reference_projection_gain"], right["signed_reference_projection_gain"]
                pairs["absolute_projection_gain_error_from_one"] = {
                    "reference": None if u is None else abs(u - 1), "candidate": None if v is None else abs(v - 1),
                    "delta": None if u is None else abs(v - 1) - abs(u - 1)}
                bands[band] = pairs
            all_pairs.append({**source, "bands": bands})
    require(len(all_pairs) == 1680 and {r["track_index"] for r in all_pairs} == set(range(14)),
            "Incomplete matched window coverage")
    quiet = [row for row in all_pairs if row["bucket"] in QUIET_BUCKETS]
    require(len(quiet) == 61, "Unexpected quiet source support")
    summary = {}
    for stem in ("drums", "bass", "vocals", "other"):
        summary[stem] = {}
        for bucket in (*QUIET_BUCKETS, "all_minus80_to_minus50"):
            selected = [row for row in quiet if row["stem"] == stem
                        and (bucket == "all_minus80_to_minus50" or row["bucket"] == bucket)]
            tracks = sorted({row["track"] for row in selected})
            bands = {}
            for band in ("full", "low_20_250"):
                fields = {}
                for field in (() if not selected else selected[0]["bands"][band]):
                    per_track = {}
                    supported = [row for row in selected if row["bands"][band][field]["reference"] is not None]
                    for track in tracks:
                        values = [row["bands"][band][field] for row in supported if row["track"] == track]
                        if values:
                            per_track[track] = {"windows": len(values), **{
                                side: statistics.mean(v[side] for v in values) for side in ("reference", "candidate", "delta")}}
                    eligible = list(per_track.values())
                    extrema = [{"track": row["track"], "excerpt_index": row["excerpt_index"],
                                "physical_start": row["physical_start"], "physical_end": row["physical_end"],
                                **row["bands"][band][field]} for row in supported]
                    fields[field] = {"eligible_tracks": len(eligible), "eligible_windows": len(supported),
                                     "equal_track_means": {side: None if not eligible else statistics.mean(v[side] for v in eligible)
                                                           for side in ("reference", "candidate", "delta")},
                                     "per_track": per_track,
                                     "largest_increases": sorted(extrema, key=lambda r: r["delta"], reverse=True)[:5],
                                     "largest_decreases": sorted(extrema, key=lambda r: r["delta"])[:5]}
                bands[band] = fields
            summary[stem][bucket] = {"source_windows": len(selected), "source_tracks": len(tracks), "bands": bands}
    return {"reference_model": reference["model"], "candidate_model": candidate["model"],
            "all_1680_source_windows_matched": True, "quiet_window_pairs": quiet, "quiet_summaries": summary,
            "reference_inventory": reference["reference_inventory"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256 and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use a frozen CPU1 quiet-fidelity comparison")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-quiet-wanted-comparison-plan-v1", "Different comparison scope")
    verify_inputs(plan)
    evidence = {}
    reference = load_completed(Path(plan["reference_directory"]), evidence)
    candidate = load_completed(Path(plan["candidate_directory"]), evidence)
    require(reference["model"]["model_state_sha256"] != candidate["model"]["model_state_sha256"], "Same model on both sides")
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "result.json").exists(), "Preserve paired measurements")
    before = require_space(plan, 3_000_000)
    compared = compare_reports(reference, candidate)
    require(all(plan["source_bindings"].get(p) == s for p, s in evidence.items()), "Unbound completed measurements")
    verify_inputs(plan)
    write(out / "result.json", {"schema": "latency58-quiet-wanted-comparison-v1", "status": "pass",
          "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
          **compared, "counted_bytes_before": before, "counted_bytes_after": require_space(plan, 0),
          "inference_executed": False, "primary_protocol_changed": False, "quality_selected": False,
          "human_listening_completed": False, "confirmation_material_used": False,
          "limitations": ["Descriptive means and extrema on the fixed development panel; no uncertainty estimate or independent confirmation.",
                          "Sparse source support is explicit; some stems and level bins have only one or two tracks.",
                          "Nonzero source audio can include recorded bleed or noise, and projection gain can include correlated interference.",
                          "Lower native output alone does not establish wanted fidelity; read gain, SDR and error together."]})
    print({"status": "pass", "quiet_windows": 61, "reference": reference["model"]["kind"],
           "candidate": candidate["model"]["kind"]}, flush=True)


if __name__ == "__main__":
    main()
