"""Paired track comparisons for saved direct-evaluation reports."""

import argparse
import json
from pathlib import Path

import numpy as np

STEMS = ("drums", "bass", "vocals", "other")


def compare(reference, candidate, *, samples=20_000, seed=91):
    names = [track["name"] for track in reference["tracks"]]
    if names != [track["name"] for track in candidate["tracks"]]:
        raise ValueError("Comparisons require identical track order")
    # Different callback latencies have different estimate indices, but the
    # scored physical reference samples must be exactly the same.
    def intervals(result):
        return [[(row["id"], row["reference_start"], row["reference_end"])
                 for row in track["excerpts"]] for track in result["tracks"]]
    if intervals(reference) != intervals(candidate):
        raise ValueError("Comparisons require identical scored intervals")
    indices = np.random.default_rng(seed).integers(0, len(names), (samples, len(names)))
    selectors = {
        "full_sdr_db": lambda stem: stem["full_sdr_db"],
        "low_sdr_db": lambda stem: stem["band_sdr_db"]["low_20_250"],
        "bleed_sir_db": lambda stem: stem["sir_db"],
    }
    metrics = {}
    for name, select in selectors.items():
        arrays = [np.array([
            [select(track["per_stem"][source]) for source in STEMS]
            for track in result["tracks"]
        ], dtype=np.float64) for result in (reference, candidate)]
        if not np.array_equal(np.isfinite(arrays[0]), np.isfinite(arrays[1])):
            raise ValueError(f"Reference/candidate eligibility differs for {name}")
        differences = arrays[1] - arrays[0]
        per_stem = np.nanmean(differences, axis=0)
        # Resample whole tracks together, keeping stems and excerpts clustered.
        bootstraps = np.nanmean(np.nanmean(differences[indices], axis=1), axis=1)
        metrics[name] = {
            "reference": reference["aggregate"][name],
            "candidate": candidate["aggregate"][name],
            "delta": float(per_stem.mean()),
            "paired_track_bootstrap_95_percent": np.quantile(bootstraps, [.025, .975]).tolist(),
            "per_stem_delta": dict(zip(STEMS, per_stem.tolist())),
            "per_track_macro_delta": dict(zip(names, np.nanmean(differences, axis=1).tolist())),
        }
    return {
        "reference": reference["checkpoint"], "candidate": candidate["checkpoint"],
        "tracks": len(names), "bootstrap_samples": samples, "bootstrap_seed": seed,
        "uncertainty_scope": "Track sampling on this validation panel only; not training-seed uncertainty or correction for checkpoint selection.",
        "metrics": metrics,
        "absence_dbfs_delta": {
            source: candidate["aggregate"]["per_stem"][source]["absent_fp_dbfs"]
                    - reference["aggregate"]["per_stem"][source]["absent_fp_dbfs"]
            if candidate["aggregate"]["per_stem"][source]["absent_fp_dbfs"] is not None
            and reference["aggregate"]["per_stem"][source]["absent_fp_dbfs"] is not None else None
            for source in STEMS
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-index", type=int, default=0)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in (args.reference, args.candidate)]
    for field in ("manifest_sha256", "output_policy", "precision", "metrics"):
        if reports[0][field] != reports[1][field]:
            raise ValueError(f"Evaluation protocols differ: {field}")
    result = compare(reports[0]["results"][args.reference_index], reports[1]["results"][args.candidate_index])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: {key: value[key] for key in ("delta", "paired_track_bootstrap_95_percent")}
                      for name, value in result["metrics"].items()}, indent=2))


if __name__ == "__main__":
    main()
